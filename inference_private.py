from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from src.models.builder import MDDModelBuilder
from src.utils.audio_utils import load_and_resample, to_mono
from src.utils.file_utils import load_config, load_json

# ── Constants ────────────────────────────────────────────────────────────────
CONFIG_PATH       = "config.yaml"
VOCAB_PATH        = "data/processed/vocab.json"

PRIVATE_TEST_CSV     = "data/MDD-Challenge-2025-private-test/metadata/private_test_submission.csv"
PRIVATE_TEST_DIR    = "data/MDD-Challenge-2025-private-test"

OUTPUT_CSV_PATH   = "results.csv"

CHECKPOINT_CANDIDATES: Tuple[str, ...] = (
    "experiments/checkpoints/best_model.pt",
    "checkpoints/best_model.pt",
)
# =============================================================================

SAMPLE_RATE: int = 16000
BLANK_ID: int = 0
UNK_ID: int = 1
BATCH_SIZE: int = 8

def greedy_ctc_decode(frame_logits: torch.Tensor, blank_id: int = BLANK_ID) -> List[int]:
    pred_ids: List[int] = torch.argmax(frame_logits, dim=-1).tolist()
    collapsed: List[int] = []
    prev = None
    for token_id in pred_ids:
        if token_id != prev and token_id != blank_id:
            collapsed.append(token_id)
        prev = token_id
    return collapsed

@torch.no_grad()
def run_inference(model: MDDModelBuilder, rows: List[Dict[str, str]], vocab: Dict[str, int], device: torch.device) -> List[str]:
    id_to_token: Dict[int, str] = {idx: token for token, idx in vocab.items()}
    predictions: List[str] = []

    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start : start + BATCH_SIZE]
        waveforms: List[torch.Tensor] = []
        canonicals: List[List[int]] = []
        input_lengths: List[int] = []

        for row in batch_rows:
            raw_path = row.get("audio_filepath") or row.get("path", "")
            wav_path = _resolve_audio_path(raw_path, PRIVATE_TEST_DIR)
            wav, _ = load_and_resample(wav_path, target_sr=SAMPLE_RATE)
            wav = to_mono(wav).squeeze(0)                     
            waveforms.append(wav)
            input_lengths.append(wav.shape[0])

            canonical_tokens = row.get("canonical", "").strip().split()
            c_ids = [vocab.get(t, UNK_ID) for t in canonical_tokens] if canonical_tokens else [UNK_ID]
            canonicals.append(torch.tensor(c_ids, dtype=torch.long))

        input_values = pad_sequence(waveforms, batch_first=True).to(device)
        linguistic = pad_sequence(canonicals, batch_first=True, padding_value=0).to(device)
        input_len_tensor = torch.tensor(input_lengths, dtype=torch.long, device=device)

        outputs = model(input_values, linguistic)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        feat_lengths = model.wav2vec2._get_feat_extract_output_lengths(input_len_tensor)

        for b in range(logits.size(0)):
            valid_frames = feat_lengths[b].item()
            sample_logits = logits[b, :valid_frames, :]         
            decoded_ids = greedy_ctc_decode(sample_logits, blank_id=BLANK_ID)
            pred_str = " ".join(id_to_token.get(tid, "[UNK]") for tid in decoded_ids)
            predictions.append(pred_str)

    return predictions

def _resolve_audio_path(raw_path: str, audio_base: str) -> str:
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return str(p)
    if audio_base:
        candidate = Path(audio_base) / raw_path
        if candidate.exists():
            return str(candidate)
    if p.exists():
        return str(p)
    return str(Path(audio_base) / raw_path) if audio_base else str(p)

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = load_config(CONFIG_PATH)
    vocab = load_json(VOCAB_PATH)

       # ── Load model ────────────────────────────────────────────────────────

    checkpoint_path = None
    for candidate in CHECKPOINT_CANDIDATES:
        if Path(candidate).exists():
            checkpoint_path = candidate
            break
            
    if checkpoint_path is None:
        raise FileNotFoundError("Checkpoint not found: {checkpoint_path}")
    print(f"Checkpoint: {checkpoint_path}")

    model = MDDModelBuilder(config=config, vocab_size=len(vocab))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()

   # ── Load data ─────────────────────────────────────────────────────────
    if not os.path.exists(PRIVATE_TEST_CSV):
        raise FileNotFoundError(f"Dataset  not found: {PRIVATE_TEST_CSV}")
        
    with open(PRIVATE_TEST_CSV, "r", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if (row.get("audio_filepath") or row.get("path"))]
    print(f"Sample: {len(rows)}")

    # ── Run inference ─────────────────────────────────────────────────────
    predictions = run_inference(model, rows, vocab, device)

# ── Write results.csv ─────────────────────────────────────────────
    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "path", "predict"])
        for i, pred in enumerate(predictions):
            row_id = rows[i]["id"]
            row_path = rows[i].get("path", rows[i].get("audio_filepath", ""))
            writer.writerow([row_id, row_path, pred])
    print(f"Wrote: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()