"""
Evaluation bridge — runs inference with a trained MDD model checkpoint
and scores predictions using the official challenge metrics (F1, PER, DER).

Usage::

    python inference_and_score.py

Requires a trained checkpoint at ``experiments/checkpoints/best_model.pt``
(created by ``python main.py``) and a preprocessed validation split at
``data/processed/val_split.csv``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from src.models.builder import MDDModelBuilder
from src.utils.audio_utils import load_and_resample, to_mono
from src.utils.evaluate import compute_der, compute_f1, compute_per
from src.utils.file_utils import load_config, load_json

# ── Constants ────────────────────────────────────────────────────────────────
CONFIG_PATH: str = "config.yaml"
CHECKPOINT_CANDIDATES: Tuple[str, ...] = (
    "experiments/checkpoints/best_model.pt",
    "checkpoints/best_model.pt",
)
VAL_CSV: str = "data/processed/val_split.csv"
VOCAB_PATH: str = "data/processed/vocab.json"
SAMPLE_RATE: int = 16000
BLANK_ID: int = 0
UNK_ID: int = 1
BATCH_SIZE: int = 8


# ── CTC decoder ──────────────────────────────────────────────────────────────

def greedy_ctc_decode(frame_logits: torch.Tensor, blank_id: int = BLANK_ID) -> List[int]:
    """Greedy CTC decode a single sample's frame-level logits.

    Steps:
      1. ``argmax(dim=-1)`` → most-likely token per frame.
      2. Collapse consecutive identical token IDs.
      3. Remove *blank* tokens.

    Args:
        frame_logits: Logits tensor of shape ``[T, V]`` (time × vocab).
        blank_id: Token ID treated as the CTC blank (default: 0).

    Returns:
        List of decoded token IDs (without blanks).
    """
    pred_ids: List[int] = torch.argmax(frame_logits, dim=-1).tolist()

    collapsed: List[int] = []
    prev = None
    for token_id in pred_ids:
        if token_id != prev and token_id != blank_id:
            collapsed.append(token_id)
        prev = token_id

    return collapsed


# ── Batch inference ──────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: MDDModelBuilder,
    val_rows: List[Dict[str, str]],
    vocab: Dict[str, int],
    device: torch.device,
) -> List[str]:
    """Run inference over all validation samples and return decoded phoneme strings.

    Processes samples in mini-batches.  Audio waveforms and canonical
    sequences are padded independently per batch to the batch-maximum
    length.
    """
    id_to_token: Dict[int, str] = {idx: token for token, idx in vocab.items()}
    predictions: List[str] = []

    for start in range(0, len(val_rows), BATCH_SIZE):
        batch_rows = val_rows[start : start + BATCH_SIZE]

        # ── Load and pad audio ──────────────────────────────────────────
        waveforms: List[torch.Tensor] = []
        canonicals: List[List[int]] = []
        input_lengths: List[int] = []

        for row in batch_rows:
            wav, _ = load_and_resample(row["audio_filepath"], target_sr=SAMPLE_RATE)
            wav = to_mono(wav).squeeze(0)                     # [T]
            waveforms.append(wav)
            input_lengths.append(wav.shape[0])

            # Tokenise canonical phoneme sequence
            canonical_tokens = row["canonical"].strip().split()
            c_ids = [vocab.get(t, UNK_ID) for t in canonical_tokens]
            canonicals.append(torch.tensor(c_ids, dtype=torch.long))

        # Pad waveforms to batch-max length
        input_values = pad_sequence(waveforms, batch_first=True).to(device)
        # Pad canonical sequences to batch-max length (0 = [PAD] ID)
        linguistic = pad_sequence(canonicals, batch_first=True, padding_value=0).to(device)
        input_len_tensor = torch.tensor(input_lengths, dtype=torch.long, device=device)

        # ── Forward pass ────────────────────────────────────────────────
        logits = model(input_values, linguistic)                # [B, Tf, V]

        # Compute down-sampled frame count per sample
        feat_lengths = model.wav2vec2._get_feat_extract_output_lengths(input_len_tensor)

        # ── Decode each sample ──────────────────────────────────────────
        for b in range(logits.size(0)):
            valid_frames = feat_lengths[b].item()
            sample_logits = logits[b, :valid_frames, :]         # [Tf_valid, V]
            decoded_ids = greedy_ctc_decode(sample_logits, blank_id=BLANK_ID)
            pred_str = " ".join(id_to_token.get(tid, "[UNK]") for tid in decoded_ids)
            predictions.append(pred_str)

    return predictions


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Configuration ─────────────────────────────────────────────────────
    config = load_config(CONFIG_PATH)
    vocab: Dict[str, int] = load_json(VOCAB_PATH)
    vocab_size: int = len(vocab)
    print(f"Vocabulary: {vocab_size} tokens")

    # ── Locate checkpoint ─────────────────────────────────────────────────
    checkpoint_path = None
    for candidate in CHECKPOINT_CANDIDATES:
        if Path(candidate).exists():
            checkpoint_path = candidate
            break
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found. Tried: {', '.join(CHECKPOINT_CANDIDATES)}. "
            "Run 'python main.py' to train a model first."
        )
    print(f"Checkpoint: {checkpoint_path}")

    # ── Load model ────────────────────────────────────────────────────────
    model = MDDModelBuilder(config=config, vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both raw state_dict and trainer-style checkpoint dicts
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "?")
        best_loss = checkpoint.get("best_val_loss", float("nan"))
        print(f"Loaded checkpoint (epoch {epoch}, val_loss={best_loss:.4f})")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # ── Load validation data ──────────────────────────────────────────────
    with open(VAL_CSV, "r", encoding="utf-8") as f:
        val_rows = [
            row for row in csv.DictReader(f) if row.get("audio_filepath")
        ]
    print(f"Validation samples: {len(val_rows)}")

    # ── Run inference ─────────────────────────────────────────────────────
    predictions = run_inference(model, val_rows, vocab, device)
    print(f"Predictions generated: {len(predictions)}")

    # ── Write predictions.csv (columns: id, predict) ─────────────────────
    with open("predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "predict"])
        for idx, pred in enumerate(predictions):
            writer.writerow([idx, pred])
    print("Wrote predictions.csv")

    # ── Write ground_truth.csv (columns: canonical, transcript) ───────────
    with open("ground_truth.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["canonical", "transcript"])
        writer.writeheader()
        for row in val_rows:
            writer.writerow(
                {
                    "canonical": row["canonical"],
                    "transcript": row["transcript"],
                }
            )
    print("Wrote ground_truth.csv")

    # ── Score ─────────────────────────────────────────────────────────────
    f1 = compute_f1("ground_truth.csv", "predictions.csv")
    per = compute_per("ground_truth.csv", "predictions.csv")
    der = compute_der("ground_truth.csv", "predictions.csv")

    # ── Report ────────────────────────────────────────────────────────────
    print()
    print("=" * 45)
    print("  MDD Challenge — Evaluation Results")
    print("=" * 45)
    print(f"  F1  │ Mispronunciation Detection   {f1:.4f}")
    print(f"  PER │ Phoneme Error Rate           {per:.4f}")
    print(f"  DER │ Diagnosis Error Rate         {der:.4f}")
    print("=" * 45)


if __name__ == "__main__":
    main()
