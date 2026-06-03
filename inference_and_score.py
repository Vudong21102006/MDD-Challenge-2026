"""
Evaluation bridge — runs inference with a trained MDD model checkpoint
on the public test dataset and scores predictions using the official
challenge metrics (F1, PER, DER).

Usage::

    # Run on the public test dataset (default)
    python inference_and_score.py

    # Run on a custom CSV with a specific audio directory
    python inference_and_score.py --csv path/to/metadata.csv --audio-dir path/to/audio
"""

from __future__ import annotations

import argparse
import csv
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
VOCAB_PATH: str = "data/processed/vocab.json"

# Public test dataset paths
PUBLIC_TEST_DIR: str = "D:/Challenge2/MDD-Challenge-2026/data/MDD-Challenge-2025-public-test"
PUBLIC_TEST_CSV: str = "D:/Challenge2/MDD-Challenge-2026/data/MDD-Challenge-2025-public-test/metadata/public_test_phones.csv"

SAMPLE_RATE: int = 16000
BLANK_ID: int = 0
UNK_ID: int = 1
BATCH_SIZE: int = 8


# ── CTC decoder ──────────────────────────────────────────────────────────────

def greedy_ctc_decode(
    frame_logits: torch.Tensor, blank_id: int = BLANK_ID
) -> List[int]:
    """Greedy CTC decode frame-level logits into a phoneme token-ID list.

    1. ``argmax(dim=-1)`` → most-likely token per frame.
    2. Collapse consecutive identical token IDs.
    3. Remove *blank* tokens.
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
    rows: List[Dict[str, str]],
    vocab: Dict[str, int],
    device: torch.device,
    audio_base: str = "",
) -> List[str]:
    """Run inference over all samples and return decoded phoneme strings.

    Args:
        model:      Trained :class:`MDDModelBuilder`.
        rows:       List of CSV rows.  Must contain ``audio_filepath`` (or
                    ``path``) and ``canonical`` columns.
        vocab:      Phoneme token → ID mapping.
        device:     PyTorch device.
        audio_base: Directory prepended to relative audio paths.

    Returns:
        List of space-separated phoneme predictions, one per row.
    """
    id_to_token: Dict[int, str] = {idx: token for token, idx in vocab.items()}
    predictions: List[str] = []

    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start : start + BATCH_SIZE]

        # ── Load and pad audio ──────────────────────────────────────────
        waveforms: List[torch.Tensor] = []
        canonicals: List[List[int]] = []
        input_lengths: List[int] = []

        for row in batch_rows:
            # Resolve audio path — supports both "audio_filepath" and "path" columns
            raw_path = row.get("audio_filepath") or row.get("path", "")
            wav_path = _resolve_audio_path(raw_path, audio_base)
            wav, _ = load_and_resample(wav_path, target_sr=SAMPLE_RATE)
            wav = to_mono(wav).squeeze(0)                     # [T]
            waveforms.append(wav)
            input_lengths.append(wav.shape[0])

            # Tokenise canonical phoneme sequence (space-separated IPA)
            canonical_tokens = row["canonical"].strip().split()
            c_ids = [vocab.get(t, UNK_ID) for t in canonical_tokens]
            canonicals.append(torch.tensor(c_ids, dtype=torch.long))

        # Pad waveforms and canonical sequences to batch-max length
        input_values = pad_sequence(waveforms, batch_first=True).to(device)
        linguistic = pad_sequence(
            canonicals, batch_first=True, padding_value=0
        ).to(device)
        input_len_tensor = torch.tensor(
            input_lengths, dtype=torch.long, device=device
        )

        # ── Forward pass ────────────────────────────────────────────────
        logits = model(input_values, linguistic)                # [B, Tf, V]

        # Compute down-sampled frame count per sample
        feat_lengths = model.wav2vec2._get_feat_extract_output_lengths(
            input_len_tensor
        )

        # ── Decode each sample ──────────────────────────────────────────
        for b in range(logits.size(0)):
            valid_frames = feat_lengths[b].item()
            sample_logits = logits[b, :valid_frames, :]         # [Tf_valid, V]
            decoded_ids = greedy_ctc_decode(sample_logits, blank_id=BLANK_ID)
            pred_str = " ".join(
                id_to_token.get(tid, "[UNK]") for tid in decoded_ids
            )
            predictions.append(pred_str)

    return predictions


# ── Path helpers ─────────────────────────────────────────────────────────────

def _resolve_audio_path(raw_path: str, audio_base: str) -> str:
    """Resolve an audio file path, trying multiple strategies.

    1. If absolute → use as-is.
    2. If ``audio_base / raw_path`` exists → use it.
    3. If ``raw_path`` exists relative to CWD → use it.
    4. Otherwise → return ``audio_base / raw_path`` (let it fail with a
       clear FileNotFoundError).
    """
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return str(p)

    if audio_base:
        candidate = Path(audio_base) / raw_path
        if candidate.exists():
            return str(candidate)

    if p.exists():
        return str(p)

    # Last resort — will raise FileNotFoundError downstream if missing
    return str(Path(audio_base) / raw_path) if audio_base else str(p)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MDD inference on the public test dataset and score."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to a custom metadata CSV (overrides the public test default).",
    )
    parser.add_argument(
        "--audio-dir",
        default=None,
        help="Base directory for audio files (prepended to path column).",
    )
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Path to write the predictions CSV (default: predictions.csv).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a specific checkpoint (auto-detected if omitted).",
    )
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Configuration ─────────────────────────────────────────────────────
    config = load_config(CONFIG_PATH)
    vocab: Dict[str, int] = load_json(VOCAB_PATH)
    vocab_size: int = len(vocab)
    print(f"Vocabulary: {vocab_size} tokens")

    # ── Locate checkpoint ─────────────────────────────────────────────────
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = None
        for candidate in CHECKPOINT_CANDIDATES:
            if Path(candidate).exists():
                checkpoint_path = candidate
                break
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint found. Tried: {', '.join(CHECKPOINT_CANDIDATES)}. "
                "Run 'python main.py' to train a model first, "
                "or pass --checkpoint PATH."
            )
    print(f"Checkpoint: {checkpoint_path}")

    # ── Load model ────────────────────────────────────────────────────────
    model = MDDModelBuilder(config=config, vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "?")
        best_loss = checkpoint.get("best_val_loss", float("nan"))
        print(f"Loaded checkpoint (epoch {epoch}, val_loss={best_loss:.4f})")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # ── Data source ───────────────────────────────────────────────────────
    if args.csv:
        csv_path = args.csv
        audio_base = args.audio_dir or str(Path(csv_path).parent.parent)
    else:
        # Default: public test dataset
        csv_path = PUBLIC_TEST_CSV
        audio_base = args.audio_dir or PUBLIC_TEST_DIR

    print(f"Data CSV:   {csv_path}")
    print(f"Audio base: {audio_base}")

    # ── Load data ─────────────────────────────────────────────────────────
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = [
            row for row in csv.DictReader(f)
            if (row.get("audio_filepath") or row.get("path"))
        ]
    print(f"Samples:    {len(rows)}")

    # ── Run inference ─────────────────────────────────────────────────────
    predictions = run_inference(model, rows, vocab, device, audio_base=audio_base)
    print(f"Predictions generated: {len(predictions)}")

    # ── Write predictions.csv ─────────────────────────────────────────────
    has_id = "id" in rows[0] if rows else False
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "predict"])
        for i, pred in enumerate(predictions):
            row_id = rows[i]["id"] if has_id else i
            writer.writerow([row_id, pred])
    print(f"Wrote {args.output}")

    # ── Write ground_truth.csv ────────────────────────────────────────────
    gt_path = "ground_truth.csv"
    with open(gt_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["canonical", "transcript"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "canonical": row["canonical"],
                "transcript": row["transcript"],
            })
    print(f"Wrote {gt_path}")

    # ── Score ─────────────────────────────────────────────────────────────
    f1 = compute_f1(gt_path, args.output)
    per = compute_per(gt_path, args.output)
    der = compute_der(gt_path, args.output)

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