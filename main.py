"""
MDD Challenge — Training entry point.

Wires together the data pipeline, the multimodal model, and the custom
training loop.  Run from the project root::

    python main.py
"""

from __future__ import annotations

import csv
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.collator import DataCollatorCTCWithPadding
from src.data.dataset import PhonemeDataset
from src.models.builder import MDDModelBuilder
from src.training.train import MDDTrainer
from src.utils.file_utils import load_config


# ── Paths (relative to project root) ─────────────────────────────────────────
TRAIN_CSV: str = "data/processed/train_split.csv"
VAL_CSV: str = "data/processed/val_split.csv"
VOCAB_PATH: str = "data/processed/vocab.json"
CONFIG_PATH: str = "config.yaml"

# Weight multiplier for mispronounced samples in the weighted sampler
MISPRONUNCIATION_WEIGHT: float = 5.0


def _compute_sample_weights(csv_path: str) -> Tuple[list, int, int]:
    """Assign higher sampling weights to mispronounced utterances.

    Reads the split CSV and compares each row's ``canonical`` and
    ``transcript`` columns.  Samples where the two differ receive
    ``MISPRONUNCIATION_WEIGHT``; correct-pronunciation samples receive
    weight 1.0.

    Returns:
        ``(weights, num_correct, num_mispronounced)``.
    """
    weights: list = []
    correct = 0
    mis = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["canonical"].strip() == row["transcript"].strip():
                weights.append(1.0)
                correct += 1
            else:
                weights.append(MISPRONUNCIATION_WEIGHT)
                mis += 1

    return weights, correct, mis


def _build_loaders(
    config: dict,
    vocab_path: str,
) -> Tuple[DataLoader, DataLoader, int]:
    """Instantiate datasets, collator, and DataLoaders.

    Returns:
        ``(train_loader, dev_loader, vocab_size)``.
    """
    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = PhonemeDataset(
        split_csv=TRAIN_CSV,
        vocab_path=vocab_path,
        add_noise=True,
        noise_prob=0.3,
    )

    val_dataset = PhonemeDataset(
        split_csv=VAL_CSV,
        vocab_path=vocab_path,
        add_noise=False,
    )

    vocab_size: int = len(train_dataset.tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size}")

    # ── Collator ──────────────────────────────────────────────────────────
    collator = DataCollatorCTCWithPadding(
        padding_value=0.0,
        transcript_pad_token_id=-100,
        canonical_pad_token_id=0,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_cfg: dict = config["training"]
    batch_size: int = int(train_cfg["batch_size"])

    # Weighted sampling — mispronounced samples get 5× more exposure
    sample_weights, num_correct, num_mis = _compute_sample_weights(TRAIN_CSV)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    print(
        f"Train samples: {len(sample_weights)} "
        f"(correct={num_correct}, mispronounced={num_mis})"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    dev_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(dev_loader)}")
    return train_loader, dev_loader, vocab_size


def main() -> None:
    """Load config, build model & data, and launch the training loop."""
    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Configuration ─────────────────────────────────────────────────────
    config = load_config(CONFIG_PATH)

    # ── Data pipeline ─────────────────────────────────────────────────────
    train_loader, dev_loader, vocab_size = _build_loaders(config, VOCAB_PATH)

    # ── Model ─────────────────────────────────────────────────────────────
    model = MDDModelBuilder(config=config, vocab_size=vocab_size)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = MDDTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
    )

    # ── Go! ───────────────────────────────────────────────────────────────
    trainer.train()


if __name__ == "__main__":
    main()
