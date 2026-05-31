"""
MDD Challenge — Training entry point.

Wires together the data pipeline, the multimodal model, and the custom
training loop.  Run from the project root::

    python main.py
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader

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
    # canonical (linguistic) labels padded with 0  = [PAD] token id
    # transcript labels          padded with -100 = CTC ignore_index
    collator = DataCollatorCTCWithPadding(
        padding_value=0.0,
        transcript_pad_token_id=-100,
        canonical_pad_token_id=0,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_cfg: dict = config["training"]
    batch_size: int = int(train_cfg["batch_size"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
