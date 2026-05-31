from __future__ import annotations
import os
from typing import Any, Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


class MDDTrainer:
    """High-performance trainer for the MDD multimodal model.

    Orchestrates training with Automatic Mixed Precision (AMP), gradient
    accumulation, linear learning-rate warmup + decay, and CTC loss
    optimisation.  Checkpoints are saved whenever the validation loss
    reaches a new minimum.

    Args:
        config:
            Full configuration dictionary (parsed from ``config.yaml``).
            The ``config['training']`` sub-dict must contain:
            ``learning_rate``, ``epochs``, ``warmup_ratio``,
            ``weight_decay``, ``fp16``, ``gradient_accumulation_steps``.
        model:
            An instance of :class:`MDDModelBuilder` (or any ``nn.Module``
            whose ``forward(input_values, linguistic)`` returns logits).
        train_loader:
            Training DataLoader. Each batch is expected to yield
            ``(input_values, linguistic, transcript, target_lengths,
            input_lengths)``.
        dev_loader:
            Validation DataLoader with the same batch format.
        device:
            PyTorch device to run on (e.g. ``torch.device('cuda')``).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.device = device

        # ── Parse training hyper-parameters ────────────────────────────────
        train_cfg: Dict[str, Any] = config["training"]

        self.learning_rate: float = float(train_cfg["learning_rate"])
        self.epochs: int = int(train_cfg["epochs"])
        self.warmup_ratio: float = float(train_cfg["warmup_ratio"])
        self.weight_decay: float = float(train_cfg["weight_decay"])
        self.fp16: bool = bool(train_cfg.get("fp16", False))
        self.gradient_accumulation_steps: int = int(
            train_cfg.get("gradient_accumulation_steps", 1)
        )
        self.metric_for_best_model: str = train_cfg.get(
            "metric_for_best_model", "loss"
        )
        self.load_best_model_at_end: bool = bool(
            train_cfg.get("load_best_model_at_end", True)
        )

        # ── Optimiser ──────────────────────────────────────────────────────
        # Do not apply weight decay to bias and LayerNorm parameters
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
        )

        # ── LR Scheduler ───────────────────────────────────────────────────
        steps_per_epoch: int = len(self.train_loader)
        self.num_training_steps: int = (
            (steps_per_epoch + self.gradient_accumulation_steps - 1)
            // self.gradient_accumulation_steps
        ) * self.epochs
        num_warmup_steps: int = int(
            self.num_training_steps * self.warmup_ratio
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        # ── Mixed Precision ────────────────────────────────────────────────
        self.scaler = torch.amp.GradScaler(device='cuda', enabled=self.fp16)

        # ── CTC Loss ───────────────────────────────────────────────────────
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

        # ── Checkpointing ──────────────────────────────────────────────────
        self.checkpoint_dir: str = config.get(
            "checkpoint_dir", "experiments/checkpoints"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path: str = os.path.join(
            self.checkpoint_dir, "best_model.pt"
        )
        self.best_val_loss: float = float("inf")

        # Move model to device
        self.model = self.model.to(self.device)

    # ── Public API ───────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop across all epochs.

        After each epoch the model is evaluated on the validation set.
        If the validation loss improves, a checkpoint is saved.  When
        ``load_best_model_at_end`` is ``True`` the best checkpoint is
        reloaded before returning.
        """
        print(f"Device: {self.device}")
        print(
            f"Training steps: {self.num_training_steps} | "
            f"Epochs: {self.epochs} | "
            f"Gradient accumulation: {self.gradient_accumulation_steps} | "
            f"FP16: {self.fp16}"
        )
        self._print_trainable_parameters()

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val_loss = self._evaluate()

            print(
                f"Epoch {epoch:3d}/{self.epochs} — "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            # ── Checkpoint on best validation loss ─────────────────────
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)
                print(f"  ↳ New best model saved (val_loss={val_loss:.4f})")

        # ── Reload best checkpoint ─────────────────────────────────────
        if self.load_best_model_at_end and os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            print(
                f"Loaded best checkpoint with "
                f"val_loss={self.best_val_loss:.4f}"
            )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _train_one_epoch(self, epoch: int) -> float:
        """Execute a single training epoch.

        Args:
            epoch: Current epoch number (1-indexed, for logging).

        Returns:
            Average training loss over all batches in the epoch.
        """
        self.model.train()
        running_loss: float = 0.0

        for step, batch in enumerate(self.train_loader):
            (
                input_values, linguistic, transcript,
                target_lengths, input_lengths, attention_mask,
            ) = self._batch_to_device(batch)

            with torch.amp.autocast(device_type='cuda', enabled=self.fp16):
                # Forward pass
                logits = self.model(
                    input_values, linguistic, attention_mask=attention_mask
                )  # [B, T, V]

                # Prepare logits for CTC: log_softmax + [B, T, V] → [T, B, V]
                logits = logits.log_softmax(dim=2).transpose(0, 1)

                # Compute CTC-compatible input lengths from raw waveform lengths
                ctc_input_lengths = (
                    self.model.wav2vec2._get_feat_extract_output_lengths(
                        input_lengths
                    )
                )

                loss = self.ctc_loss(
                    logits, transcript, ctc_input_lengths, target_lengths
                )
                loss = loss / self.gradient_accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Step optimiser & scheduler after accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            running_loss += loss.item() * self.gradient_accumulation_steps

        # Handle any remaining gradients at end of epoch
        if (step + 1) % self.gradient_accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Evaluate the model on the validation set.

        Returns:
            Average validation CTC loss.
        """
        self.model.eval()
        total_loss: float = 0.0

        for batch in self.dev_loader:
            (
                input_values, linguistic, transcript,
                target_lengths, input_lengths, attention_mask,
            ) = self._batch_to_device(batch)

            logits = self.model(
                input_values, linguistic, attention_mask=attention_mask
            )  # [B, T, V]
            logits = logits.log_softmax(dim=2).transpose(0, 1)

            ctc_input_lengths = (
                self.model.wav2vec2._get_feat_extract_output_lengths(
                    input_lengths
                )
            )

            loss = self.ctc_loss(
                logits, transcript, ctc_input_lengths, target_lengths
            )
            total_loss += loss.item()

        return total_loss / len(self.dev_loader)

    def _batch_to_device(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Move every tensor in a batch tuple to the configured device."""
        return tuple(
            t.to(self.device) if isinstance(t, torch.Tensor) else t
            for t in batch
        )

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Persist model weights and metadata to disk."""
        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": val_loss,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def _print_trainable_parameters(self) -> None:
        """Log a summary of trainable vs. total parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        backbone_total = sum(
            p.numel() for p in self.model.wav2vec2.parameters()
        )
        backbone_trainable = sum(
            p.numel()
            for p in self.model.wav2vec2.parameters()
            if p.requires_grad
        )
        print(
            f"Trainable params: {trainable:,}/{total:,} | "
            f"wav2vec2 trainable: {backbone_trainable:,}/{backbone_total:,}"
        )