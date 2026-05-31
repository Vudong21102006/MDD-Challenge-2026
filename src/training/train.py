from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


class MDDTrainer:
    """High-performance trainer for the MDD multimodal model.

    Orchestrates training with Automatic Mixed Precision (AMP), gradient
    accumulation, linear learning-rate warmup + decay, a joint **CTC +
    detection** loss, and gated-fusion mispronunciation detection.
    Checkpoints are saved whenever the combined validation loss reaches a
    new minimum.

    Args:
        config:  Full configuration dictionary (parsed from ``config.yaml``).
        model:   :class:`MDDModelBuilder` whose ``forward(…,
                 return_detection=True)`` returns ``(logits, detection)``.
        train_loader:  Training DataLoader yielding the 6-tuple produced by
                       :class:`DataCollatorCTCWithPadding`.
        dev_loader:    Validation DataLoader (same format).
        device:        PyTorch device to run on.
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
        # Weight for the detection (BCE) loss relative to CTC loss
        self.detection_lambda: float = float(
            train_cfg.get("detection_lambda", 0.3)
        )

        # ── Optimiser ──────────────────────────────────────────────────────
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
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.fp16)

        # ── Losses ─────────────────────────────────────────────────────────
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        self.det_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

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
        """Run the full training loop across all epochs."""
        print(f"Device: {self.device}")
        print(
            f"Training steps: {self.num_training_steps} | "
            f"Epochs: {self.epochs} | "
            f"Gradient accumulation: {self.gradient_accumulation_steps} | "
            f"FP16: {self.fp16} | "
            f"Detection λ: {self.detection_lambda}"
        )
        self._print_trainable_parameters()

        for epoch in range(1, self.epochs + 1):
            train_ctc, train_det = self._train_one_epoch(epoch)
            val_ctc, val_det = self._evaluate()
            val_combined = val_ctc + self.detection_lambda * val_det

            print(
                f"Epoch {epoch:3d}/{self.epochs} — "
                f"Train CTC: {train_ctc:.4f}  Det: {train_det:.4f} | "
                f"Val CTC: {val_ctc:.4f}  Det: {val_det:.4f}  "
                f"Combined: {val_combined:.4f}"
            )

            # ── Checkpoint on best combined validation loss ─────────────
            if val_combined < self.best_val_loss:
                self.best_val_loss = val_combined
                self._save_checkpoint(epoch, val_ctc, val_det)
                print(f"  ↳ New best model saved (combined={val_combined:.4f})")

        # ── Reload best checkpoint ─────────────────────────────────────
        if self.load_best_model_at_end and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Loaded best checkpoint with "
                f"combined_val_loss={self.best_val_loss:.4f}"
            )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """Execute a single training epoch.

        Returns:
            ``(avg_ctc_loss, avg_detection_loss)`` over all batches.
        """
        self.model.train()
        running_ctc: float = 0.0
        running_det: float = 0.0

        for step, batch in enumerate(self.train_loader):
            (
                input_values, linguistic, transcript,
                target_lengths, input_lengths, attention_mask,
            ) = self._batch_to_device(batch)

            with torch.amp.autocast(device_type="cuda", enabled=self.fp16):
                # Forward pass — returns both phoneme logits and detection scores
                logits, detection = self.model(
                    input_values, linguistic,
                    attention_mask=attention_mask,
                    return_detection=True,
                )  # logits: [B, Tf, V]  |  detection: [B, Tf]

                # ── CTC loss ────────────────────────────────────────────
                ctc_logits = logits.log_softmax(dim=2).transpose(0, 1)
                ctc_input_lengths = (
                    self.model.wav2vec2._get_feat_extract_output_lengths(
                        input_lengths
                    )
                )
                ctc_loss = self.ctc_loss_fn(
                    ctc_logits, transcript, ctc_input_lengths, target_lengths
                )

                # ── Detection (BCE) loss ────────────────────────────────
                det_targets = self._build_detection_targets(
                    linguistic, transcript, input_lengths
                )
                det_mask = self._build_frame_mask(detection, ctc_input_lengths)
                det_loss = (
                    self.det_loss_fn(detection, det_targets) * det_mask
                ).sum() / det_mask.sum().clamp(min=1)

                # ── Combined loss ───────────────────────────────────────
                loss = (
                    ctc_loss + self.detection_lambda * det_loss
                ) / self.gradient_accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Step optimiser & scheduler after accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            running_ctc += ctc_loss.item()
            running_det += det_loss.item()

        # Handle any remaining gradients at end of epoch
        if (step + 1) % self.gradient_accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        n = len(self.train_loader)
        return running_ctc / n, running_det / n

    @torch.no_grad()
    def _evaluate(self) -> Tuple[float, float]:
        """Evaluate the model on the validation set.

        Returns:
            ``(avg_ctc_loss, avg_detection_loss)``.
        """
        self.model.eval()
        total_ctc: float = 0.0
        total_det: float = 0.0

        for batch in self.dev_loader:
            (
                input_values, linguistic, transcript,
                target_lengths, input_lengths, attention_mask,
            ) = self._batch_to_device(batch)

            logits, detection = self.model(
                input_values, linguistic,
                attention_mask=attention_mask,
                return_detection=True,
            )

            # CTC loss
            ctc_logits = logits.log_softmax(dim=2).transpose(0, 1)
            ctc_input_lengths = (
                self.model.wav2vec2._get_feat_extract_output_lengths(
                    input_lengths
                )
            )
            ctc_loss = self.ctc_loss_fn(
                ctc_logits, transcript, ctc_input_lengths, target_lengths
            )

            # Detection loss
            det_targets = self._build_detection_targets(
                linguistic, transcript, input_lengths
            )
            det_mask = self._build_frame_mask(detection, ctc_input_lengths)
            det_loss = (
                self.det_loss_fn(detection, det_targets) * det_mask
            ).sum() / det_mask.sum().clamp(min=1)

            total_ctc += ctc_loss.item()
            total_det += det_loss.item()

        n = len(self.dev_loader)
        return total_ctc / n, total_det / n

    # ── Detection label helpers ──────────────────────────────────────────────

    def _build_detection_targets(
        self,
        linguistic: torch.Tensor,
        transcript: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Build frame-level binary mismatch labels.

        Compares *transcript* (what the student said) against *canonical*
        (what they should have said) phoneme-by-phoneme, then stretches the
        phoneme-level labels to the Wav2Vec2 frame grid via uniform
        stretching.

        Args:
            linguistic:    Canonical phoneme IDs, padded with 0  ``[B, Nc]``.
            transcript:    Transcript phoneme IDs, padded with -100 ``[B, Nt]``.
            input_lengths: Raw waveform sample counts ``[B]``.

        Returns:
            Binary float tensor ``[B, max_Tf]`` where 1.0 indicates a
            mispronounced frame.
        """
        feat_lengths = self.model.wav2vec2._get_feat_extract_output_lengths(
            input_lengths
        )
        B = linguistic.size(0)
        max_T = feat_lengths.max().item()
        targets = torch.zeros(B, max_T, device=linguistic.device)

        for b in range(B):
            T = feat_lengths[b].item()
            if T <= 0:
                continue

            # Extract non-padded phoneme IDs
            c_ids = linguistic[b][linguistic[b] != 0].tolist()       # canonical (pad=0)
            t_ids = transcript[b][transcript[b] != -100].tolist()    # transcript (pad=-100)

            min_len = min(len(c_ids), len(t_ids))
            if min_len == 0:
                continue

            # Phoneme-level mismatch: 1 where student ≠ canonical
            phoneme_labels = [
                1.0 if t_ids[i] != c_ids[i] else 0.0
                for i in range(min_len)
            ]
            n = len(phoneme_labels)

            # Uniform stretching: map each phoneme position to T/n frames
            for i, label in enumerate(phoneme_labels):
                start = int(i * T / n)
                end = int((i + 1) * T / n)
                targets[b, start:end] = label

        return targets

    @staticmethod
    def _build_frame_mask(
        detection: torch.Tensor,
        feat_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Build a binary mask that is 1 for valid frames, 0 for padding.

        Args:
            detection:    Detection scores ``[B, max_T]``.
            feat_lengths: Valid frame counts per sample ``[B]``.

        Returns:
            Float mask of shape ``[B, max_T]``.
        """
        B, max_T = detection.shape
        mask = torch.zeros(B, max_T, device=detection.device)
        for b in range(B):
            mask[b, : feat_lengths[b].item()] = 1.0
        return mask

    # ── Utilities ────────────────────────────────────────────────────────────

    def _batch_to_device(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Move every tensor in a batch tuple to the configured device."""
        return tuple(
            t.to(self.device) if isinstance(t, torch.Tensor) else t
            for t in batch
        )

    def _save_checkpoint(
        self, epoch: int, val_ctc: float, val_det: float
    ) -> None:
        """Persist model weights and metadata to disk."""
        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "val_ctc_loss": val_ctc,
            "val_det_loss": val_det,
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
