from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.utils.evaluate import _align  # Needleman-Wunsch aligner


class MDDTrainer:
    """High-performance trainer for the MDD multimodal model.

    Jointly optimises **CTC loss** (phoneme recognition) and **detection
    loss** (mispronunciation flagging) with:

    * Proper Needleman-Wunsch alignment for frame-level detection labels
      (replaces crude uniform stretching).
    * Gate regularisation — encourages the gated-fusion gate to make
      decisive per-frame choices rather than sitting at 0.5.
    * Detection warmup — freezes the detection head for the first
      ``detection_warmup_epochs`` so phoneme recognition stabilises before
      the detection signal kicks in.
    """

    # ── CTC blank token ID ──────────────────────────────────────────────────
    BLANK_ID: int = 0

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
        # Joint-loss weights
        self.detection_lambda: float = float(
            train_cfg.get("detection_lambda", 1.0)
        )
        self.gate_lambda: float = float(
            train_cfg.get("gate_lambda", 0.1)
        )
        self.detection_warmup_epochs: int = int(
            train_cfg.get("detection_warmup_epochs", 5)
        )
        self.unfreeze_fe_epoch: int = int(
            train_cfg.get("unfreeze_feature_extractor_epoch", 0)
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
        self.ctc_loss_fn = nn.CTCLoss(blank=self.BLANK_ID, zero_infinity=True)
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
        self._current_epoch: int = 0

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
            f"Detection λ: {self.detection_lambda} | "
            f"Gate λ: {self.gate_lambda} | "
            f"Detection warmup: {self.detection_warmup_epochs} epochs"
        )
        self._print_trainable_parameters()

        for epoch in range(1, self.epochs + 1):
            self._current_epoch = epoch
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

    # ── Private: training & evaluation ───────────────────────────────────────

    def _train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """Execute a single training epoch.

        Returns:
            ``(avg_ctc_loss, avg_detection_loss)`` over all batches.
        """
        self.model.train()

        # ── Unfreeze Wav2Vec2 feature extractor at the scheduled epoch ──
        if epoch == self.unfreeze_fe_epoch and self.unfreeze_fe_epoch > 0:
            for param in self.model.wav2vec2.feature_extractor.parameters():
                param.requires_grad = True
            print(
                f"  ↳ Unfroze Wav2Vec2 feature extractor "
                f"({sum(p.numel() for p in self.model.wav2vec2.feature_extractor.parameters()):,} params)"
            )

        running_ctc: float = 0.0
        running_det: float = 0.0

        # Detection warmup: freeze detection head for early epochs
        det_enabled = epoch > self.detection_warmup_epochs
        if self.detection_warmup_epochs > 0:
            self._set_detection_grad(det_enabled)

        for step, batch in enumerate(self.train_loader):
            (
                input_values, linguistic, transcript,
                target_lengths, input_lengths, attention_mask,
            ) = self._batch_to_device(batch)

            with torch.amp.autocast(device_type="cuda", enabled=self.fp16):
                # Forward pass — returns phoneme logits, detection scores, gate
                logits, detection, gate = self.model(
                    input_values, linguistic,
                    attention_mask=attention_mask,
                    return_detection=True,
                )  # logits: [B,Tf,V]  detection: [B,Tf]  gate: [B,Tf]

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
                # Build clean targets via CTC segmentation + NW alignment
                det_targets = self._build_detection_targets(
                    logits, linguistic, ctc_input_lengths
                )
                det_mask = self._build_frame_mask(detection, ctc_input_lengths)

                if det_enabled:
                    det_loss = (
                        self.det_loss_fn(detection, det_targets) * det_mask
                    ).sum() / det_mask.sum().clamp(min=1)
                else:
                    det_loss = torch.tensor(0.0, device=self.device)

                # ── Gate regularisation ─────────────────────────────────
                gate_reg = self._compute_gate_regularization(gate, det_mask)

                # ── Combined loss ───────────────────────────────────────
                loss = (
                    ctc_loss
                    + self.detection_lambda * det_loss
                    + self.gate_lambda * gate_reg
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
        """Evaluate the model on the validation set."""
        self.model.eval()
        total_ctc: float = 0.0
        total_det: float = 0.0

        for batch in self.dev_loader:
            (
                input_values, linguistic, transcript,
                target_lengths, input_lengths, attention_mask,
            ) = self._batch_to_device(batch)

            logits, detection, _gate = self.model(
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
                logits, linguistic, ctc_input_lengths
            )
            det_mask = self._build_frame_mask(detection, ctc_input_lengths)
            det_loss = (
                self.det_loss_fn(detection, det_targets) * det_mask
            ).sum() / det_mask.sum().clamp(min=1)

            total_ctc += ctc_loss.item()
            total_det += det_loss.item()

        n = len(self.dev_loader)
        return total_ctc / n, total_det / n

    # ── Detection-target builder (NW-alignment based) ────────────────────────

    def _build_detection_targets(
        self,
        logits: torch.Tensor,
        linguistic: torch.Tensor,
        feat_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Build frame-level binary mismatch labels using CTC segmentation
        followed by Needleman-Wunsch alignment against the canonical sequence.

        This replaces the crude uniform-stretching approach with proper
        sequence alignment, giving much cleaner supervision to the detection
        head.

        Args:
            logits:        Phoneme logits ``[B, Tf, V]`` (pre-softmax).
            linguistic:    Canonical phoneme IDs, padded with 0  ``[B, Nc]``.
            feat_lengths:  Valid frame counts per sample ``[B]``.

        Returns:
            Binary float tensor ``[B, max_Tf]``.  1.0 = mispronounced frame.
        """
        B, max_T, _ = logits.shape
        targets = torch.zeros(B, max_T, device=logits.device)

        for b in range(B):
            T = feat_lengths[b].item()
            if T <= 0:
                continue

            # ── 1. CTC segmentation: collapse argmax into (phoneme, boundaries) ──
            pred_ids = logits[b, :T].argmax(dim=-1).tolist()  # [T]
            phonemes, boundaries = self._ctc_collapse_with_boundaries(pred_ids)
            # phonemes:   list of token IDs
            # boundaries: list of (start_frame, end_frame) pairs

            # ── 2. Get canonical phoneme IDs (strip PAD = 0) ─────────────────
            canon_ids = linguistic[b][linguistic[b] != 0].tolist()

            if not canon_ids or not phonemes:
                continue

            # ── 3. Needleman-Wunsch alignment ────────────────────────────────
            aligned_pred, aligned_canon = _align(phonemes, canon_ids)
            # aligned_pred:   list of token IDs (with <eps> for gaps)
            # aligned_canon:  list of token IDs (with <eps> for gaps)

            # ── 4. Walk the aligned sequences and mark mismatch frames ───────
            pred_idx = 0  # index into the non-gap predicted phonemes
            for p_tok, c_tok in zip(aligned_pred, aligned_canon):
                if p_tok == "<eps>" or c_tok == "<eps>":
                    # Insertion or deletion — flag as potential mismatch
                    if pred_idx < len(boundaries):
                        s, e = boundaries[pred_idx]
                        targets[b, s:e] = 1.0
                    if p_tok != "<eps>":
                        pred_idx += 1
                    continue

                if pred_idx < len(boundaries):
                    if p_tok != c_tok:
                        # Mismatch — mark these frames as mispronounced
                        s, e = boundaries[pred_idx]
                        targets[b, s:e] = 1.0
                    # else: match — leave as 0.0 (correct)
                pred_idx += 1

        return targets

    # ── Gate regularisation ──────────────────────────────────────────────────

    @staticmethod
    def _compute_gate_regularization(
        gate: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Penalise gate values that hover near 0.5 (indecision).

        The gate should be *bimodal*: close to 0 (trust audio) or close to 1
        (trust canonical).  Values in the flat region [0.2, 0.8] incur a
        linear penalty.
        """
        # gate is [B, T] — mean gate value over the hidden dimension
        deviation = (gate - 0.5).abs()           # distance from 0.5
        penalty = (0.3 - deviation).clamp(min=0)  # penalise when within 0.3 of 0.5
        return (penalty * mask).sum() / mask.sum().clamp(min=1)

    # ── CTC segmentation helper ──────────────────────────────────────────────

    @staticmethod
    def _ctc_collapse_with_boundaries(
        pred_ids: List[int],
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Collapse a CTC argmax sequence into phonemes with frame boundaries.

        Consecutive identical token IDs are merged; *blank* tokens are
        removed.  Each surviving phoneme is paired with its ``(start, end)``
        frame range.

        Args:
            pred_ids:  List of token IDs over T frames (e.g. from ``argmax``).

        Returns:
            ``(phonemes, boundaries)`` where *phonemes* is a list of token
            IDs and *boundaries* is a list of ``(start, end)`` frame indices
            (end is exclusive).
        """
        phonemes: List[int] = []
        boundaries: List[Tuple[int, int]] = []

        prev = MDDTrainer.BLANK_ID
        start = 0

        for t, pid in enumerate(pred_ids):
            if pid != prev:
                if prev != MDDTrainer.BLANK_ID:
                    phonemes.append(prev)
                    boundaries.append((start, t))
                start = t
            prev = pid

        # Don't forget the last segment
        if prev != MDDTrainer.BLANK_ID:
            phonemes.append(prev)
            boundaries.append((start, len(pred_ids)))

        return phonemes, boundaries

    # ── Frame mask builder ───────────────────────────────────────────────────

    @staticmethod
    def _build_frame_mask(
        detection: torch.Tensor,
        feat_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Build a binary mask that is 1 for valid frames, 0 for padding."""
        B, max_T = detection.shape
        mask = torch.zeros(B, max_T, device=detection.device)
        for b in range(B):
            mask[b, : feat_lengths[b].item()] = 1.0
        return mask

    # ── Detection-head grad control ──────────────────────────────────────────

    def _set_detection_grad(self, enabled: bool) -> None:
        """Enable or disable gradients for the detection head and gate."""
        for name, param in self.model.named_parameters():
            if "detection_head" in name or "gate_proj" in name:
                param.requires_grad = enabled

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
