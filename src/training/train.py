from __future__ import annotations

import os
import csv
import tempfile
from typing import Any, Dict, Tuple, List

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.utils.file_utils import load_json
from src.utils.evaluate import compute_f1
from src.inference.beam_search import BeamSearchDecoder, build_vocab_counts


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
        decay_params_backbone: list[nn.Parameter] = []
        no_decay_params_backbone: list[nn.Parameter] = []
        decay_params_head: list[nn.Parameter] = []
        no_decay_params_head: list[nn.Parameter] = []

        backbone_lr = float(train_cfg.get("backbone_lr", self.learning_rate))

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = "wav2vec2" in name
            if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
                if is_backbone:
                    no_decay_params_backbone.append(param)
                else:
                    no_decay_params_head.append(param)
            else:
                if is_backbone:
                    decay_params_backbone.append(param)
                else:
                    decay_params_head.append(param)

        param_groups = []
        if decay_params_backbone or no_decay_params_backbone:
            param_groups.append({
                "params": decay_params_backbone + no_decay_params_backbone,
                "weight_decay": self.weight_decay,
                "lr": backbone_lr,
            })
        if decay_params_head or no_decay_params_head:
            param_groups.append({
                "params": decay_params_head + no_decay_params_head,
                "weight_decay": self.weight_decay,
                "lr": self.learning_rate,
            })

        # If no param groups were created, fall back to all model parameters
        if param_groups:
            self.optimizer = AdamW(param_groups)
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

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

        # Detection loss settings (pos_weight approx from main.py)
        self.pos_weight: float = float(train_cfg.get("detection_pos_weight", 1.0))
        self.use_focal: bool = bool(train_cfg.get("use_focal_loss", False))
        self.focal_gamma: float = float(train_cfg.get("focal_gamma", 2.0))
        self.focal_alpha: float = float(train_cfg.get("focal_alpha", 0.75))
        self.label_smoothing_frames: int = int(train_cfg.get("detection_label_smoothing_frames", 0))
        self.ohem_topk: float = float(train_cfg.get("detection_ohem_topk", 1.0))

        if not self.use_focal:
            # BCEWithLogitsLoss supports pos_weight to address class imbalance
            try:
                pw = torch.tensor(self.pos_weight, device=self.device)
                self.det_loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
            except Exception:
                self.det_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        else:
            # We'll compute focal loss manually in the training loop
            self.det_loss_fn = None

        # Gradient clipping
        self.max_grad_norm: float = float(train_cfg.get("max_grad_norm", 1.0))

        # ── Checkpointing ──────────────────────────────────────────────────
        self.checkpoint_dir: str = config.get(
            "checkpoint_dir", "experiments/checkpoints"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path: str = os.path.join(
            self.checkpoint_dir, "best_model.pt"
        )
        # Track best metric (loss by default — lower is better). If using
        # an evaluation metric like F1, we invert the comparison accordingly.
        if self.metric_for_best_model.lower() == "f1":
            self.best_metric: float = -float("inf")
            self.metric_mode: str = "max"
        else:
            self.best_metric = float("inf")
            self.metric_mode = "min"

        # Best validation loss is tracked separately for checkpoint metadata.
        self.best_val_loss: float = float("inf")

        # Move model to device
        self.model = self.model.to(self.device)

        # ── Vocabulary (for greedy decoding during validation) ───────────
        # Attempt to load vocab path from config; fall back to standard location.
        vocab_path = config.get("vocab_path", "data/processed/vocab.json")
        try:
            self.vocab = load_json(vocab_path)
            self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        except Exception:
            self.vocab = None
            self.id_to_token = None

        # ── Beam search decoder (optional) ─────────────────────────────────
        # Used for validation F1 computation if enabled
        use_beam_search = config.get("use_beam_search", False)
        beam_width = config.get("beam_width", 3)
        lm_weight = config.get("lm_weight", 0.0)
        self.beam_decoder = None

        if use_beam_search and self.id_to_token is not None:
            # Build vocab counts from training dataset for LM prior
            try:
                vocab_counts = build_vocab_counts(train_loader.dataset)
                self.beam_decoder = BeamSearchDecoder(
                    vocab_size=len(self.vocab),
                    beam_width=beam_width,
                    lm_weight=lm_weight,
                    vocab_counts=vocab_counts,
                )
                self.beam_decoder.set_id_to_token(self.id_to_token)
                print(f"Beam search decoder enabled: width={beam_width}, lm_weight={lm_weight}")
            except Exception as e:
                print(f"Warning: failed to initialize beam decoder: {e}")
                self.beam_decoder = None

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
            f"Max grad norm: {self.max_grad_norm}"
        )
        self._print_trainable_parameters()

        # Track metric history for early stopping insight
        f1_history = []
        patience_counter = 0
        patience_limit = 5

        for epoch in range(1, self.epochs + 1):
            train_ctc, train_det = self._train_one_epoch(epoch)
            val_ctc, val_det = self._evaluate()
            val_combined = val_ctc + self.detection_lambda * val_det

            # Compute validation F1 (if possible) by greedy-decoding the
            # dev_loader logits and running the existing scorer.
            val_f1 = None
            if self.id_to_token is not None:
                try:
                    val_f1 = self._compute_validation_f1()
                    f1_history.append(val_f1)
                except Exception as e:
                    print(f"  Warning: F1 computation failed: {e}")
                    val_f1 = None

            if val_f1 is None:
                print(
                    f"Epoch {epoch:3d}/{self.epochs} — "
                    f"Train CTC: {train_ctc:.4f}  Det: {train_det:.4f} | "
                    f"Val CTC: {val_ctc:.4f}  Det: {val_det:.4f}  "
                    f"Combined: {val_combined:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch:3d}/{self.epochs} — "
                    f"Train CTC: {train_ctc:.4f}  Det: {train_det:.4f} | "
                    f"Val CTC: {val_ctc:.4f}  Det: {val_det:.4f}  "
                    f"Combined: {val_combined:.4f}  Val F1: {val_f1:.4f}"
                )

            # ── Checkpoint based on chosen metric ───────────────────────
            if self.metric_for_best_model.lower() == "f1" and val_f1 is not None:
                cur_metric = val_f1
            else:
                cur_metric = val_combined

            is_better = (
                (self.metric_mode == "max" and cur_metric > self.best_metric)
                or (self.metric_mode == "min" and cur_metric < self.best_metric)
            )

            if is_better:
                self.best_metric = cur_metric
                self._save_checkpoint(epoch, val_ctc, val_det)
                print(f"  ↳ New best model saved ({self.metric_for_best_model}={cur_metric:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit and epoch > self.epochs // 2:
                    print(f"  ↳ No improvement for {patience_counter} epochs. Stopping early.")
                    break

        # ── Reload best checkpoint ─────────────────────────────────────
        if self.load_best_model_at_end and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Loaded best checkpoint with "
                f"best_{self.metric_for_best_model}={self.best_metric:.4f}"
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
                det_loss = self._detection_loss(detection, det_targets, det_mask)

                # ── Combined loss ───────────────────────────────────────
                loss = (
                    ctc_loss + self.detection_lambda * det_loss
                ) / self.gradient_accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Step optimiser & scheduler after accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient unscale + clipping when using AMP
                try:
                    self.scaler.unscale_(self.optimizer)
                except Exception:
                    pass
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            running_ctc += ctc_loss.item()
            running_det += det_loss.item()

        # Handle any remaining gradients at end of epoch
        if (step + 1) % self.gradient_accumulation_steps != 0:
            try:
                self.scaler.unscale_(self.optimizer)
            except Exception:
                pass
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

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
            det_loss = self._detection_loss(detection, det_targets, det_mask)

            total_ctc += ctc_loss.item()
            total_det += det_loss.item()

        n = len(self.dev_loader)
        return total_ctc / n, total_det / n

    def _greedy_decode_tokens(self, frame_logits: torch.Tensor) -> str:
        """Decode logits using beam search (if enabled) or greedy CTC."""
        if self.beam_decoder is not None:
            return self.beam_decoder.decode(frame_logits)
        else:
            return self._greedy_ctc_decode(frame_logits)

    def _greedy_ctc_decode(self, frame_logits: torch.Tensor) -> str:
        """Greedy CTC decode (same logic as the inference pipeline)."""
        pred_ids: List[int] = torch.argmax(frame_logits, dim=-1).tolist()

        collapsed: List[int] = []
        prev = None
        for token_id in pred_ids:
            if token_id != prev:
                collapsed.append(token_id)
            prev = token_id

        tokens: List[str] = []
        for token_id in collapsed:
            if token_id == 0:
                continue
            token = self.id_to_token.get(token_id, "")
            if token and token != "<eps>":
                tokens.append(token)

        return " ".join(tokens)

    def _compute_validation_f1(self) -> float:
        """Decode the whole dev set greedily and compute F1 using scorer.

        Writes two temporary CSVs (ground truth and predictions) and uses
        the existing `compute_f1` utility to return the scalar F1.
        """
        # Collect predictions and ground-truth rows in dataset order
        preds: List[str] = []
        gt_rows: List[Dict[str, str]] = []

        # If the dev_loader uses a collate_fn that pads, iteration order
        # with shuffle=False preserves dataset order, so we can iterate
        # and map predictions back to dataset.records.
        with torch.no_grad():
            for batch in self.dev_loader:
                (
                    input_values, linguistic, transcript,
                    target_lengths, input_lengths, attention_mask,
                ) = self._batch_to_device(batch)

                logits = self.model(input_values, linguistic)
                for b in range(logits.shape[0]):
                    hypothesis = self._greedy_decode_tokens(logits[b])
                    preds.append(hypothesis)

        # Ground truth — read canonical & transcript from underlying dataset
        try:
            dataset = self.dev_loader.dataset
            for rec in dataset.records:
                gt_rows.append({
                    "canonical": rec.get("canonical", ""),
                    "transcript": rec.get("transcript", ""),
                })
        except Exception:
            raise RuntimeError("Unable to extract ground-truth rows from dev dataset")

        if len(preds) != len(gt_rows):
            raise RuntimeError("Prediction / ground-truth length mismatch during val F1 computation")

        # Write temporary CSVs and call existing compute_f1
        with tempfile.TemporaryDirectory() as td:
            gt_path = os.path.join(td, "ground_truth.csv")
            res_path = os.path.join(td, "results.csv")

            with open(gt_path, "w", encoding="utf-8", newline="") as gf:
                writer = csv.DictWriter(gf, fieldnames=["canonical", "transcript"])
                writer.writeheader()
                for row in gt_rows:
                    writer.writerow(row)

            with open(res_path, "w", encoding="utf-8", newline="") as rf:
                writer = csv.DictWriter(rf, fieldnames=["predict"])
                writer.writeheader()
                for p in preds:
                    writer.writerow({"predict": p})

            f1 = compute_f1(gt_path, res_path)
        return f1

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

        # Optional label smoothing across frames: apply a simple box filter
        # (moving average) of radius r on the target grid to soften boundaries.
        r = int(getattr(self, "label_smoothing_frames", 0))
        if r > 0:
            kernel = torch.ones((2 * r + 1,), device=targets.device, dtype=targets.dtype)
            kernel = kernel / kernel.sum()
            # Convolve per-batch row with 1D conv using F.pad for simplicity
            padded = torch.nn.functional.pad(targets.unsqueeze(1), (r, r))  # [B, 1, T+2r]
            smoothed = torch.nn.functional.conv1d(padded, kernel.view(1, 1, -1))
            targets = smoothed.squeeze(1)

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

    def _detection_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute detection loss supporting focal or weighted BCE.

        Args:
            logits: [B, T]
            targets: [B, T]
            mask: [B, T]
        """
        if self.use_focal:
            probs = torch.sigmoid(logits)
            # p_t: probability of the true class
            p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
            # BCE per-element
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            modulating = (1.0 - p_t) ** self.focal_gamma
            alpha_factor = targets * self.focal_alpha + (1.0 - targets) * (1.0 - self.focal_alpha)
            per_elem = alpha_factor * modulating * bce
        else:
            per_elem = self.det_loss_fn(logits, targets)

        # Apply mask
        per_elem = per_elem * mask

        # Optional OHEM: keep top fraction of hardest frames
        topk = float(getattr(self, "ohem_topk", 1.0))
        if topk < 1.0:
            flat = per_elem.view(-1)
            k = max(1, int(flat.numel() * topk))
            top_vals, _ = torch.topk(flat, k)
            loss = top_vals.mean()
        else:
            loss = per_elem.sum() / mask.sum().clamp(min=1)
        return loss

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
        self.best_val_loss = val_ctc + self.detection_lambda * val_det
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
