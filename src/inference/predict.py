"""
MDD Inference Pipeline — Generates the ``predictions.csv`` submission file.

Loads a trained :class:`MDDModelBuilder` checkpoint, ingests raw ``.wav``
files via :func:`~src.utils.audio_utils.load_and_resample`, runs greedy
CTC decoding, and writes the phoneme hypotheses to disk.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch import nn
from transformers import Wav2Vec2FeatureExtractor

from src.models.builder import MDDModelBuilder
from src.utils.audio_utils import load_and_resample, to_mono
from src.utils.file_utils import load_config, load_json


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE: int = 16000
BLANK_TOKEN_ID: int = 0

# Default vocabulary path (relative to project root)
DEFAULT_VOCAB_PATH: str = "data/processed/vocab.json"


class MDDPredictor:
    """Inference orchestrator for the MDD multimodal model.

    Loads a trained checkpoint, processes raw audio into ``input_values``,
    encodes canonical phoneme sequences, runs a forward pass, and decodes
    the output logits with greedy CTC decoding.

    Args:
        config_path:
            Path to ``config.yaml`` (default: ``"config.yaml"``).
        checkpoint_path:
            Path to the ``.pth`` checkpoint file (model weights only or
            full training state dict — both are handled).
        vocab_path:
            Path to the ``vocab.json`` file mapping phoneme tokens → IDs.
            Defaults to ``data/processed/vocab.json``.
        device:
            PyTorch device string (``"cuda"``, ``"cpu"``, or ``None`` for
            auto-detection).
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        checkpoint_path: str = "experiments/checkpoints/best_model.pt",
        vocab_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        # ── Device ───────────────────────────────────────────────────────
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # ── Configuration & vocabulary ───────────────────────────────────
        self.config = load_config(config_path)
        self.vocab_path = vocab_path or self.config.get(
            "vocab_path", DEFAULT_VOCAB_PATH
        )
        self.vocab: Dict[str, int] = load_json(self.vocab_path)

        # Reverse mapping: int ID → phoneme token string
        self.id_to_token: Dict[int, str] = {
            idx: token for token, idx in self.vocab.items()
        }

        self.vocab_size: int = len(self.vocab)

        # ── Model ────────────────────────────────────────────────────────
        self.model = MDDModelBuilder(
            config=self.config, vocab_size=self.vocab_size
        )

        # Load checkpoint (handles both raw state_dict and full checkpoint dict)
        checkpoint: Dict[str, Any] = torch.load(
            checkpoint_path, map_location=self.device
        )
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        # ── Feature extractor ────────────────────────────────────────────
        pretrained_name: str = self.config["model"]["pretrained_name"]
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            pretrained_name,
            return_attention_mask=False,
        )

        print(f"MDDPredictor initialised on {self.device}")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Checkpoint: {checkpoint_path}")

    # ── Public API ───────────────────────────────────────────────────────────

    def predict_dataset(
        self,
        csv_path: str,
        wav_dir: Optional[str] = None,
        batch_size: int = 1,
    ) -> pd.DataFrame:
        """Run inference over every row in a CSV and return a predictions
        DataFrame.

        The CSV must contain at least the following columns:
          - ``path`` (or ``Path``) — relative or absolute ``.wav`` path.
          - ``canonical`` (or ``Canonical``) — space-separated phoneme
            sequence used as the linguistic input.

        Args:
            csv_path:
                Path to the CSV file listing test samples.
            wav_dir:
                Directory prepended to relative audio paths.  If ``None``
                the raw ``path`` column is used as-is.
            batch_size:
                Number of samples to process at once (default: 1).

        Returns:
            A ``DataFrame`` with columns ``[id, predict]`` where each row
            holds the decoded phoneme hypothesis.
        """
        df = pd.read_csv(csv_path)

        # ── Resolve column names ─────────────────────────────────────────
        path_col = "path" if "path" in df.columns else "Path"
        canonical_col = (
            "canonical" if "canonical" in df.columns else "Canonical"
        )

        paths: List[str] = df[path_col].astype(str).tolist()
        canonicals: List[str] = df[canonical_col].astype(str).tolist()
        predictions: List[str] = []

        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i : i + batch_size]
                batch_canonicals = canonicals[i : i + batch_size]

                # (a) Load and process audio
                input_values, _ = self._prepare_audio_batch(
                    batch_paths, wav_dir
                )

                # (b) Encode canonical sequences
                linguistic = self._prepare_linguistic_batch(batch_canonicals)

                # (c) Forward pass
                logits: torch.Tensor = self.model(input_values, linguistic)
                # [B, T, V]

                # (d) Decode each sample in the batch
                for b in range(logits.shape[0]):
                    hypothesis = self._greedy_decode_tokens(logits[b])
                    predictions.append(hypothesis)

        # ── Build submission DataFrame ───────────────────────────────────
        result_df = pd.DataFrame(
            {"id": range(len(predictions)), "predict": predictions}
        )
        return result_df

    def predict_and_save(
        self,
        csv_path: str,
        output_path: str = "predictions.csv",
        wav_dir: Optional[str] = None,
        batch_size: int = 1,
    ) -> str:
        """Run inference and write results directly to a CSV file.

        Args:
            csv_path: Path to the input CSV.
            output_path: Destination path for the predictions CSV.
            wav_dir: Optional base directory for audio files.
            batch_size: Batch size for inference.

        Returns:
            The ``output_path`` (for convenience).
        """
        result_df = self.predict_dataset(
            csv_path=csv_path,
            wav_dir=wav_dir,
            batch_size=batch_size,
        )
        result_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}  ({len(result_df)} rows)")
        return output_path

    # ── Decoding ─────────────────────────────────────────────────────────────

    def _greedy_decode_tokens(self, frame_logits: torch.Tensor) -> str:
        """Greedy CTC decode a single sample's frame-level logits.

        Steps:
          1. ``argmax(dim=-1)`` → most-likely token per frame.
          2. Collapse consecutive identical token IDs.
          3. Remove ``BLANK_TOKEN_ID`` (0).
          4. Map remaining IDs to phoneme strings; join with spaces.

        Args:
            frame_logits: Logits tensor of shape ``[T, V]`` (time × vocab).

        Returns:
            A space-separated string of predicted phoneme tokens.
        """
        pred_ids: List[int] = torch.argmax(frame_logits, dim=-1).tolist()

        # Collapse consecutive duplicates (CTC "merge" rule)
        collapsed: List[int] = []
        prev: Optional[int] = None
        for token_id in pred_ids:
            if token_id != prev:
                collapsed.append(token_id)
            prev = token_id

        # Remove blank tokens and map IDs → strings
        tokens: List[str] = []
        for token_id in collapsed:
            if token_id == BLANK_TOKEN_ID:
                continue
            token: str = self.id_to_token.get(token_id, "")
            if token and token != "<eps>":
                tokens.append(token)

        return " ".join(tokens)

    # ── Batch preparation helpers ────────────────────────────────────────────

    def _prepare_audio_batch(
        self,
        paths: List[str],
        wav_dir: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load, resample, and feature-extract a batch of audio files.

        Args:
            paths: List of ``.wav`` file paths (absolute or relative).
            wav_dir: Optional base directory prepended to relative paths.

        Returns:
            ``(input_values, input_lengths)`` tuple:
              - **input_values** — ``[B, T_audio]`` padded tensor.
              - **input_lengths** — ``[B]`` raw sample counts.
        """
        waveforms: List[torch.Tensor] = []
        input_lengths: List[int] = []

        for p in paths:
            wav_path = self._resolve_wav_path(p, wav_dir)
            waveform, sr = load_and_resample(wav_path, target_sr=SAMPLE_RATE)
            waveform = to_mono(waveform)                         # [1, num_samples]
            waveform = waveform.squeeze(0)                       # [num_samples]
            waveforms.append(waveform.numpy())
            input_lengths.append(waveform.shape[0])

        # Feature-extract: normalise & pad to batch-max length
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_tensors="pt",
        )
        input_values: torch.Tensor = inputs.input_values.to(self.device)
        input_lengths_t = torch.tensor(
            input_lengths, dtype=torch.long, device=self.device
        )

        return input_values, input_lengths_t

    def _prepare_linguistic_batch(
        self,
        canonicals: List[str],
    ) -> torch.Tensor:
        """Tokenise canonical phoneme sequences and pad to batch max length.

        Args:
            canonicals: List of space-separated phoneme strings.

        Returns:
            Long tensor of shape ``[B, N_max]`` (padded).
        """
        token_lists: List[List[int]] = [
            self._text_to_ids(seq) for seq in canonicals
        ]

        max_len: int = max((len(t) for t in token_lists), default=1)

        # Use 0 as pad ID (consistent with CTC blank, but distinct in practice
        # since linguistic sequences never contain blank tokens)
        linguistic = torch.zeros(
            (len(token_lists), max_len), dtype=torch.long, device=self.device
        )
        for j, ids in enumerate(token_lists):
            if ids:
                linguistic[j, : len(ids)] = torch.tensor(
                    ids, dtype=torch.long, device=self.device
                )

        return linguistic

    # ── Utility methods ──────────────────────────────────────────────────────

    def _text_to_ids(self, canonical: str) -> List[int]:
        """Convert a space-separated canonical string to a list of token IDs.

        Unknown tokens are silently skipped.

        Args:
            canonical: Space-separated phoneme sequence (e.g. ``"aː-0 m aː-4"``).

        Returns:
            List of integer token IDs.
        """
        ids: List[int] = []
        for token in canonical.split(" "):
            token = token.strip()
            if token and token in self.vocab:
                ids.append(self.vocab[token])
        return ids

    @staticmethod
    def _resolve_wav_path(path: str, wav_dir: Optional[str] = None) -> str:
        """Resolve a ``.wav`` path, optionally prefixing a base directory.

        Args:
            path: Raw path string from the CSV.
            wav_dir: Base directory for relative paths.

        Returns:
            Absolute or absolute-relative path to the audio file.
        """
        raw = str(path).strip()
        if wav_dir and not raw.startswith("/") and not raw.startswith("\\"):
            import os as _os

            if not _os.path.isabs(raw):
                raw = _os.path.join(wav_dir, raw)
        if not raw.lower().endswith(".wav"):
            raw = f"{raw}.wav"
        return raw