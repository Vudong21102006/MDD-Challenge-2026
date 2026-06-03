from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model


# ──────────────────────────────────────────────────────────────────────────────
# Phonetic Encoder Sub-Modules
# ──────────────────────────────────────────────────────────────────────────────

class PhoneCNNStack(nn.Module):
    """Convolutional stack for local phonetic feature refinement.

    Applies a 2D convolution (with a singleton channel dimension added
    dynamically), followed by BatchNorm1d, ReLU activation, and Dropout.

    Args:
        hidden_dim: Number of feature channels (must equal the input's
                    last dimension, typically 768 for Wav2Vec2-base).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine phonetic features with Conv2d → BatchNorm → ReLU → Dropout.

        Args:
            x: Input tensor of shape ``[B, T, hidden_dim]`` or
               ``[T, hidden_dim]``.

        Returns:
            Tensor of the same shape as ``x``.
        """
        ndim = x.dim()
        if ndim == 3:
            x = x.unsqueeze(1)                    # [B, 1, T, hidden_dim]
        elif ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)       # [1, 1, T, hidden_dim]

        x = self.conv2d(x)                        # [*, 1, T, hidden_dim]
        x = x.squeeze(1)                          # [*, T, hidden_dim]  (≥3D)

        # BatchNorm1d expects [N, C, L] — already 3D at this point
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)
        x = self.dropout(x)

        if ndim == 2:
            x = x.squeeze(0)                      # [T, hidden_dim]

        return x


class PhoneRNNStack(nn.Module):
    """Bidirectional LSTM stack for temporal modelling of phonetic features.

    Halves the hidden size internally (since bidirectional doubles it back),
    then applies BatchNorm1d and Dropout.

    Args:
        hidden_dim: Input and output feature dimension.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply BiLSTM → BatchNorm → Dropout.

        Args:
            x: Tensor of shape ``[B, T, hidden_dim]``.

        Returns:
            Tensor of shape ``[B, T, hidden_dim]``.
        """
        x, _ = self.bilstm(x)
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        return x


class PhoneticEncoder(nn.Module):
    """Phonetic encoder chaining ``PhoneCNNStack → PhoneRNNStack``.

    Args:
        hidden_dim: Feature dimension (768 for Wav2Vec2-base).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.cnn = PhoneCNNStack(hidden_dim=hidden_dim)
        self.rnn = PhoneRNNStack(hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine Wav2Vec2 hidden states through CNN and RNN stacks.

        Args:
            x: Raw Wav2Vec2 last hidden state, ``[B, T, hidden_dim]``.

        Returns:
            Refined tensor of shape ``[B, T, hidden_dim]``.
        """
        x = self.cnn(x)
        x = self.rnn(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Linguistic Encoder
# ──────────────────────────────────────────────────────────────────────────────

class LinguisticEncoder(nn.Module):
    """Linguistic encoder mapping canonical phoneme indices to key/value pairs.

    Architecture::

        Embedding(vocab=256, dim=64)
            → BiLSTM(64 → 64, bidirectional)
            → key_proj(128 → 2304)    # h_k
            → value_proj(128 → 2304)  # h_v

    The output dimension 2304 equals 3 × 768 (Wav2Vec2-base hidden dim),
    giving each of the 16 attention heads a 48-dim slice from the query
    against a 144-dim slice from the key/value space.
    """

    _OUTPUT_DIM: int = 2304

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        lstm_hidden: int = embed_dim               # 64
        lstm_output: int = lstm_hidden * 2          # 128 (bidirectional)

        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True,
        )

        self.key_proj = nn.Linear(lstm_output, self._OUTPUT_DIM)    # h_k
        self.value_proj = nn.Linear(lstm_output, self._OUTPUT_DIM)  # h_v

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode canonical phoneme tokens into key and value representations.

        Args:
            x: Integer token indices of shape ``[B, N]``, or ``[N]`` for a
               single sample.

        Returns:
            A ``(h_k, h_v)`` tuple:
                - **h_k** — Key projection,   ``[B, N, 2304]``.
                - **h_v** — Value projection, ``[B, N, 2304]``.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.long()
        embedded: torch.Tensor = self.embedding(x)      # [B, N, 64]
        out, _ = self.bilstm(embedded)                   # [B, N, 128]

        h_k: torch.Tensor = self.key_proj(out)           # [B, N, 2304]
        h_v: torch.Tensor = self.value_proj(out)         # [B, N, 2304]

        return h_k, h_v


# ──────────────────────────────────────────────────────────────────────────────
# Main Multimodal Model Builder
# ──────────────────────────────────────────────────────────────────────────────

class MDDModelBuilder(nn.Module):
    """Multimodal MDD model with cross-attention between audio and text.

    **Pipeline**::

        1. Wav2Vec2 backbone  → raw phonetic hidden states  [B, T, 768]
        2. PhoneticEncoder    → CNN + BiLSTM refinement     [B, T, 768]
        3. LinguisticEncoder  → key & value projections     [B, N, 2304]
        4. MultiheadAttention → phonetic (Q) attends to linguistic (K, V)
        5. Concat(attn_out, phonetic) → Linear → logits    [B, T, vocab_size]

    Args:
        config:
            Full configuration dictionary (as parsed from ``config.yaml``).
            The ``config['model']`` sub-dictionary is expected to contain:
            - ``pretrained_name`` (str)
            - ``attention_dropout`` (float)
            - ``hidden_dropout`` (float)
            - ``mask_time_prob`` (float)
            - ``mask_time_length`` (int)
            - ``freeze_feature_extractor`` (bool)
        vocab_size:
            Number of output phoneme tokens (CTC vocabulary size).
    """

    # ── Architectural constants ──────────────────────────────────────────────
    HIDDEN_DIM: int = 768          # Wav2Vec2-base hidden size
    LINGUISTIC_PROJ_DIM: int = 2304  # 3 × HIDDEN_DIM for key / value
    NUM_HEADS: int = 16            # MultiheadAttention heads

    def __init__(self, config: Dict[str, Any], vocab_size: int) -> None:
        super().__init__()

        model_cfg: Dict[str, Any] = config["model"]

        # ── 1. Wav2Vec2 backbone ─────────────────────────────────────────
        pretrained_name: str = model_cfg["pretrained_name"]

        # Load the pretrained config and override dropout / mask parameters
        wav2vec2_config = Wav2Vec2Config.from_pretrained(pretrained_name)
        wav2vec2_config.attention_dropout = model_cfg.get(
            "attention_dropout", 0.1
        )
        wav2vec2_config.hidden_dropout = model_cfg.get(
            "hidden_dropout", 0.1
        )
        wav2vec2_config.mask_time_prob = model_cfg.get(
            "mask_time_prob", 0.05
        )
        wav2vec2_config.mask_time_length = model_cfg.get(
            "mask_time_length", 10
        )

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            pretrained_name, config=wav2vec2_config
        )

        # Optionally freeze the convolutional feature extractor
        if model_cfg.get("freeze_feature_extractor", False):
            self.wav2vec2.feature_extractor._freeze_parameters()

        # ── 2. Domain-specific encoders ───────────────────────────────────
        self.phonetic_encoder = PhoneticEncoder(hidden_dim=self.HIDDEN_DIM)
        self.linguistic_encoder = LinguisticEncoder()

        # ── 3. Cross-modal attention ─────────────────────────────────────
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.HIDDEN_DIM,
            num_heads=self.NUM_HEADS,
            kdim=self.LINGUISTIC_PROJ_DIM,
            vdim=self.LINGUISTIC_PROJ_DIM,
            batch_first=True,
        )

        # ── 4. Gated fusion ──────────────────────────────────────────────
        # Learned gate decides per-frame how much to trust the canonical
        # (attn_output) vs. the raw audio (phonetic).  Replaces naive concat.
        self.gate_proj = nn.Linear(self.HIDDEN_DIM * 2, self.HIDDEN_DIM)
        self.output_projection = nn.Linear(self.HIDDEN_DIM, vocab_size)

        # ── 5. Detection head (MLP) ──────────────────────────────────────
        # Small MLP learns non-linear mismatch patterns.
        # 768 → 256 → 64 → 1  with ReLU + Dropout.
        self.detection_head = nn.Sequential(
            nn.Linear(self.HIDDEN_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def forward(
        self,
        input_values: torch.Tensor,
        linguistic: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_detection: bool = False,
    ):
        """Forward pass through the full multimodal pipeline.

        Args:
            input_values:
                Raw audio waveforms.  Shape ``[B, T_audio]``.
            linguistic:
                Canonical phoneme token indices (padded).  Shape ``[B, N]``.
            attention_mask:
                Optional binary mask for waveform padding.  ``[B, T_audio]``.
            return_detection:
                When ``True``, also returns frame-level mispronunciation
                logits (used during training).

        Returns:
            - **logits** — ``[B, Tf, vocab_size]`` phoneme logits (always).
            - **detection** — ``[B, Tf]`` binary mispronunciation logits
              (only when ``return_detection=True``).
        """
        # (a) Wav2Vec2 backbone — raw phonetic hidden states
        phonetic: torch.Tensor = self.wav2vec2(
            input_values, attention_mask=attention_mask
        )[0]  # [B, T, 768]

        # (b) Phonetic refinement — CNN + BiLSTM
        phonetic = self.phonetic_encoder(phonetic)  # [B, T, 768]

        # (c) Linguistic encoding — key & value projections
        h_k, h_v = self.linguistic_encoder(linguistic)
        # h_k: [B, N, 2304]  |  h_v: [B, N, 2304]

        # (d) Cross-modal attention — phonetic attends to linguistic
        attn_output, _ = self.multihead_attn(
            query=phonetic, key=h_k, value=h_v,
        )  # [B, T, 768]

        # (e) Gated fusion — learned per-frame weighting of the two streams
        gate_input = torch.cat((attn_output, phonetic), dim=2)       # [B, T, 1536]
        gate = torch.sigmoid(self.gate_proj(gate_input))             # [B, T, 768]
        fused: torch.Tensor = gate * attn_output + (1.0 - gate) * phonetic  # [B, T, 768]

        # (f) Phoneme output — pure audio features (bypasses gate)
        #     Decoupled so the gate can swing aggressively for detection
        #     without corrupting phoneme recognition.
        logits: torch.Tensor = self.output_projection(phonetic)      # [B, T, V]

        if return_detection:
            detection: torch.Tensor = self.detection_head(fused).squeeze(-1)  # [B, T]
            # Gate mean per frame (averaged over hidden dim) for regularisation
            gate_mean: torch.Tensor = gate.mean(dim=-1)  # [B, T]
            return logits, detection, gate_mean

        return logits
