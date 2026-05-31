from typing import Any, Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


class DataCollatorCTCWithPadding:
    """Pads variable-length waveforms and label sequences into a uniform batch.

    Returns a **tuple** of six tensors designed to feed directly into
    :class:`MDDTrainer` and :class:`MDDModelBuilder`::

        (input_values, linguistic, transcript, target_lengths,
         input_lengths, attention_mask)

    - ``transcript_labels`` are padded with ``-100`` (CTC ``ignore_index``)
      so that padded positions are excluded from the CTC loss.
    - ``canonical_labels`` are padded with ``0`` (the ``[PAD]`` token ID)
      since they are fed into the :class:`LinguisticEncoder`, not the CTC
      loss.
    - ``input_lengths`` records the original sample count of each waveform
      *before* padding, so that the trainer can compute the down-sampled
      CTC time dimension via
      ``wav2vec2._get_feat_extract_output_lengths()``.
    - ``target_lengths`` records the original number of phoneme tokens in
      each ``transcript_labels`` sequence, required by
      :func:`torch.nn.CTCLoss`.
    - ``attention_mask`` marks real samples (1) vs. padding (0) for the
      Wav2Vec2 transformer layers.
    """

    def __init__(
        self,
        padding_value: float = 0.0,
        transcript_pad_token_id: int = -100,
        canonical_pad_token_id: int = 0,
    ) -> None:
        self.padding_value = padding_value
        self.transcript_pad_token_id = transcript_pad_token_id
        self.canonical_pad_token_id = canonical_pad_token_id

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ── Separate fields ─────────────────────────────────────────────────
        input_values = [feature["input_values"] for feature in features]
        transcript_labels = [feature["transcript_labels"] for feature in features]
        canonical_labels = [feature["canonical_labels"] for feature in features]

        # ── Capture original lengths BEFORE any padding ─────────────────────
        input_lengths = torch.tensor(
            [iv.shape[0] for iv in input_values], dtype=torch.long
        )
        target_lengths = torch.tensor(
            [tl.shape[0] for tl in transcript_labels], dtype=torch.long
        )

        # ── Pad waveforms (each is a 1-D tensor of float samples) ───────────
        padded_inputs = pad_sequence(
            [iv if iv.ndim == 1 else iv.squeeze(0) for iv in input_values],
            batch_first=True,
            padding_value=self.padding_value,
        )  # → [B, T_audio]

        # ── Attention mask: 1 for real samples, 0 for padding ───────────────
        attention_mask = torch.stack(
            [
                torch.cat(
                    [
                        torch.ones(iv.shape[0], dtype=torch.long),
                        torch.zeros(
                            padded_inputs.shape[1] - iv.shape[0], dtype=torch.long
                        ),
                    ]
                )
                for iv in input_values
            ]
        )  # → [B, T_audio]

        # ── Pad label sequences ─────────────────────────────────────────────
        padded_transcript = pad_sequence(
            transcript_labels,
            batch_first=True,
            padding_value=self.transcript_pad_token_id,
        )  # → [B, N_transcript]    pad = -100 (CTC ignore_index)

        padded_canonical = pad_sequence(
            canonical_labels,
            batch_first=True,
            padding_value=self.canonical_pad_token_id,
        )  # → [B, N_canonical]     pad = 0 ([PAD] token)

        return (
            padded_inputs,      # input_values    [B, T_audio]
            padded_canonical,   # linguistic      [B, N_canonical]
            padded_transcript,  # transcript      [B, N_transcript]
            target_lengths,     # target_lengths  [B]
            input_lengths,      # input_lengths   [B]
            attention_mask,     # attention_mask  [B, T_audio]
        )
