from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


class DataCollatorCTCWithPadding:
    def __init__(
        self,
        padding_value: float = 0.0,
        transcript_pad_token_id: int = -100,
        canonical_pad_token_id: int = 0,
    ):
        """Data collator that pads audio inputs and two label streams.

        - `transcript_labels` are padded with `-100` for CTC loss ignore_index.
        - `canonical_labels` are padded with the normal pad token id.
        """
        self.padding_value = padding_value
        self.transcript_pad_token_id = transcript_pad_token_id
        self.canonical_pad_token_id = canonical_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_values = [feature["input_values"] for feature in features]
        transcript_labels = [feature["transcript_labels"] for feature in features]
        canonical_labels = [feature["canonical_labels"] for feature in features]

        # Pad inputs (waveforms). Each is a 1D tensor of samples.
        padded_inputs = pad_sequence(
            [iv if iv.ndim == 1 else iv.squeeze(0) for iv in input_values],
            batch_first=True,
            padding_value=self.padding_value,
        )

        # Attention mask: 1 where samples exist, 0 where padded.
        attention_mask = torch.stack(
            [torch.cat([torch.ones(iv.shape[0], dtype=torch.long), torch.zeros(padded_inputs.shape[1] - iv.shape[0], dtype=torch.long)]) for iv in input_values]
        )

        padded_transcript = pad_sequence(
            transcript_labels,
            batch_first=True,
            padding_value=self.transcript_pad_token_id,
        )

        padded_canonical = pad_sequence(
            canonical_labels,
            batch_first=True,
            padding_value=self.canonical_pad_token_id,
        )

        return {
            "input_values": padded_inputs,
            "attention_mask": attention_mask,
            "transcript_labels": padded_transcript,
            "canonical_labels": padded_canonical,
        }
