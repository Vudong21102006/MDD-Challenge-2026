"""
Beam-search CTC decoder with optional LM fusion (vocabulary prior).
"""

from typing import List, Tuple, Dict, Optional
import torch
import math


class BeamSearchDecoder:
    """Beam-search CTC decoder with vocabulary-based LM.

    Decodes frame-by-frame logits via beam search, keeping top-K hypotheses.
    Optionally blends a vocabulary prior (LM) with model logits.

    Args:
        vocab_size: Number of output tokens.
        blank_id: CTC blank token ID (default 0).
        beam_width: Number of hypotheses to track (default 3).
        lm_weight: Blend factor for LM prior [0, 1] (default 0).
        vocab_counts: Optional dict of token_id → count (for LM prior).
    """

    def __init__(
        self,
        vocab_size: int,
        blank_id: int = 0,
        beam_width: int = 3,
        lm_weight: float = 0.0,
        vocab_counts: Optional[Dict[int, float]] = None,
    ):
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.lm_weight = max(0.0, min(1.0, lm_weight))

        # Build LM prior from vocab counts (log probability)
        self.lm_prior = torch.zeros(vocab_size)
        if vocab_counts is not None and self.lm_weight > 0:
            total = sum(vocab_counts.values())
            for token_id, count in vocab_counts.items():
                if count > 0:
                    self.lm_prior[token_id] = math.log(count / total + 1e-8)
        else:
            # Uniform prior if not provided
            self.lm_prior.fill_(1.0 / vocab_size)

    def decode(self, logits: torch.Tensor) -> str:
        """Decode frame-by-frame logits using beam search.

        Args:
            logits: Tensor of shape [T, V] (time steps × vocabulary).

        Returns:
            Space-separated phoneme string.
        """
        device = logits.device
        lm_prior = self.lm_prior.to(device)

        # Initialize: beam of hypotheses, each as (prefix_ids, score)
        # prefix_ids: list of collapsed token IDs (excluding blanks)
        # score: cumulative log-prob
        beams = [
            ([], 0.0),  # empty hypothesis
        ]

        # Process each frame
        for t in range(logits.shape[0]):
            frame_logits = logits[t]  # [V]

            # Blend with LM prior
            if self.lm_weight > 0:
                blended = (
                    (1.0 - self.lm_weight) * frame_logits
                    + self.lm_weight * lm_prior
                )
            else:
                blended = frame_logits

            # Expand each hypothesis with top-K tokens
            new_beams = []
            for prefix_ids, prefix_score in beams:
                for token_id in range(self.vocab_size):
                    token_score = blended[token_id].item()

                    if token_id == self.blank_id:
                        # Blank: don't add to hypothesis
                        new_hyp = (prefix_ids, prefix_score + token_score)
                    elif len(prefix_ids) > 0 and prefix_ids[-1] == token_id:
                        # Merge with previous token (CTC rule)
                        new_hyp = (prefix_ids, prefix_score + token_score)
                    else:
                        # New token: append
                        new_hyp = (prefix_ids + [token_id], prefix_score + token_score)

                    new_beams.append((new_hyp[0], new_hyp[1]))

            # Keep top-K by score
            new_beams = sorted(new_beams, key=lambda x: -x[1])
            beams = new_beams[: self.beam_width]

        # Return best hypothesis
        best_ids, _ = beams[0]
        tokens = self._ids_to_tokens(best_ids)
        return " ".join(tokens)

    def set_id_to_token(self, id_to_token: Dict[int, str]) -> None:
        """Set token ID → string mapping for decoding output."""
        self.id_to_token = id_to_token

    def _ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert token IDs to strings, skipping blanks and <eps>."""
        tokens = []
        for token_id in ids:
            if token_id == self.blank_id:
                continue
            token = getattr(self, "id_to_token", {}).get(token_id, "")
            if token and token != "<eps>":
                tokens.append(token)
        return tokens


def build_vocab_counts(dataset) -> Dict[int, float]:
    """Build vocabulary frequency prior from dataset.

    Args:
        dataset: PhonemeDataset instance with tokenizer.

    Returns:
        Dict mapping token_id → count.
    """
    counts = {}
    for record in dataset.records:
        tokens = dataset.tokenizer.encode(record["canonical"])
        for token_id in tokens:
            counts[token_id] = counts.get(token_id, 0) + 1
    return counts
