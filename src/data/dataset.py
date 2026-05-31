import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import random
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T

from src.utils.audio_utils import load_and_resample, to_mono


class PhonemeTokenizer:
    def __init__(self, vocab_path: str):
        self.vocab_path = Path(vocab_path)
        self.vocab = self._load_vocab(self.vocab_path)
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 1)

    def _load_vocab(self, path: Path) -> Dict[str, int]:
        with path.open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        if not isinstance(vocab, dict):
            raise ValueError(f"Vocab file must be a JSON object: {path}")
        return vocab

    def encode(self, text: str) -> List[int]:
        """Encode a space-separated phoneme sequence into token IDs.

        Args:
            text: Space-delimited IPA phoneme string
                  (e.g. ``"aː-0 m aː-4 ɗ"``).

        Returns:
            List of integer token IDs, one per phoneme token.
        """
        return [
            self.vocab.get(token, self.unk_token_id)
            for token in text.split()
        ]

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to a space-separated phoneme string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Space-separated phoneme string.
        """
        reverse = {idx: token for token, idx in self.vocab.items()}
        return " ".join(reverse.get(i, self.unk_token) for i in ids)


class PhonemeDataset(Dataset):
    """Dataset returning raw waveforms and two label streams: canonical and transcript.

    The dataset uses `load_and_resample` (soundfile + torchaudio resample) to
    avoid librosa on Windows. Optionally applies simple noise augmentation.
    """

    def __init__(
        self,
        split_csv: str,
        vocab_path: str,
        audio_dir: Optional[str] = None,
        target_sr: int = 16000,
        add_noise: bool = False,
        noise_prob: float = 0.2,
        noise_level: float = 0.005,
    ):
        self.split_csv = Path(split_csv)
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.target_sr = target_sr
        self.tokenizer = PhonemeTokenizer(vocab_path)
        self.records = self._load_split_csv(self.split_csv)
        self.add_noise = add_noise
        self.noise_prob = noise_prob
        self.noise_level = noise_level
        # small Torchaudio transform used for random gain
        self._vol = T.Vol(gain=0.0, gain_type="amplitude")

    def _load_split_csv(self, csv_path: Path) -> List[Dict[str, Any]]:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if row.get("audio_filepath") and (row.get("transcript") or row.get("canonical"))]

        if not rows:
            raise ValueError(f"Split CSV must contain rows with 'audio_filepath' and 'transcript'/'canonical': {csv_path}")

        normalized = []
        for row in rows:
            audio_path = Path(row["audio_filepath"])
            if not audio_path.is_absolute() and self.audio_dir:
                candidate = self.audio_dir / audio_path
                if candidate.exists():
                    audio_path = candidate
            transcript = (row.get("transcript") or "").strip()
            canonical = (row.get("canonical") or transcript).strip()
            normalized.append({
                "audio_filepath": str(audio_path),
                "transcript": transcript,
                "canonical": canonical,
            })
        return normalized

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        audio_path = record["audio_filepath"]
        waveform, sample_rate = load_and_resample(audio_path, target_sr=self.target_sr)
        waveform = to_mono(waveform).squeeze(0)  # shape: (num_samples,)

        # Data augmentation: random gain + additive gaussian noise
        if self.add_noise and random.random() < self.noise_prob:
            # random gain between 0.8 and 1.2
            gain = random.uniform(0.8, 1.2)
            self._vol.gain = gain
            waveform = self._vol(waveform.unsqueeze(0)).squeeze(0)
            noise = torch.randn_like(waveform) * self.noise_level
            waveform = waveform + noise

        transcript_ids = self.tokenizer.encode(record["transcript"])
        canonical_ids = self.tokenizer.encode(record["canonical"])

        transcript_tensor = torch.tensor(transcript_ids, dtype=torch.long)
        canonical_tensor = torch.tensor(canonical_ids, dtype=torch.long)

        return {
            "input_values": waveform,
            "transcript_labels": transcript_tensor,
            "canonical_labels": canonical_tensor,
            "audio_path": audio_path,
            "sample_rate": sample_rate,
        }
