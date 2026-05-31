import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Ensure project root is on sys.path so `from src...` imports work when
# running this script directly (python scripts/preprocess.py).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import soundfile as sf
from sklearn.model_selection import train_test_split

from src.utils.file_utils import load_config, save_json
from src.utils.file_utils import load_json

AUDIO_KEYS = ["audio", "audio_filepath", "wav", "path", "file", "filename"]
TRANSCRIPT_KEYS = ["transcript", "ipa", "text", "label"]
CANONICAL_KEYS = ["canonical", "canonical_transcript"]
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "|"]


def find_metadata_file(metadata_dir: Path) -> Path:
    """Return the most suitable metadata CSV from ``metadata_dir``.

    Prefers ``*_phones.csv`` (phoneme-level annotations) over readable-text
    CSVs, since phoneme tokens are required for both vocabulary alignment
    and the model's phoneme-prediction task.  If no phones file exists, the
    alphabetically-first CSV is returned.
    """
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    csv_files = sorted(metadata_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV metadata file found in {metadata_dir}."
        )

    # Prefer phoneme-level metadata when available
    phones_files = [p for p in csv_files if p.name.endswith("_phones.csv")]
    if phones_files:
        return phones_files[0]

    return csv_files[0]


def identify_columns(row: Dict[str, str]) -> Tuple[str, str]:
    lower_keys = {key.lower(): key for key in row.keys()}

    audio_key = next((lower_keys[k] for k in AUDIO_KEYS if k in lower_keys), None)
    transcript_key = next((lower_keys[k] for k in TRANSCRIPT_KEYS if k in lower_keys), None)
    canonical_key = next((lower_keys[k] for k in CANONICAL_KEYS if k in lower_keys), None)

    if audio_key is None or transcript_key is None:
        raise ValueError(
            "Could not find audio/transcript columns in metadata. "
            f"Expected one of {AUDIO_KEYS} for audio and one of {TRANSCRIPT_KEYS} for transcript. "
            f"Found columns: {list(row.keys())}"
        )

    return audio_key, transcript_key, canonical_key


def _get_audio_duration(audio_path: Path) -> Optional[float]:
    try:
        info = sf.info(str(audio_path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def load_dataset_rows(csv_path: Path, raw_audio_dir: Path, max_duration: Optional[float] = None) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if any((value or "").strip() for value in row.values())]

    if not rows:
        raise ValueError(f"Metadata CSV is empty: {csv_path}")

    audio_key, transcript_key, canonical_key = identify_columns(rows[0])
    collected = []
    for row in rows:
        audio_value = (row.get(audio_key) or "").strip()
        transcript_value = (row.get(transcript_key) or "").strip()
        canonical_value = (row.get(canonical_key) or "").strip() if canonical_key else ""

        if not audio_value or not transcript_value:
            continue

        audio_path = Path(audio_value)
        if not audio_path.is_absolute():
            candidate = raw_audio_dir / audio_path
            if candidate.exists():
                audio_path = candidate
            else:
                audio_path = csv_path.parent.parent / audio_value

        # Filter by duration without loading full file.
        if max_duration is not None:
            duration = _get_audio_duration(audio_path)
            if duration is None:
                # Can't read file info -> skip
                continue
            if duration > float(max_duration):
                continue

        collected.append({
            "audio_filepath": str(audio_path),
            "transcript": transcript_value,
            "canonical": canonical_value or transcript_value,
        })

    if not collected:
        raise ValueError(
            "No valid rows were found in metadata after normalizing paths. "
            f"Check {csv_path} and the audio directory {raw_audio_dir}."
        )

    return collected


def build_vocab(ipa_texts: List[str]) -> Dict[str, int]:
    """Build a token-level vocabulary from space-separated IPA phoneme strings.

    Each element in ``ipa_texts`` is a string of space-delimited phoneme
    tokens (e.g. ``"aː-0 m aː-4 ɗ"``).  The function splits on whitespace
    to collect unique *phoneme tokens* (not individual characters).
    """
    tokens: set[str] = set()
    for text in ipa_texts:
        tokens.update(text.split())

    tokens -= set(SPECIAL_TOKENS)
    sorted_tokens = sorted(tokens)
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for token in sorted_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def build_vocab_from_lexicon(lexicon_path: Path) -> Dict[str, int]:
    """Build vocabulary from a lexicon file.

    The lexicon is expected to be a file where each line has the format
    ``word PHONEME_SEQ`` — the word and the IPA phoneme sequence are
    separated by the **first** space on the line.  The phoneme sequence
    itself is a space-delimited series of IPA tokens (e.g. ``ɓ aː-0 nz``).

    All unique phoneme tokens found across every line are collected and
    assigned IDs, with the standard special tokens placed first.
    """
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")

    phonemes: set[str] = set()
    with lexicon_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on the *first* space to separate word from phoneme seq
            first_space = line.find(" ")
            if first_space == -1:
                continue
            ipa_column = line[first_space + 1:].strip()
            if ipa_column:
                phonemes.update(ipa_column.split())

    if not phonemes:
        raise ValueError(f"No phoneme tokens found in lexicon: {lexicon_path}")

    # Remove any special tokens that might appear as real phonemes (unlikely)
    tokens = phonemes - set(SPECIAL_TOKENS)
    sorted_tokens = sorted(tokens)

    # Build vocab: special tokens first, then alphabetically
    vocab: Dict[str, int] = {
        token: idx for idx, token in enumerate(SPECIAL_TOKENS)
    }
    for token in sorted_tokens:
        vocab[token] = len(vocab)

    return vocab


def split_dataset(rows: List[Dict[str, str]], validation_split: float, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    train_rows, val_rows = train_test_split(rows, test_size=validation_split, random_state=seed)
    return train_rows, val_rows


def save_split(rows: List[Dict[str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_filepath", "transcript", "canonical"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw IPA/audio metadata into train/val splits and vocab.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to project configuration YAML file.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata CSV file path. If omitted, the first CSV in data/raw/metadata is used.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = Path(config["data"]["raw_dir"])
    metadata_dir = Path(config["data"]["metadata_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    validation_split = float(config["data"].get("validation_split", 0.1))
    seed = int(config["training"].get("seed", 42))
    audio_dir = Path(config["data"].get("audio_dir", raw_dir))
    max_duration = config["data"].get("max_duration", None)

    metadata_path = Path(args.metadata) if args.metadata else find_metadata_file(metadata_dir)
    rows = load_dataset_rows(metadata_path, audio_dir, max_duration=max_duration)

    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Build vocabulary ───────────────────────────────────────────────────
    vocab_path = processed_dir / "vocab.json"
    lexicon_path = metadata_dir / "lexicon_vmd.txt"

    if vocab_path.exists():
        try:
            vocab = load_json(vocab_path)
            print(f"Loaded existing vocab with {len(vocab)} entries")
        except Exception:
            if lexicon_path.exists():
                vocab = build_vocab_from_lexicon(lexicon_path)
                print(f"Built vocab from lexicon ({len(vocab)} entries)")
            else:
                vocab = build_vocab([r["transcript"] for r in rows])
                print(f"Built vocab from metadata CSV ({len(vocab)} entries)")
    elif lexicon_path.exists():
        vocab = build_vocab_from_lexicon(lexicon_path)
        print(f"Built vocab from lexicon ({len(vocab)} entries)")
    else:
        vocab = build_vocab([r["transcript"] for r in rows])
        print(f"Built vocab from metadata CSV ({len(vocab)} entries)")

    save_json(vocab, vocab_path)

    train_rows, val_rows = split_dataset(rows, validation_split, seed)
    save_split(train_rows, processed_dir / "train_split.csv")
    save_split(val_rows, processed_dir / "val_split.csv")

    print(f"Saved vocab with {len(vocab)} entries to {vocab_path}")
    print(f"Saved {len(train_rows)} train rows to {processed_dir / 'train_split.csv'}")
    print(f"Saved {len(val_rows)} val rows to {processed_dir / 'val_split.csv'}")


if __name__ == "__main__":
    main()
