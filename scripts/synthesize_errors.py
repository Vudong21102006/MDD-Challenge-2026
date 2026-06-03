"""
Synthesize mispronunciation training data.

Reads the training split CSV and the lexicon, then creates phonetically-
plausible mispronunciations for correct-pronunciation samples by
substituting phonemes in the transcript while keeping the canonical
unchanged.

Usage::

    python scripts/synthesize_errors.py

Output: ``data/processed/train_split_synthetic.csv`` containing the
original rows plus the synthetic error rows.
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_CSV: str = "data/processed/train_split.csv"
LEXICON_PATH: str = "data/raw/metadata/lexicon_vmd.txt"
VOCAB_PATH: str = "data/processed/vocab.json"
OUTPUT_CSV: str = "data/processed/train_split_synthetic.csv"

# ── Synthesis parameters ─────────────────────────────────────────────────────
ERRORS_PER_SAMPLE_MIN: int = 1      # minimum phoneme errors per synthetic sample
ERRORS_PER_SAMPLE_MAX: int = 3      # maximum
SYNTHESIS_RATIO: float = 0.6        # fraction of correct samples to synthesize
SEED: int = 42


def _load_vocab(path: str) -> Dict[str, int]:
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_phoneme_groups(vocab: Dict[str, int]) -> Dict[str, List[str]]:
    """Group phonemes by phonetic similarity for plausible substitutions.

    Returns a dict mapping each phoneme to a list of substitute candidates.
    """
    # Separate special tokens
    phonemes = [t for t in vocab if not t.startswith("[")]
    vowels: List[str] = []
    consonants: List[str] = []

    vowel_chars = set("aeiouɨəɔɛ")
    for p in phonemes:
        stripped = p.rstrip("012345")
        if any(c in vowel_chars for c in stripped if c not in "-"):
            vowels.append(p)
        else:
            consonants.append(p)

    # ── Vowel groups by base ──────────────────────────────────────────────
    vowel_base: Dict[str, List[str]] = defaultdict(list)
    for v in vowels:
        parts = v.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            base = parts[0]
        else:
            base = v
        vowel_base[base].append(v)

    # ── Consonant groups by ending pattern ────────────────────────────────
    cons_endings: Dict[str, List[str]] = defaultdict(list)
    for c in consonants:
        if c in ("|",):
            continue
        if c.endswith("z"):
            # Group by last 1-2 chars before z
            ending = c[-2:] if len(c) >= 2 else c
            cons_endings[ending].append(c)
        else:
            cons_endings["plain"].append(c)

    # ── Build substitute map ──────────────────────────────────────────────
    substitutes: Dict[str, List[str]] = {}

    for p in phonemes:
        if p in ("|",):
            substitutes[p] = [p]  # never substitute the word separator
            continue

        candidates: Set[str] = set()

        # Determine if vowel or consonant
        stripped = p.rstrip("012345")
        is_vowel = any(c in vowel_chars for c in stripped if c not in "-")

        if is_vowel:
            # Same base, different tone
            parts = p.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                base = parts[0]
                tone = int(parts[1])
                for t in range(6):
                    if t != tone and f"{base}-{t}" in vocab:
                        candidates.add(f"{base}-{t}")
            # Same tone, different base (pick up to 3 similar bases)
            if len(parts) == 2:
                tone = parts[1]
                for base, group in vowel_base.items():
                    if base != parts[0] and len(group) >= 2:
                        for v in group:
                            if v.endswith(f"-{tone}"):
                                candidates.add(v)
        else:
            # Similar consonant — same ending pattern
            for ending, group in cons_endings.items():
                if p in group:
                    candidates.update(g for g in group if g != p)
                    break
            # Also add some random plain consonants
            if len(candidates) < 3:
                candidates.update(
                    random.sample(
                        [c for c in cons_endings.get("plain", []) if c != p],
                        min(3, len(cons_endings.get("plain", []))),
                    )
                )

        if candidates:
            substitutes[p] = list(candidates)
        else:
            substitutes[p] = [p]

    return substitutes


def _synthesize_error(
    phonemes: List[str],
    substitutes: Dict[str, List[str]],
    num_errors: int,
) -> List[str]:
    """Create a synthetic mispronunciation by substituting phonemes.

    Args:
        phonemes:    Original phoneme sequence (from canonical/transcript).
        substitutes: Map of each phoneme to plausible replacement candidates.
        num_errors:  Number of phoneme positions to modify.

    Returns:
        Modified phoneme sequence with substitutions applied.
    """
    # Pick positions to modify (exclude word separators '|')
    eligible = [i for i, p in enumerate(phonemes) if p != "|"]
    if len(eligible) < num_errors:
        num_errors = max(1, len(eligible))

    positions = sorted(random.sample(eligible, min(num_errors, len(eligible))))

    result = list(phonemes)
    for pos in positions:
        original = phonemes[pos]
        candidates = substitutes.get(original, [original])
        if candidates and candidates != [original]:
            result[pos] = random.choice(candidates)

    return result


def main() -> None:
    random.seed(SEED)

    # ── Load data ──────────────────────────────────────────────────────────
    vocab = _load_vocab(VOCAB_PATH)
    substitutes = _build_phoneme_groups(vocab)
    print(f"Phoneme substitution map: {len(substitutes)} entries")

    # Show a few examples
    for p, subs in random.sample(
        [(k, v) for k, v in substitutes.items() if len(v) >= 2],
        min(5, len(substitutes)),
    ):
        print(f"  {p:8s} → {subs[:5]}")

    # ── Read training data ─────────────────────────────────────────────────
    with open(TRAIN_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    correct_rows = [r for r in rows if r["canonical"] == r["transcript"]]
    mis_rows = [r for r in rows if r["canonical"] != r["transcript"]]
    print(
        f"\nOriginal: {len(rows)} total, "
        f"{len(correct_rows)} correct, {len(mis_rows)} mispronounced"
    )

    # ── Synthesize errors ──────────────────────────────────────────────────
    num_to_synthesize = int(len(correct_rows) * SYNTHESIS_RATIO)
    selected = random.sample(correct_rows, num_to_synthesize)

    synthetic_rows: List[Dict[str, str]] = []
    for row in selected:
        phonemes = row["canonical"].strip().split()
        if len(phonemes) < 2:
            continue

        num_errors = random.randint(ERRORS_PER_SAMPLE_MIN, ERRORS_PER_SAMPLE_MAX)
        synthetic_transcript = _synthesize_error(phonemes, substitutes, num_errors)

        synthetic_rows.append(
            {
                "audio_filepath": row["audio_filepath"],
                "transcript": " ".join(synthetic_transcript),
                "canonical": row["canonical"],
            }
        )

    print(f"Synthesized: {len(synthetic_rows)} new error samples")

    # ── Combine and shuffle ────────────────────────────────────────────────
    all_rows = rows + synthetic_rows
    random.shuffle(all_rows)

    # Count final distribution
    final_correct = sum(1 for r in all_rows if r["canonical"] == r["transcript"])
    final_mis = len(all_rows) - final_correct
    print(
        f"Final: {len(all_rows)} total, "
        f"{final_correct} correct ({final_correct/len(all_rows)*100:.1f}%), "
        f"{final_mis} mispronounced ({final_mis/len(all_rows)*100:.1f}%)"
    )

    # ── Save ───────────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["audio_filepath", "transcript", "canonical"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {OUTPUT_CSV}")
    print(
        f"  Mispronunciation ratio improved from "
        f"{len(mis_rows)/len(rows)*100:.1f}% → {final_mis/len(all_rows)*100:.1f}%"
    )


if __name__ == "__main__":
    main()
