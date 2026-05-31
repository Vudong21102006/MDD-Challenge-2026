import sys
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

# Ensure src package is importable when running the script from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Create synthetic test data in an isolated temp directory ─────────────────
TEST_DIR = Path("data/smoke_test")
TEST_AUDIO = TEST_DIR / "audio"
TEST_PROCESSED = TEST_DIR / "processed"

# Clean up any previous smoke test artifacts
if TEST_DIR.exists():
    shutil.rmtree(str(TEST_DIR))

TEST_AUDIO.mkdir(parents=True, exist_ok=True)
TEST_PROCESSED.mkdir(parents=True, exist_ok=True)
sr = 16000

# 5 short test WAVs (1 second at 16 kHz, random noise)
for i in range(1, 6):
    arr = np.random.uniform(-0.1, 0.1, sr).astype("float32")
    sf.write(str(TEST_AUDIO / f"audio_{i:03d}.wav"), arr, sr)
print("WAVs created")

# Synthetic token-level vocab (matching the space-split phoneme tokenizer)
vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "b": 4, "c": 5, "d": 6, "e": 7}
vocab_path = TEST_PROCESSED / "vocab.json"
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)
print("vocab.json created")

# Synthetic train split CSV with varying-length samples so padding is exercised
split_path = TEST_PROCESSED / "train_split.csv"
with open(split_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["audio_filepath", "transcript", "canonical"])
    writer.writeheader()
    samples = [
        ("a b c", "a b c"),           # 3 tokens / 3 tokens
        ("a", "a b c d e"),           # 1 token  / 5 tokens
        ("a b", "a b"),               # 2 tokens / 2 tokens
        ("a b c d e", "a"),           # 5 tokens / 1 token
        ("a b c", "a b c d"),         # 3 tokens / 4 tokens
    ]
    for i, (transcript, canonical) in enumerate(samples, start=1):
        writer.writerow(
            {
                "audio_filepath": str(TEST_AUDIO / f"audio_{i:03d}.wav"),
                "transcript": transcript,
                "canonical": canonical,
            }
        )
print("train_split.csv created")

# ── Smoke test: import data pipeline and fetch a batch ───────────────────────
from src.data.dataset import PhonemeDataset
from src.data.collator import DataCollatorCTCWithPadding
from torch.utils.data import DataLoader

dataset = PhonemeDataset(
    str(split_path),
    str(vocab_path),
    audio_dir=str(TEST_AUDIO),
    add_noise=True,
)
collator = DataCollatorCTCWithPadding()
loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

# Collator returns a 6-tuple, matching what MDDTrainer expects
(
    input_values,
    linguistic,
    transcript,
    target_lengths,
    input_lengths,
    attention_mask,
) = next(iter(loader))

print("\nBatch shapes:")
print(f"  input_values:     {input_values.shape}       # [B, T_audio]")
print(f"  linguistic:       {linguistic.shape}           # [B, N_canonical]")
print(f"  transcript:       {transcript.shape}           # [B, N_transcript]")
print(f"  target_lengths:   {target_lengths}             # original transcript lengths")
print(f"  input_lengths:    {input_lengths}              # original sample counts")
print(f"  attention_mask:   {attention_mask.shape}       # [B, T_audio]")
print(f"\n  transcript pad value: -100 (CTC ignore) -> {transcript[1, -1].item()}")
print(f"  linguistic pad value: 0    ([PAD] token)  -> {linguistic[0, -1].item()}")

# ── Clean up ─────────────────────────────────────────────────────────────────
shutil.rmtree(str(TEST_DIR))
print("\n[PASS] Data pipeline smoke test passed!")
