import argparse
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.file_utils import load_config, save_json


def augment_waveform(waveform: np.ndarray, gain: float, noise_level: float) -> np.ndarray:
    return (waveform * gain) + (np.random.randn(*waveform.shape).astype(waveform.dtype) * noise_level)


def load_split_csv(csv_path: Path) -> List[dict]:
    import csv

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    return rows


def save_split(rows: List[dict], csv_path: Path) -> None:
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_filepath", "transcript", "canonical"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Augment and expand dataset by creating augmented audio copies.")
    parser.add_argument("--input-split", required=True, help="Path to existing split CSV (train or combined CSV)")
    parser.add_argument("--output-split", required=True, help="Path to write expanded split CSV (will overwrite)")
    parser.add_argument("--copies-per-file", type=int, default=50, help="Number of augmented copies to generate per original file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_gain", type=float, default=1.2)
    parser.add_argument("--min_gain", type=float, default=0.8)
    parser.add_argument("--noise_level", type=float, default=0.005)
    parser.add_argument("--config", default="config.yaml")

    args = parser.parse_args()
    random.seed(args.seed)

    config = load_config(args.config)
    audio_dir = Path(config["data"].get("audio_dir", "data/raw/audio_data"))

    in_rows = load_split_csv(Path(args.input_split))

    out_rows = []
    out_audio_dir = audio_dir
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    for row in in_rows:
        raw_fp = row.get("audio_filepath", "")
        src_path = Path(raw_fp)
        # Resolve common possibilities without creating incorrect nested paths
        if not src_path.exists():
            # try as name inside audio_dir
            candidate = out_audio_dir / src_path.name
            if candidate.exists():
                src_path = candidate
            else:
                # try joining the raw value (if it contains subdirs)
                candidate2 = out_audio_dir / raw_fp
                if candidate2.exists():
                    src_path = candidate2
                else:
                    # give up on this row
                    continue
        try:
            waveform, sr = sf.read(str(src_path))
        except Exception:
            continue

        # keep original
        out_rows.append({"audio_filepath": str(src_path), "transcript": row.get("transcript", ""), "canonical": row.get("canonical", "")})

        for i in range(args.copies_per_file):
            gain = random.uniform(args.min_gain, args.max_gain)
            noise_level = args.noise_level * random.uniform(0.5, 1.5)
            aug = augment_waveform(waveform.astype('float32'), gain, noise_level)
            out_name = f"aug_{src_path.stem}_{i:03d}.wav"
            out_path = out_audio_dir / out_name
            sf.write(str(out_path), aug, sr)
            out_rows.append({"audio_filepath": str(out_path), "transcript": row.get("transcript", ""), "canonical": row.get("canonical", "")})

    # shuffle and optionally split using config validation split
    train_rows, val_rows = train_test_split(out_rows, test_size=config["data"].get("validation_split", 0.1), random_state=args.seed)

    # If output-split path contains 'train' write train_rows else write combined
    out_path = Path(args.output_split)
    if "train" in out_path.name.lower():
        save_split(train_rows, out_path)
    elif "val" in out_path.name.lower():
        save_split(val_rows, out_path)
    else:
        # write combined
        save_split(out_rows, out_path)

    print(f"Wrote {len(out_rows)} total rows; train {len(train_rows)}, val {len(val_rows)} to {out_path}")


if __name__ == "__main__":
    main()
