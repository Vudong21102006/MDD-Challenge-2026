import sys
from pathlib import Path
import numpy as np
import soundfile as sf

# Ensure src package is importable when running the script from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ad = Path('data/raw/audio_data')
ad.mkdir(parents=True, exist_ok=True)
# create 5 short test wavs (1s at 16k)
sr = 16000
for i in range(1,6):
    arr = (np.random.uniform(-0.1,0.1, sr).astype('float32'))
    sf.write(str(ad / f'audio_{i:03d}.wav'), arr, sr)
print('wavs created')

# Smoke test: import classes and fetch a batch
from src.data.dataset import PhonemeDataset
from src.data.collator import DataCollatorCTCWithPadding
from torch.utils.data import DataLoader

dataset = PhonemeDataset('data/processed/train_split.csv', 'data/processed/vocab.json', audio_dir='data/raw/audio_data', add_noise=True)
collator = DataCollatorCTCWithPadding()
loader = DataLoader(dataset, batch_size=2, collate_fn=collator)
b = next(iter(loader))
print('Batch keys:', list(b.keys()))
print('input_values shape:', b['input_values'].shape)
print('attention_mask shape:', b['attention_mask'].shape)
print('transcript_labels shape:', b['transcript_labels'].shape)
print('canonical_labels shape:', b['canonical_labels'].shape)
