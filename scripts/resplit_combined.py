import csv
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

combined_paths = [
    'data/processed/train_split_expanded.csv',
    'data/processed/val_split_expanded.csv'
]
combined = []
for p in combined_paths:
    with open(p, 'r', encoding='utf-8') as f:
        combined += list(csv.DictReader(f))

cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))
val_ratio = cfg['data'].get('validation_split', 0.1)
seed = cfg['training'].get('seed', 42)
train_rows, val_rows = train_test_split(combined, test_size=val_ratio, random_state=seed)

Path('data/processed').mkdir(parents=True, exist_ok=True)
with open('data/processed/train_final.csv','w',encoding='utf-8',newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['audio_filepath','transcript','canonical'])
    writer.writeheader(); writer.writerows(train_rows)
with open('data/processed/val_final.csv','w',encoding='utf-8',newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['audio_filepath','transcript','canonical'])
    writer.writeheader(); writer.writerows(val_rows)

print('Wrote train_final:', len(train_rows), 'val_final:', len(val_rows))
