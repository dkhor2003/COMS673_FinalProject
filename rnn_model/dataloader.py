import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class FallDataset(Dataset):
    def __init__(self, data_dir, seq_len=20):
        self.seq_len = seq_len
        self.samples = []

        all_csvs = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        for file in all_csvs:
            try:
                df = pd.read_csv(os.path.join(data_dir, file))
                df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]

                if 'fallover' not in df.columns or df.shape[0] < seq_len:
                    continue

                feature_cols = [col for col in df.columns if col != 'fallover']

                self.extra_feature_drops = getattr(self, 'extra_feature_drops', {})
                if len(feature_cols) > 54:
                    extra = feature_cols[54:]
                    for col in extra:
                        self.extra_feature_drops[col] = self.extra_feature_drops.get(col, 0) + 1
                    feature_cols = feature_cols[:54]

                elif len(feature_cols) < 54:
                    print(f"[SKIP] {file} has insufficient features ({len(feature_cols)} < 54)")
                    continue

                features = df[feature_cols].values
                labels = df['fallover'].values

                for i in range(len(df) - seq_len):
                    x_seq = features[i:i+seq_len]
                    y = labels[i+seq_len-1]
                    self.samples.append((x_seq, y))

            except Exception as e:
                print(f"[ERROR] Could not load {file}: {e}")
                
        if self.extra_feature_drops:
            for col, count in self.extra_feature_drops.items():
                print(f"[WARN] Dropped extra column '{col}' from {count} files")

        print(f"[DONE] Total sequences loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_dataloaders(data_dir, batch_size=32, seq_len=20, val_split=0.2):
    dataset = FallDataset(data_dir, seq_len)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(f"[ERROR] Not enough data. Only {len(dataset)} samples loaded.")
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size)
    )
