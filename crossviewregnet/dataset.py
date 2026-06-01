import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class TreeNPZDataset(torch.utils.data.Dataset):
    """Dataset for preprocessed individual-tree NPZ tensors."""

    def __init__(
        self,
        csv_path,
        data_dir,
        transform=None,
        filename_column="filename",
        label_column="species_id",
        height_column="tree_H",
        height_mean=None,
        height_std=None,
        height_noise=0.0,
        require_labels=True,
    ):
        self.frame = pd.read_csv(csv_path)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.filename_column = filename_column
        self.label_column = label_column
        self.height_column = height_column
        self.height_mean = height_mean
        self.height_std = height_std
        self.height_noise = height_noise
        self.require_labels = require_labels

        if filename_column not in self.frame.columns:
            raise ValueError(f"Missing filename column: {filename_column}")
        if require_labels and label_column not in self.frame.columns:
            raise ValueError(f"Missing label column: {label_column}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        identifier = str(row[self.filename_column])
        npz_path = self.data_dir / f"{Path(identifier).stem}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)

        with np.load(npz_path) as data:
            images = data["images"]
            stored_height = float(np.asarray(data["height"]).reshape(-1)[0])

        images = torch.from_numpy(images).float()
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if self.transform is not None:
            images = self.transform(images)

        height_value = float(row[self.height_column]) if self.height_column in row else stored_height
        if self.height_noise > 0:
            height_value = height_value + torch.randn(1).item() * self.height_noise
        if self.height_mean is not None and self.height_std is not None:
            height_value = (height_value - self.height_mean) / (self.height_std + 1e-8)
        height_tensor = torch.tensor([height_value], dtype=torch.float32)

        if self.require_labels:
            label = torch.tensor(int(row[self.label_column]), dtype=torch.long)
            return images, height_tensor, label
        return images, height_tensor, identifier

    def sampler_weights(self):
        if "weight" in self.frame.columns:
            return torch.tensor(self.frame["weight"].values, dtype=torch.double)
        return None


def filename_stem(value):
    return os.path.splitext(os.path.basename(str(value)))[0]
