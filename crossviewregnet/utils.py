import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform():
    gray_mean, gray_std = imagenet_gray_stats()
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=20)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
            transforms.Normalize(mean=[gray_mean], std=[gray_std]),
        ]
    )


def build_eval_transform():
    gray_mean, gray_std = imagenet_gray_stats()
    return transforms.Normalize(mean=[gray_mean], std=[gray_std])


def imagenet_gray_stats():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return sum(imagenet_mean) / 3.0, sum(imagenet_std) / 3.0


def height_stats(csv_path, height_column="tree_H"):
    frame = pd.read_csv(csv_path)
    values = frame[height_column].astype(float).values
    return float(np.mean(values)), float(np.std(values))


def classification_metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
