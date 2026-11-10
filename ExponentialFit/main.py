import numpy as np
import torch
import torch.nn as nn
import struct
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

#define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

#define training dataset loader
x=int(input("batch_size= "))

def load_dataset(split, batch_size=x):
    if split not in ("train", "test"):
        raise ValueError("split musi mieć wartość 'train' albo 'test'")

    path = Path(__file__).parent / "data" / split / "curves.npz"

    with np.load(path, allow_pickle=False) as data:
        dataset = TensorDataset(
            torch.from_numpy(data["x"].copy()),
            torch.from_numpy(data["y"].copy()),
            torch.from_numpy(data["y_clean"].copy()),
            torch.from_numpy(data["mask"].copy()),
            torch.from_numpy(data["params"].copy()),
            torch.from_numpy(data["n_points"].copy()),
            torch.from_numpy(data["outlier_mask"].copy()),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )


train_loader = load_dataset("train")
test_loader = load_dataset("test")

#define structure
point_features = 2
embedding_size = 64
score_hidden_size = 64
#a, b, c
parameter_count = 3
