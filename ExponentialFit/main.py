"""Load generated exponential-curve datasets for a PyTorch model."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Select a GPU when one is available; otherwise use the CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


def load_dataset(split: str, batch_size: int) -> DataLoader:
    """Load one saved split as batches of padded curves."""
    if split not in ("train", "test"):
        raise ValueError("split must be either 'train' or 'test'")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    path = Path(__file__).resolve().parent / "data" / split / "curves.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run data_generator.py to create the dataset."
        )

    with np.load(path, allow_pickle=False) as data:
        # These names and meanings match export_dataset() in data_generator.py.
        required_keys = (
            "x_points",
            "y_points",
            "y_true",
            "valid_mask",
            "curve_params",
            "outlier_mask",
        )
        missing_keys = [key for key in required_keys if key not in data.files]
        if missing_keys:
            raise ValueError(
                f"{path} is missing {missing_keys}. Regenerate it with "
                "data_generator.py."
            )

        # copy() makes arrays independent of the NPZ file before it is closed.
        tensors = [torch.from_numpy(data[key].copy()) for key in required_keys]

    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

#define pointFeatures nn
class pointFeatures(nn.Module):
    def __init__(self, feature_size: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
        )
    def forward(self, inputs):
        return self.network(inputs)

#define parameterPredictor nn
class parameterPredictor(nn.Module):
    def __init__(self, feature_size: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, 3)
        )
    def forward(self, curve_features):
        return self.network(curve_features)

#load data to DEVICE
if __name__ == "__main__":

    #load datasets
    batch_size = int(input("batch_size= "))
    train_loader = load_dataset("train", batch_size)
    test_loader = load_dataset("test", batch_size)

    #load networks
    parameter_predictor = parameterPredictor().to(DEVICE)
    point_features = pointFeatures().to(DEVICE)
    loss_function = nn.MSELoss()

    #optimizer setup
    lr = float(input("learning_rate = "))
    momentum = float(input("momentum = "))
    optimizer = torch.optim.SGD(
        list(point_features.parameters()) + list(parameter_predictor.parameters()),
        lr = lr,
        momentum = momentum
    )

    #enter #epochs
    epochs = int(input("epochs ="))
    for epoch in range(epochs):
        #zero counts
        total_loss = 0.0
        total_curves = 0

        for batch in train_loader:
            #break down batch
            x_points, y_points, y_true, valid_mask, curve_params, outlier_mask = batch

            #stack x, y pairs into inputs [B, N, 2]
            inputs = torch.stack((x_points, y_points), dim = -1).to(DEVICE)

            #load mask [B, N]
            mask = valid_mask.to(DEVICE)
            #fit mask to [B, N, 1]
            mask_3d = mask.unsqueeze(-1)

            #call feature assigner
            features = point_features(inputs)

            #sum and average features
            feature_sum = (features * mask_3d).sum(dim=1)
            point_count = mask_3d.sum(dim=1).clamp(min=1)
            curve_features = feature_sum / point_count

            #call parameter predictor
            parameters = parameter_predictor(curve_features)

            #call loss
            targets = curve_params.to(DEVICE).float()
            loss = loss_function(parameters, targets)

            #backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_now = targets.size(0)
            total_loss += loss.item() * batch_size_now
            total_curves += batch_size_now

        average_loss = total_loss / total_curves
        print(f"Epoch {epoch + 1}/{epochs}, average loss: {average_loss:.6f}")






    

