from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Select a GPU when one is available; otherwise use the CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

#load dataset
def load_dataset():

      #define file path
      data_dir = Path(__file__).resolve().parent / "data"
      curves_path = data_dir / "curves.npz"
      parameters_path = data_dir / "parameters.pt"

      #verify path
      if not data_dir.exists():
            raise ValueError(f"couldnt find {data_dir}")
      if not curves_path.exists():
            raise ValueError(f"couldnt find {curves_path}")
      if not parameters_path.exists():
            raise ValueError(f"couldnt find {parameters_path}")

      #verify data structure
      curve_keys = (
            "x_points",
            "y_points",
            "valid_mask",
      )
      with np.load(curves_path, allow_pickle=False) as data:
            missing_keys = [key for key in curve_keys if key not in data.files]
            if missing_keys:
                  raise ValueError(f"{curves_path} is missing {missing_keys}")

      curve_tensors = [torch.from_numpy(data[key].copy()) for key in curve_keys]

      curves = {key: torch.from_numpy(data[key].copy()) for key in curve_keys}

      parameters = torch.load(
            parameters_path,
            map_location = DEVICE,
            weights_only = False,
      )

      return curves, parameters

#define pointFeatures nn
class pointFeatures(nn.Module):
      def __init__(self, feature_size: int = 128):
            super().__init__()
            self.network = nn.Sequential(
                  nn.Linear(2, feature_size),
                  nn.ReLU(),
                  nn.Linear(feature_size, feature_size),
                  nn.ReLU(),
                  nn.Linear(feature_size, feature_size),
                  nn.ReLU(),
            )
      def forward(self, inputs):
            return self.network(inputs)

#define parameterPredictor nn
class parameterPredictor(nn.Module):
      def __init__(self, feature_size: int = 128):
            super().__init__()
            self.network = nn.Sequential(
                  nn.Linear(feature_size, feature_size),
                  nn.ReLU(),
                  nn.Linear(feature_size, feature_size),
                  nn.ReLU(),
                  nn.Linear(feature_size, feature_size),
                  nn.ReLU(),
                  nn.Linear(feature_size, 3)
            )
      def forward(self, curve_features):
            return self.network(curve_features)

if __name__ == "__main__":

      #load dataset
      curves, parameters = load_dataset()

      #load networks to DEVICE
      parameter_predictor = parameterPredictor().to(DEVICE)
      point_features = pointFeatures().to(DEVICE)

      #load weights and biases
      point_features.load_state_dict(parameters["point_features"])
      parameter_predictor.load_state_dict(parameters["parameter_predictor"])

      #set networks to evaluation mode
      point_features.eval()
      parameter_predictor.eval()