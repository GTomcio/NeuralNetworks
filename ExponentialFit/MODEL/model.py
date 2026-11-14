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
      parameters_path = data_dir / "parameters.npz"

      #verify path
      if not data_dir.exists:
            raise ValueError(f"couldnt find {data_dir}")
      if not curves_path.exists:
            raise ValueError(f"couldnt find {curves_path}")
      if not parameters_path.exists:
            raise ValueError(f"couldnt find {parameters_path}")

      #verify data structure
      with np.load(curves_path, allow_pickle = False) as data:
            required_keys = (
                  "x_points",
                  "y_points",
                  "valid_mask",
            )

      with np.load(parameters_path, allow_pickle=True) as data:
            required_keys = (

            )
      
      #check for missing data
      missing_keys = [key for key in required_keys if key not in data.files]
      if missing_keys:
            raise ValueError(
                  f"{required_keys} is missing {missing_keys}."
            )

      #create tensors before closing .npz
      tensors = [torch.from_numpy(data[key].copy()) for key in required_keys]

      #return tensors
      return DataLoader(
            TensorDataset(*tensors),
            num_workers=0,
            pin_memory=(DEVICE.type == "cuda"),
      )

