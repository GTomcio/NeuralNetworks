from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Select a GPU when one is available; otherwise use the CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

#load dataset
def load_dataset(str):
      if str not in ("data"):
            raise ValueError("wrong filename")

      path = Path(__file__)

