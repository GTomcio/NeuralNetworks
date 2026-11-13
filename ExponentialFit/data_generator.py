from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import torch

if TYPE_CHECKING:
    # Import this type only for editor and type-checker support.
    from matplotlib.figure import Figure


#########################################################################
# Dataset location and data format
#########################################################################

# Keep generated files in this project's data folder.
DATA_DIR = Path(__file__).resolve().parent / "data"


# Name and type the tensors returned by generate_batch.
class CurveBatch(TypedDict):
    # Store padded x coordinates for every curve.
    x_points: torch.Tensor
    # Store the noisy y values given to the model.
    y_points: torch.Tensor
    # Store the clean y values from the true equation.
    y_true: torch.Tensor
    # Mark which entries are real points and which are padding.
    valid_mask: torch.Tensor
    # Store the true a, b, c parameters for each curve.
    curve_params: torch.Tensor
    # Mark which real points were replaced by large outliers.
    outlier_mask: torch.Tensor


#########################################################################
# Generate batches of exponential curves
#########################################################################

# Create a batch of randomly generated curves and their observations.
def generate_batch(
    batch_size: int,
    *,
    min_points: int = 16,
    max_points: int = 64,
    x_range: tuple[float, float] = (0.0, 2.0),
    a_range: tuple[float, float] = (0.1, 0.5),
    b_range: tuple[float, float] = (-2.0, 2.0),
    min_abs_b: float = 0.1,
    c_range: tuple[float, float] = (-2.0, 2.0),
    noise_std_fraction: float = 0.1,
    outlier_probability: float = 0.03,
    outlier_std_fraction: float = 1.0,
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> CurveBatch:
    # Generate curves, add noise, shuffle points, then pad each row.
    # Check that the requested ranges and probabilities are valid.
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if min_points < 2 or max_points < min_points:
        raise ValueError("require 2 <= min_points <= max_points")
    if not x_range[0] < x_range[1]:
        raise ValueError("x_range must be increasing")
    if not a_range[0] < a_range[1] or a_range[0] <= 0:
        raise ValueError("a_range must be increasing and strictly positive")
    if not b_range[0] < 0 < b_range[1]:
        raise ValueError("b_range must include both negative and positive values")
    if min_abs_b <= 0 or min_abs_b >= min(abs(b_range[0]), abs(b_range[1])):
        raise ValueError("min_abs_b must be positive and smaller than both |b| bounds")
    if not c_range[0] < c_range[1]:
        raise ValueError("c_range must be increasing")
    if not 0 <= outlier_probability <= 1:
        raise ValueError("outlier_probability must be between 0 and 1")
    if noise_std_fraction < 0:
        raise ValueError("noise_std_fraction must be nonnegative")
    if outlier_std_fraction < 0:
        raise ValueError("outlier_std_fraction must be nonnegative")

    # Create a random number generator and choose each curve's point count.
    rng = np.random.default_rng(seed)
    point_counts = rng.integers(min_points, max_points + 1, size=batch_size)
    padded_length = int(point_counts.max())

    # Allocate padded arrays; padding stays zero and is identified by valid_mask.
    x_array = np.zeros((batch_size, padded_length), dtype=np.float32)
    y_points = np.zeros_like(x_array)
    y_true = np.zeros_like(x_array)
    valid_mask = np.zeros((batch_size, padded_length), dtype=np.bool_)
    outlier_mask = np.zeros_like(valid_mask)
    curve_params = np.empty((batch_size, 3), dtype=np.float32)

    # Generate one curve and its observations for each batch row.
    for row, count in enumerate(point_counts):
        # Choose the curve parameters a, b, and c.
        a = rng.uniform(*a_range)
        if rng.random() < 0.5:
            # Sample a positive b value for an increasing exponential.
            b = rng.uniform(min_abs_b, b_range[1])
        else:
            # Sample a negative b value for a decreasing exponential.
            b = -rng.uniform(min_abs_b, abs(b_range[0]))
        c = rng.uniform(*c_range)

        # Sample x locations and calculate the clean exponential curve.
        x_points = rng.uniform(*x_range, size=int(count))
        y_true_values = a * np.exp(b * x_points) + c

        # Add small Gaussian noise to every clean y value.
        small_noise = rng.normal(
            loc=0.0,
            scale=np.abs(y_true_values) * noise_std_fraction,
            size=int(count),
        )
        y_point_values = y_true_values + small_noise
        # Replace a random subset of points with much larger outliers.
        is_outlier = rng.random(int(count)) < outlier_probability
        outlier_directions = rng.choice((-1.0, 1.0), size=int(count))
        outlier_values = rng.normal(
            loc=5.0 * y_true_values[is_outlier] * outlier_directions[is_outlier],
            scale=np.abs(y_true_values[is_outlier]) * outlier_std_fraction,
            size=int(is_outlier.sum()),
        )
        y_point_values[is_outlier] = outlier_values

        # Shuffle points so the model cannot rely on their order.
        permutation = rng.permutation(int(count))
        # Copy this curve's values into its row and mark real entries.
        x_array[row, :count] = x_points[permutation]
        y_points[row, :count] = y_point_values[permutation]
        y_true[row, :count] = y_true_values[permutation]
        valid_mask[row, :count] = True
        outlier_mask[row, :count] = is_outlier[permutation]
        curve_params[row] = (a, b, c)

    # Convert the completed NumPy arrays to PyTorch tensors on the chosen device.
    target_device = torch.device(device)
    return {
        "x_points": torch.as_tensor(x_array, device=target_device),
        "y_points": torch.as_tensor(y_points, device=target_device),
        "y_true": torch.as_tensor(y_true, device=target_device),
        "valid_mask": torch.as_tensor(valid_mask, device=target_device),
        "curve_params": torch.as_tensor(curve_params, device=target_device),
        "outlier_mask": torch.as_tensor(outlier_mask, device=target_device),
    }


#########################################################################
# Save generated datasets for main.py
#########################################################################

# Write one generated batch to a compressed dataset file.
def export_dataset(
    batch: CurveBatch,
    *,
    split: Literal["train", "test"],
    filename: str = "curves.npz",
    data_dir: str | Path = DATA_DIR,
) -> Path:
    # Save arrays using the names expected by main.py.
    # Check the split and filename before creating the output path.
    if split not in ("train", "test"):
        raise ValueError("split must be either 'train' or 'test'")
    if Path(filename).name != filename or not filename.endswith(".npz"):
        raise ValueError("filename must be a simple .npz filename")

    # Create the split folder if it does not exist yet.
    destination = Path(data_dir) / split / filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Move tensors to CPU NumPy arrays so NumPy can save them.
    arrays = {
        "x_points": batch["x_points"].detach().cpu().numpy(),
        "y_points": batch["y_points"].detach().cpu().numpy(),
        "y_true": batch["y_true"].detach().cpu().numpy(),
        "valid_mask": batch["valid_mask"].detach().cpu().numpy(),
        "curve_params": batch["curve_params"].detach().cpu().numpy(),
        "outlier_mask": batch["outlier_mask"].detach().cpu().numpy(),
    }
    # Save curve arrays and small metadata in one compressed file.
    np.savez_compressed(
        destination,
        x_points=arrays["x_points"],
        y_points=arrays["y_points"],
        y_true=arrays["y_true"],
        valid_mask=arrays["valid_mask"],
        curve_params=arrays["curve_params"],
        outlier_mask=arrays["outlier_mask"],
        param_names=np.asarray(["a", "b", "c"]),
        equation=np.asarray("y = a * exp(b * x) + c"),
        format_version=np.asarray(2, dtype=np.int32),
    )
    # Return the full path so the caller can display or reuse it.
    return destination


#########################################################################
# Optional plotting for inspecting generated data
#########################################################################

# Draw selected curves and optionally save or display the figure.
def plot_batch(
    batch: CurveBatch,
    *,
    num_curves: int = 4,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple["Figure", np.ndarray]:
    # Plot noisy points and the true curves for a quick visual check.
    # Import plotting here so it is needed only when this function is called.
    import matplotlib.pyplot as plt

    # Check the requested number of curves before plotting.
    batch_size = int(batch["x_points"].shape[0])
    if num_curves < 1 or num_curves > batch_size:
        raise ValueError("num_curves must be between 1 and batch size")

    # Convert tensors to NumPy arrays for Matplotlib.
    x_array = batch["x_points"].detach().cpu().numpy()
    y_points = batch["y_points"].detach().cpu().numpy()
    y_true = batch["y_true"].detach().cpu().numpy()
    valid_mask = batch["valid_mask"].detach().cpu().numpy()
    outlier_mask = batch["outlier_mask"].detach().cpu().numpy()
    curve_params = batch["curve_params"].detach().cpu().numpy()

    # Create one plot row for each requested curve.
    figure, axes = plt.subplots(
        num_curves,
        1,
        figsize=(8, 3.2 * num_curves),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[:, 0]
    # Plot each curve using only its non-padding entries.
    for index, axis in enumerate(axes):
        valid = valid_mask[index]
        x_for_curve = x_array[index, valid]
        y_points_for_curve = y_points[index, valid]
        y_true_for_curve = y_true[index, valid]
        outliers = outlier_mask[index, valid]

        # Sort by x for a connected true-curve line.
        order = np.argsort(x_for_curve)
        a, b, c = curve_params[index]
        # Draw the clean curve and all observed noisy points.
        axis.plot(
            x_for_curve[order],
            y_true_for_curve[order],
            color="tab:blue",
            label="True curve",
        )
        axis.scatter(
            x_for_curve,
            y_points_for_curve,
            color="tab:orange",
            s=24,
            label="Y points",
            zorder=3,
        )
        # Highlight outliers separately when this curve has any.
        if np.any(outliers):
            axis.scatter(
                x_for_curve[outliers],
                y_points_for_curve[outliers],
                color="tab:red",
                marker="x",
                s=48,
                label="Outliers",
                zorder=4,
            )
        # Add labels, grid, and legend to make the plot readable.
        axis.set_title(f"Krzywa {index + 1}: a={a:.3f}, b={b:.3f}, c={c:.3f}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(True, alpha=0.25)
        axis.legend()

    # Save the figure only when a path was provided.
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=160, bbox_inches="tight")
    # Display the figure only when requested.
    if show:
        plt.show()
    # Return the figure objects so the caller can customize or close them.
    return figure, axes


#########################################################################
# Generate and save the datasets when this file is run directly
#########################################################################
if __name__ == "__main__":
    # Generate training and test examples with repeatable random seeds.
    # Generate a larger training split and a smaller test split.
    train_batch = generate_batch(batch_size=10_000, seed=1)
    test_batch = generate_batch(batch_size=2_000, seed=2)

    # Save each batch using the split names that main.py expects.
    train_path = export_dataset(train_batch, split="train")
    test_path = export_dataset(test_batch, split="test")

    # Show the locations of the saved files.
    print(f"Saved training data to: {train_path}")
    print(f"Saved test data to: {test_path}")
