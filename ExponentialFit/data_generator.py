"""Generate synthetic point sets for exponential-curve fitting.

Each example follows ``y = a * exp(b * x) + c``.  The returned batch is
ready for a set-based PyTorch model: examples have a common padded length and
``valid_mask`` identifies which entries are real observations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# -----------------------------------------------------------------------------
# Data format and defaults
# -----------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"


class CurveBatch(TypedDict):
    """A batch of padded curves with noisy y values and true y targets."""

    x_points: torch.Tensor
    y_points: torch.Tensor
    y_true: torch.Tensor
    valid_mask: torch.Tensor
    curve_params: torch.Tensor
    outlier_mask: torch.Tensor


# -----------------------------------------------------------------------------
# Curve generation
# -----------------------------------------------------------------------------

def generate_batch(
    batch_size: int,
    *,
    min_points: int = 4,
    max_points: int = 64,
    x_range: tuple[float, float] = (0.0, 2.0),
    a_range: tuple[float, float] = (0.5, 2.0),
    b_range: tuple[float, float] = (-1.5, 1.5),
    min_abs_b: float = 0.1,
    c_range: tuple[float, float] = (-0.5, 0.5),
    noise_std_fraction: float = 0.01,
    outlier_probability: float = 0.03,
    outlier_std_fraction: float = 1.0,
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> CurveBatch:
    """Create random exponential curves with small noise and sparse outliers.

    Point counts are sampled independently from ``[min_points, max_points]``.
    Each curve is padded to the largest count in the batch; padded entries are
    zero and must be ignored using ``valid_mask``. Points are shuffled so training
    does not depend on their order.

    The output ``curve_params`` has columns ``a, b, c``. ``y_true`` is the
    true curve, and ``y_points`` is the noisy value supplied to the model.
    Every real point receives small Gaussian noise with standard deviation
    ``abs(y_true) * noise_std_fraction``. A random subset also receives an
    outlier drawn around either +5 or -5 times its true y value, with standard
    deviation ``abs(y_true) * outlier_std_fraction``. ``outlier_mask`` marks
    only those outliers, not the small noise. Padding values are zero; use
    ``valid_mask`` to ignore them. Numeric tensors are float32 and masks are
    boolean.
    """
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
    rng = np.random.default_rng(seed)
    point_counts = rng.integers(min_points, max_points + 1, size=batch_size)
    padded_length = int(point_counts.max())

    x_array = np.zeros((batch_size, padded_length), dtype=np.float32)
    y_points = np.zeros_like(x_array)
    y_true = np.zeros_like(x_array)
    valid_mask = np.zeros((batch_size, padded_length), dtype=np.bool_)
    outlier_mask = np.zeros_like(valid_mask)
    curve_params = np.empty((batch_size, 3), dtype=np.float32)

    for row, count in enumerate(point_counts):
        a = rng.uniform(*a_range)
        if rng.random() < 0.5:
            b = rng.uniform(min_abs_b, b_range[1])
        else:
            b = -rng.uniform(min_abs_b, abs(b_range[0]))
        c = rng.uniform(*c_range)

        x_points = rng.uniform(*x_range, size=int(count))
        y_true_values = a * np.exp(b * x_points) + c

        small_noise = rng.normal(
            loc=0.0,
            scale=np.abs(y_true_values) * noise_std_fraction,
            size=int(count),
        )
        y_point_values = y_true_values + small_noise
        is_outlier = rng.random(int(count)) < outlier_probability
        outlier_directions = rng.choice((-1.0, 1.0), size=int(count))
        outlier_values = rng.normal(
            loc=5.0 * y_true_values[is_outlier] * outlier_directions[is_outlier],
            scale=np.abs(y_true_values[is_outlier]) * outlier_std_fraction,
            size=int(is_outlier.sum()),
        )
        y_point_values[is_outlier] = outlier_values

        # A random order makes the set nature of the input explicit.
        permutation = rng.permutation(int(count))
        x_array[row, :count] = x_points[permutation]
        y_points[row, :count] = y_point_values[permutation]
        y_true[row, :count] = y_true_values[permutation]
        valid_mask[row, :count] = True
        outlier_mask[row, :count] = is_outlier[permutation]
        curve_params[row] = (a, b, c)

    target_device = torch.device(device)
    return {
        "x_points": torch.as_tensor(x_array, device=target_device),
        "y_points": torch.as_tensor(y_points, device=target_device),
        "y_true": torch.as_tensor(y_true, device=target_device),
        "valid_mask": torch.as_tensor(valid_mask, device=target_device),
        "curve_params": torch.as_tensor(curve_params, device=target_device),
        "outlier_mask": torch.as_tensor(outlier_mask, device=target_device),
    }


# -----------------------------------------------------------------------------
# Saving and plotting
# -----------------------------------------------------------------------------

def export_dataset(
    batch: CurveBatch,
    *,
    split: Literal["train", "test"],
    filename: str = "curves.npz",
    data_dir: str | Path = DATA_DIR,
) -> Path:
    """Save a generated batch as a compressed NPZ dataset."""
    if split not in ("train", "test"):
        raise ValueError("split must be either 'train' or 'test'")
    if Path(filename).name != filename or not filename.endswith(".npz"):
        raise ValueError("filename must be a simple .npz filename")

    destination = Path(data_dir) / split / filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "x_points": batch["x_points"].detach().cpu().numpy(),
        "y_points": batch["y_points"].detach().cpu().numpy(),
        "y_true": batch["y_true"].detach().cpu().numpy(),
        "valid_mask": batch["valid_mask"].detach().cpu().numpy(),
        "curve_params": batch["curve_params"].detach().cpu().numpy(),
        "outlier_mask": batch["outlier_mask"].detach().cpu().numpy(),
    }
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
    return destination


# -----------------------------------------------------------------------------
# Optional visualization
# -----------------------------------------------------------------------------

def plot_batch(
    batch: CurveBatch,
    *,
    num_curves: int = 4,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple["Figure", np.ndarray]:
    """Plot y points, outliers, and the underlying true curves.

    Returns ``(figure, axes)`` so callers can further customize or close the
    figure. Set ``show=False`` for scripts and notebooks that only save it.
    """
    import matplotlib.pyplot as plt
    batch_size = int(batch["x_points"].shape[0])
    if num_curves < 1 or num_curves > batch_size:
        raise ValueError("num_curves must be between 1 and batch size")

    x_array = batch["x_points"].detach().cpu().numpy()
    y_points = batch["y_points"].detach().cpu().numpy()
    y_true = batch["y_true"].detach().cpu().numpy()
    valid_mask = batch["valid_mask"].detach().cpu().numpy()
    outlier_mask = batch["outlier_mask"].detach().cpu().numpy()
    curve_params = batch["curve_params"].detach().cpu().numpy()

    figure, axes = plt.subplots(
        num_curves,
        1,
        figsize=(8, 3.2 * num_curves),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[:, 0]
    for index, axis in enumerate(axes):
        valid = valid_mask[index]
        x_for_curve = x_array[index, valid]
        y_points_for_curve = y_points[index, valid]
        y_true_for_curve = y_true[index, valid]
        outliers = outlier_mask[index, valid]

        order = np.argsort(x_for_curve)
        a, b, c = curve_params[index]
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
        axis.set_title(f"Krzywa {index + 1}: a={a:.3f}, b={b:.3f}, c={c:.3f}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(True, alpha=0.25)
        axis.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Generate the saved training and test datasets when this file is run.
    # -------------------------------------------------------------------------
    train_batch = generate_batch(batch_size=10_000, seed=1)
    test_batch = generate_batch(batch_size=2_000, seed=2)

    train_path = export_dataset(train_batch, split="train")
    test_path = export_dataset(test_batch, split="test")

    print(f"Saved training data to: {train_path}")
    print(f"Saved test data to: {test_path}")
