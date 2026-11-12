"""Generate synthetic point sets for exponential-curve fitting.

Each example follows ``y = a * exp(b * x) + c``.  The returned batch is
ready for a set-based PyTorch model: examples have a common padded length and
``mask`` identifies which entries are real observations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch


DATA_DIR = Path(__file__).resolve().parent / "data"


class CurveBatch(TypedDict):
    """A batch of padded curves and their known clean targets."""

    x: torch.Tensor
    y: torch.Tensor
    y_clean: torch.Tensor
    mask: torch.Tensor
    params: torch.Tensor
    n_points: torch.Tensor
    outlier_mask: torch.Tensor


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
    relative_noise_range: tuple[float, float] = (0.01, 0.08),
    outlier_probability: float = 0.03,
    outlier_scale_range: tuple[float, float] = (3.0, 8.0),
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> CurveBatch:
    """Create random exponential curves with noise and sparse outliers.

    Point counts are sampled independently from ``[min_points, max_points]``.
    Each curve is padded to the largest count in the batch; padded entries are
    zero and must be ignored using ``mask``.  Points are shuffled so training
    does not depend on their order.

    The output ``params`` has columns ``a, b, c``.  ``y_clean`` is the noiseless
    curve, ``y`` is the observed value, and ``outlier_mask`` marks injected
    outliers. All returned tensors are float32 except the boolean masks and
    the integer ``n_points``.
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
    if not (0 <= relative_noise_range[0] <= relative_noise_range[1]):
        raise ValueError("relative_noise_range must be nonnegative and increasing")
    if not (0 <= outlier_probability <= 1):
        raise ValueError("outlier_probability must be between 0 and 1")
    if not (0 < outlier_scale_range[0] <= outlier_scale_range[1]):
        raise ValueError("outlier_scale_range must be positive and increasing")

    rng = np.random.default_rng(seed)
    counts = rng.integers(min_points, max_points + 1, size=batch_size)
    padded_length = int(counts.max())

    x = np.zeros((batch_size, padded_length), dtype=np.float32)
    y = np.zeros_like(x)
    y_clean = np.zeros_like(x)
    mask = np.zeros((batch_size, padded_length), dtype=np.bool_)
    outlier_mask = np.zeros_like(mask)
    params = np.empty((batch_size, 3), dtype=np.float32)

    for row, count in enumerate(counts):
        a = rng.uniform(*a_range)
        if rng.random() < 0.5:
            b = rng.uniform(min_abs_b, b_range[1])
        else:
            b = -rng.uniform(min_abs_b, abs(b_range[0]))
        c = rng.uniform(*c_range)

        x_points = rng.uniform(*x_range, size=int(count))
        clean_points = a * np.exp(b * x_points) + c

        # Scale noise to the curve's variation while keeping nearly-flat curves
        # from receiving numerically negligible noise.
        signal_scale = max(
            float(np.std(clean_points)),
            float(np.mean(np.abs(clean_points))) * 0.05,
            1e-3,
        )
        relative_noise = rng.uniform(*relative_noise_range)
        observed_points = clean_points + rng.normal(
            0.0, signal_scale * relative_noise, size=int(count)
        )

        is_outlier = rng.random(int(count)) < outlier_probability
        n_outliers = int(is_outlier.sum())
        if n_outliers:
            magnitudes = rng.uniform(*outlier_scale_range, size=n_outliers)
            directions = rng.choice(np.array([-1.0, 1.0]), size=n_outliers)
            observed_points[is_outlier] += (
                directions * magnitudes * signal_scale * relative_noise
            )

        # A random order makes the set nature of the input explicit.
        permutation = rng.permutation(int(count))
        x[row, :count] = x_points[permutation]
        y[row, :count] = observed_points[permutation]
        y_clean[row, :count] = clean_points[permutation]
        mask[row, :count] = True
        outlier_mask[row, :count] = is_outlier[permutation]
        params[row] = (a, b, c)

    target_device = torch.device(device)
    return {
        "x": torch.as_tensor(x, device=target_device),
        "y": torch.as_tensor(y, device=target_device),
        "y_clean": torch.as_tensor(y_clean, device=target_device),
        "mask": torch.as_tensor(mask, device=target_device),
        "params": torch.as_tensor(params, device=target_device),
        "n_points": torch.as_tensor(counts, dtype=torch.int64, device=target_device),
        "outlier_mask": torch.as_tensor(outlier_mask, device=target_device),
    }


def export_dataset(
    batch: CurveBatch,
    *,
    split: Literal["train", "test"],
    filename: str = "curves.npz",
    data_dir: str | Path = DATA_DIR,
) -> Path:
    """Save a generated batch as one compressed, model-ready NPZ file.

    Files are written to ``data_dir/split/filename``. The archive contains the
    same keys as :func:`generate_batch`, plus small metadata fields. NPZ keeps
    the padded tensors, masks and parameter labels together without pickle.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be either 'train' or 'test'")
    if Path(filename).name != filename or not filename.endswith(".npz"):
        raise ValueError("filename must be a simple .npz filename")

    destination = Path(data_dir) / split / filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    tensor_batch = cast(dict[str, torch.Tensor], batch)
    arrays = {
        key: value.detach().cpu().numpy()
        for key, value in tensor_batch.items()
    }
    np.savez_compressed(
        destination,
        x=arrays["x"],
        y=arrays["y"],
        y_clean=arrays["y_clean"],
        mask=arrays["mask"],
        params=arrays["params"],
        n_points=arrays["n_points"],
        outlier_mask=arrays["outlier_mask"],
        param_names=np.asarray(["a", "b", "c"]),
        equation=np.asarray("y = a * exp(b * x) + c"),
        format_version=np.asarray(1, dtype=np.int32),
    )
    return destination


def plot_batch(
    batch: CurveBatch,
    *,
    num_curves: int = 4,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot noisy points, injected outliers and the underlying clean curves.

    Returns ``(figure, axes)`` so callers can further customize or close the
    figure. Set ``show=False`` for scripts and notebooks that only save it.
    """
    batch_size = int(batch["x"].shape[0])
    if num_curves < 1 or num_curves > batch_size:
        raise ValueError("num_curves must be between 1 and batch size")

    x = batch["x"].detach().cpu().numpy()
    y = batch["y"].detach().cpu().numpy()
    y_clean = batch["y_clean"].detach().cpu().numpy()
    mask = batch["mask"].detach().cpu().numpy()
    outlier_mask = batch["outlier_mask"].detach().cpu().numpy()
    params = batch["params"].detach().cpu().numpy()

    figure, axes = plt.subplots(
        num_curves,
        1,
        figsize=(8, 3.2 * num_curves),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[:, 0]
    for index, axis in enumerate(axes):
        valid = mask[index]
        x_points = x[index, valid]
        y_points = y[index, valid]
        clean_points = y_clean[index, valid]
        outliers = outlier_mask[index, valid]

        order = np.argsort(x_points)
        a, b, c = params[index]
        axis.plot(
            x_points[order],
            clean_points[order],
            color="tab:blue",
            label="Czysta krzywa",
        )
        axis.scatter(
            x_points[~outliers],
            y_points[~outliers],
            color="tab:orange",
            s=24,
            label="Pomiary",
            zorder=3,
        )
        if outliers.any():
            axis.scatter(
                x_points[outliers],
                y_points[outliers],
                color="tab:red",
                marker="x",
                s=42,
                label="Odstające pomiary",
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
