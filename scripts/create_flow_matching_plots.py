"""Create illustrative plots for the flow matching blog post.

The figures are intentionally toy 2D examples. They are meant to clarify the
geometry of the construction rather than simulate a trained model.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "images" / "flow-matching"
MPLCONFIGDIR = ROOT / ".matplotlib-cache"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np


RNG = np.random.default_rng(7)

SOURCE_COLOR = "#4C78A8"
TARGET_COLOR = "#F58518"
MID_COLOR = "#54A24B"
PATH_COLOR = "#777777"
ARROW_COLOR = "#B279A2"


def sample_source(n: int) -> np.ndarray:
    """Simple source distribution: standard Gaussian in R^2."""
    return RNG.normal(size=(n, 2))


def sample_target(n: int) -> np.ndarray:
    """A small three-component mixture, used as a toy data distribution."""
    centers = np.array([[-2.0, -0.9], [1.8, -0.7], [0.0, 1.9]])
    scales = np.array([[0.34, 0.22], [0.30, 0.25], [0.26, 0.32]])
    weights = np.array([0.35, 0.35, 0.30])
    components = RNG.choice(len(centers), size=n, p=weights)
    return centers[components] + RNG.normal(size=(n, 2)) * scales[components]


def interp(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * x0 + t * x1


def style_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def random_paths_induce_measures() -> None:
    """Figure 1: random endpoints induce paths and intermediate laws."""
    n_paths = 34
    n_cloud = 800
    t_values = [0.0, 0.33, 0.66, 1.0]

    x0_paths = sample_source(n_paths)
    x1_paths = sample_target(n_paths)
    x0_cloud = sample_source(n_cloud)
    x1_cloud = sample_target(n_cloud)

    fig, axes = plt.subplots(1, len(t_values), figsize=(11.0, 3.0), sharex=True, sharey=True)
    for ax, t in zip(axes, t_values):
        cloud = interp(x0_cloud, x1_cloud, t)
        path_points = interp(x0_paths, x1_paths, t)

        for x0, x1 in zip(x0_paths, x1_paths):
            ax.plot([x0[0], x1[0]], [x0[1], x1[1]], color=PATH_COLOR, lw=0.7, alpha=0.22)

        ax.scatter(cloud[:, 0], cloud[:, 1], s=7, color=MID_COLOR, alpha=0.20, linewidths=0)
        ax.scatter(path_points[:, 0], path_points[:, 1], s=18, color=MID_COLOR, alpha=0.82, linewidths=0)
        if t == 0.0:
            ax.scatter(x0_paths[:, 0], x0_paths[:, 1], s=20, color=SOURCE_COLOR, label=r"$X_0$")
        if t == 1.0:
            ax.scatter(x1_paths[:, 0], x1_paths[:, 1], s=20, color=TARGET_COLOR, label=r"$X_1$")

        ax.set_title(rf"$t={t:.2f}$", fontsize=12)
        style_axis(ax)

    axes[0].set_xlim(-3.2, 3.2)
    axes[0].set_ylim(-2.3, 2.8)
    fig.suptitle("Random endpoints turn deterministic curves into a path of laws", y=1.03, fontsize=13)
    save(fig, "01-random-paths-induce-measures.png")


def conditional_average_velocity() -> None:
    """Figure 2: the local flow velocity is a conditional average."""
    n_paths = 2200
    t = 0.50
    x0 = sample_source(n_paths)
    x1 = sample_target(n_paths)
    xt = interp(x0, x1, t)
    velocities = x1 - x0

    probe = np.array([0.05, 0.05])
    radius = 0.34
    distances = np.linalg.norm(xt - probe, axis=1)
    local = np.argsort(distances)[:60]
    average_velocity = velocities[local].mean(axis=0)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.scatter(xt[:, 0], xt[:, 1], s=6, color="#BBBBBB", alpha=0.18, linewidths=0)
    circle = plt.Circle(probe, radius, edgecolor="#333333", facecolor="none", lw=1.4, ls="--")
    ax.add_patch(circle)

    starts = xt[local]
    vecs = velocities[local]
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8) * 0.35
    ax.quiver(
        starts[:, 0],
        starts[:, 1],
        vecs[:, 0],
        vecs[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.004,
        color=PATH_COLOR,
        alpha=0.45,
    )

    avg = average_velocity / np.linalg.norm(average_velocity) * 0.82
    ax.quiver(
        [probe[0]],
        [probe[1]],
        [avg[0]],
        [avg[1]],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.012,
        color=ARROW_COLOR,
    )
    ax.scatter([probe[0]], [probe[1]], s=52, color="#333333", zorder=5)
    ax.text(probe[0] + 0.12, probe[1] - 0.18, r"$x$", fontsize=13)
    ax.text(probe[0] + avg[0] + 0.10, probe[1] + avg[1], r"$v_t(x)$", color=ARROW_COLOR, fontsize=13)

    ax.set_title("Conditional average velocity", fontsize=13)
    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-2.1, 2.4)
    style_axis(ax)
    save(fig, "02-conditional-average-velocity.png")


def linear_path_snapshots() -> None:
    """Figure 3: snapshots along a linear interpolation path."""
    n = 900
    x0 = sample_source(n)
    x1 = sample_target(n)
    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(1, len(t_values), figsize=(12.8, 2.7), sharex=True, sharey=True)
    for ax, t in zip(axes, t_values):
        xt = interp(x0, x1, t)
        color = SOURCE_COLOR if t == 0.0 else TARGET_COLOR if t == 1.0 else MID_COLOR
        ax.scatter(xt[:, 0], xt[:, 1], s=7, color=color, alpha=0.36, linewidths=0)
        ax.set_title(rf"$t={t:.2f}$", fontsize=12)
        style_axis(ax)

    axes[0].set_xlim(-3.5, 3.3)
    axes[0].set_ylim(-2.5, 2.9)
    fig.suptitle("Linear interpolation path", y=1.04, fontsize=13)
    save(fig, "03-linear-path-snapshots.png")


def gaussian_conditional_path() -> None:
    """Figure 4: conditional Gaussian path around data samples."""
    n_data = 7
    n_per_data = 180
    y = sample_target(n_data)
    z = RNG.normal(size=(n_data, n_per_data, 2))
    t_values = [0.0, 0.33, 0.66, 1.0]

    fig, axes = plt.subplots(1, len(t_values), figsize=(11.0, 3.0), sharex=True, sharey=True)
    for ax, t in zip(axes, t_values):
        a_t = t
        sigma_t = 1.0 - t
        means = a_t * y
        xt = a_t * y[:, None, :] + sigma_t * z
        flat = xt.reshape(-1, 2)

        ax.scatter(flat[:, 0], flat[:, 1], s=6, color=MID_COLOR, alpha=0.16, linewidths=0)
        ax.scatter(means[:, 0], means[:, 1], s=32, color=TARGET_COLOR, alpha=0.95, linewidths=0)

        for center in means:
            circle = plt.Circle(center, sigma_t, edgecolor=TARGET_COLOR, facecolor="none", lw=1.0, alpha=0.32)
            ax.add_patch(circle)

        ax.set_title(rf"$a_t={a_t:.2f},\ \sigma_t={sigma_t:.2f}$", fontsize=12)
        style_axis(ax)

    axes[0].set_xlim(-3.5, 3.3)
    axes[0].set_ylim(-2.5, 3.0)
    fig.suptitle(r"Gaussian conditional path: $X_t=a_tY+\sigma_tZ$", y=1.04, fontsize=13)
    save(fig, "04-gaussian-conditional-path.png")


def estimate_linear_velocity(
    query: np.ndarray,
    t: float,
    x0_ref: np.ndarray,
    x1_ref: np.ndarray,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Kernel estimate of E[X_1 - X_0 | X_t = x] for the linear path."""
    xt_ref = interp(x0_ref, x1_ref, t)
    velocities = x1_ref - x0_ref
    result = np.empty_like(query)
    density = np.empty(query.shape[0])
    chunk_size = 256

    for start in range(0, query.shape[0], chunk_size):
        stop = start + chunk_size
        q = query[start:stop]
        sq_dist = ((q[:, None, :] - xt_ref[None, :, :]) ** 2).sum(axis=2)
        weights = np.exp(-0.5 * sq_dist / bandwidth**2)
        totals = weights.sum(axis=1)
        result[start:stop] = weights @ velocities / np.maximum(totals[:, None], 1e-12)
        density[start:stop] = totals

    return result, density


def estimated_velocity_field() -> None:
    """Figure 5: kernel estimate of the linear-path velocity field."""
    n_ref = 9000
    x0_ref = sample_source(n_ref)
    x1_ref = sample_target(n_ref)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    t_field = 0.50
    target_cloud = sample_target(650)
    source_cloud = sample_source(650)
    xt_cloud = interp(x0_ref[:1300], x1_ref[:1300], t_field)
    ax.scatter(source_cloud[:, 0], source_cloud[:, 1], s=7, color=SOURCE_COLOR, alpha=0.07, linewidths=0)
    ax.scatter(target_cloud[:, 0], target_cloud[:, 1], s=9, color=TARGET_COLOR, alpha=0.11, linewidths=0)
    ax.scatter(xt_cloud[:, 0], xt_cloud[:, 1], s=7, color=MID_COLOR, alpha=0.13, linewidths=0)

    grid_x = np.linspace(-3.0, 3.0, 24)
    grid_y = np.linspace(-2.0, 2.5, 19)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    field, density = estimate_linear_velocity(grid, t_field, x0_ref, x1_ref, bandwidth=0.40)
    keep = density > np.quantile(density, 0.34)
    field_norm = np.linalg.norm(field, axis=1)
    arrow = field / np.maximum(field_norm[:, None], 1e-8) * np.minimum(0.42, 0.16 + 0.08 * field_norm)[:, None]

    ax.quiver(
        grid[keep, 0],
        grid[keep, 1],
        arrow[keep, 0],
        arrow[keep, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0036,
        color=ARROW_COLOR,
        alpha=0.70,
    )

    ax.text(-2.92, -1.90, r"$\mu_0$", color=SOURCE_COLOR, fontsize=13)
    ax.text(-0.10, 0.08, r"$\mu_t$", color=MID_COLOR, fontsize=13)
    ax.text(1.84, -1.70, r"$\mu_1$", color=TARGET_COLOR, fontsize=13)
    ax.set_title(f"Estimated velocity field at t={t_field:.2f}", fontsize=13)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.2, 2.7)
    style_axis(ax)
    save(fig, "05-estimated-velocity-field.png")


def main() -> None:
    random_paths_induce_measures()
    conditional_average_velocity()
    linear_path_snapshots()
    gaussian_conditional_path()
    estimated_velocity_field()
    print(f"Wrote plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
