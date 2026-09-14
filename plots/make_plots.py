"""
Generate the project's plots for the README/demo:
  1. Springback (P99) vs radius, physics data, with the 2.0mm limit and
     the feasible/infeasible region marked.
  2. Physics vs. surrogate overlay across the radius range, to visually
     confirm the fit.
  3. Timing comparison bar chart: physics campaign sweep vs. surrogate
     sweep, using the real measured numbers (not invented).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from physics import bend_model as bm
from surrogate.train_surrogate import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, load_training_data

OUTPUT_DIR = Path("plots/output")
SPRINGBACK_LIMIT = 2.0


def plot_springback_vs_radius(r: np.ndarray, p99: np.ndarray, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(r, p99, color="black", linewidth=1.5, label="P99 springback (physics)")
    ax.axhline(SPRINGBACK_LIMIT, color="red", linestyle="--", linewidth=1,
               label=f"Springback limit ({SPRINGBACK_LIMIT}mm)")

    r_floor = bm.crack_floor()
    ax.axvline(r_floor, color="gray", linestyle=":", linewidth=1,
               label=f"Crack floor ({r_floor:.2f}mm)")

    feasible_mask = p99 <= SPRINGBACK_LIMIT
    ax.fill_between(r, 0, p99.max() * 1.1, where=feasible_mask,
                     color="green", alpha=0.08, label="Feasible region")

    ax.set_xlabel("Bend radius r (mm)")
    ax.set_ylabel("P99 springback (mm)")
    ax.set_title("Springback vs. bend radius, with feasibility bounds")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_physics_vs_surrogate(r: np.ndarray, p99: np.ndarray, model, output_path: Path):
    r_dense = np.linspace(r.min(), r.max(), 300).reshape(-1, 1)
    p99_pred = model.predict(r_dense)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(r, p99, s=15, color="black", alpha=0.6, label="Physics (training data)")
    ax.plot(r_dense, p99_pred, color="tab:blue", linewidth=2, label="Surrogate prediction")
    ax.axhline(SPRINGBACK_LIMIT, color="red", linestyle="--", linewidth=1,
               label=f"Springback limit ({SPRINGBACK_LIMIT}mm)")

    ax.set_xlabel("Bend radius r (mm)")
    ax.set_ylabel("P99 springback (mm)")
    ax.set_title("Physics vs. surrogate prediction")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_timing_comparison(
    physics_sweep_s: float,
    surrogate_sweep_s: float,
    output_path: Path,
):
    """
    Uses REAL measured numbers (see generate_training_data.py and
    train_surrogate.py output) -- not invented figures. This is the
    project's core "why a surrogate" visual.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ["Physics campaign\n(200 radii x 50k MC samples)", "Surrogate\n(200 radii)"]
    times = [physics_sweep_s, surrogate_sweep_s]
    colors = ["tab:orange", "tab:green"]

    bars = ax.bar(labels, times, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Time (s, log scale)")
    ax.set_title(f"Search-time comparison "
                 f"(~{physics_sweep_s / surrogate_sweep_s:,.0f}x speedup)")

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                 f"{t:.4f}s", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    r, p99 = load_training_data(DEFAULT_DATA_PATH)
    r_flat = r.flatten()

    plot_springback_vs_radius(r_flat, p99, OUTPUT_DIR / "springback_vs_radius.png")
    print(f"Wrote {OUTPUT_DIR / 'springback_vs_radius.png'}")

    with open(DEFAULT_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    plot_physics_vs_surrogate(r_flat, p99, model, OUTPUT_DIR / "physics_vs_surrogate.png")
    print(f"Wrote {OUTPUT_DIR / 'physics_vs_surrogate.png'}")

    # These two numbers should be replaced with your own freshly-measured
    # values from generate_training_data.py / train_surrogate.py output
    # if you regenerate the dataset -- don't let this go stale relative
    # to what's actually reported in the README.
    plot_timing_comparison(
        physics_sweep_s=9.43,
        surrogate_sweep_s=0.00033,
        output_path=OUTPUT_DIR / "timing_comparison.png",
    )
    print(f"Wrote {OUTPUT_DIR / 'timing_comparison.png'}")


if __name__ == "__main__":
    main()
