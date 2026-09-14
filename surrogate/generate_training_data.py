"""
Generate surrogate training data: run the (expensive, campaign-level)
PhysicsEvaluator across a range of bend radii, save (r -> P99 springback)
pairs to disk.

This is the data the day-6/week-2 surrogate is trained on. Deliberately
NOT trained on individual (r, t, sigma_y) samples -- see project notes:
the surrogate should replace the whole Monte Carlo campaign per radius,
not just the per-sample physics call, since that's what actually mirrors
Neural Concept's value proposition (fast prediction replaces an expensive
simulation campaign, not a single physics evaluation).

Note on n_mc_samples=50_000: this closed-form formula is fast enough
(vectorized NumPy) that even large sample counts complete in well under a
second per single campaign. The honest "expensive" cost in this project
comes from running MANY campaigns across a dense radius sweep (this
script does exactly that) -- roughly 10s for a 200-radius x 50k-sample
sweep on this machine -- not from inflating any single campaign's sample
count to manufacture slowness. Report this real, measured number in the
README, not an invented one.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from domain import DesignParameters, PhysicsEvaluator
from physics import bend_model as bm

DEFAULT_OUTPUT = Path("data/training_data.csv")


def generate_training_data(
    r_min: float | None = None,
    r_max: float = 10.0,
    n_radii: int = 200,
    n_mc_samples: int = 50_000,
    output_path: Path = DEFAULT_OUTPUT,
    seed: int = 42,
) -> Path:
    """
    Sweep n_radii bend radii between r_min (defaults to the crack floor)
    and r_max, running a full Monte Carlo campaign (n_mc_samples) at each,
    and write the results to a CSV: r_bend, p99_springback_mm,
    max_springback_mm, feasible, campaign_runtime_s.

    campaign_runtime_s is logged explicitly so the day-6/7 "physics
    campaign vs. surrogate" timing comparison uses real measured numbers,
    not invented ones -- consistent with the project's honesty-about-
    runtime framing throughout.
    """
    if r_min is None:
        r_min = bm.crack_floor()

    radii = np.linspace(r_min, r_max, n_radii)
    evaluator = PhysicsEvaluator(n_samples=n_mc_samples, seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total_start = time.perf_counter()

    for r in radii:
        result = evaluator.evaluate(DesignParameters(r_bend=float(r)))
        rows.append({
            "r_bend": r,
            "p99_springback_mm": result.p99_springback_mm,
            "max_springback_mm": result.max_springback_mm,
            "feasible": result.feasible,
            "campaign_runtime_s": result.runtime_s,
        })

    total_runtime = time.perf_counter() - total_start

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} radius campaigns to {output_path}")
    print(f"Total data-generation time: {total_runtime:.2f}s "
          f"({total_runtime / len(rows) * 1000:.1f}ms per campaign on average)")

    return output_path


if __name__ == "__main__":
    generate_training_data()
