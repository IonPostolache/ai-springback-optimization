"""
Search loop: find the maximum feasible bend radius via bisection over the
Evaluator interface (larger radius = easier tooling/die wear/force, but
more springback -- see bisection_search() docstring for why this is a
maximum-radius, not minimum-radius, search). Deliberately NOT the LLM --
candidate generation is a plain deterministic bisection so the numerical
search stays reproducible and physics-grounded. The LLM (see llm/) only
parses the initial requirement and explains the final result.

Logs every evaluated candidate as one JSON line, including rejected
candidates with a vcad-style "violation + suggested fix" message.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from domain import DesignParameters, Evaluator, EvaluationResult, get_evaluator
from physics import bend_model as bm


@dataclass
class SearchConfig:
    r_lo: float  # lower search bound (should be >= crack floor)
    r_hi: float  # upper search bound (a radius known to be feasible)
    tolerance_mm: float = 0.05  # stop when [r_lo, r_hi] narrower than this
    max_iterations: int = 30  # hard stop condition -- see README rationale


def suggested_fix_message(params: DesignParameters, result: EvaluationResult) -> str:
    """
    vcad-style violation message: name the constraint that failed and
    propose a concrete correction, rather than a bare pass/fail.
    """
    if not result.crack_constraint_ok:
        floor = bm.crack_floor()
        return (f"BendRadiusBelowCrackLimit at r={params.r_bend:.3f}mm "
                f"(requires r >= {floor:.3f}mm)")
    if not result.springback_constraint_ok:
        return (f"SpringbackExceedsLimit at r={params.r_bend:.3f}mm "
                f"(P99={result.p99_springback_mm:.3f}mm > 2.0mm limit); "
                f"try decreasing r")
    return "OK"


class RadiusSearchLog:
    """Append-only JSON-lines log of every evaluated candidate."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._iteration = 0

    def record(self, params: DesignParameters, result: EvaluationResult):
        self._iteration += 1
        entry = result.as_log_dict(params, rationale=suggested_fix_message(params, result))
        entry["iteration"] = self._iteration
        entry["timestamp"] = time.time()
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry


def bisection_search(
    evaluator: Evaluator,
    config: SearchConfig,
    log: RadiusSearchLog,
) -> dict:
    """
    Find the MAXIMUM feasible radius via bisection on feasibility.

    Why maximum, not minimum: springback increases monotonically with r
    (confirmed in physics/bend_model.py and tests/test_bend_model.py), so
    a "minimum radius s.t. springback <= limit" search is always trivial
    -- the crack floor itself already satisfies any springback limit
    above its own worst-case value, since smaller r means less
    springback. The springback constraint only ever excludes an upper
    region of the radius range, never a lower one.

    The physically meaningful question is therefore: "what's the largest
    radius I can tool the die to (easier tooling, less die wear, lower
    force) while keeping springback within tolerance?" -- an upper-bound
    search, where r_lo = crack_floor (always feasible on springback, by
    the monotonicity above) and r_hi is a radius known to violate the
    springback constraint. The feasible window is
    [crack_floor, r_max_springback].

    Stop condition: bracket width < tolerance_mm, OR max_iterations
    reached (hard safety net so the loop can never run unbounded --
    see README for why this matters in a customer/HPC deployment).
    """
    r_lo, r_hi = config.r_lo, config.r_hi

    # Sanity checks on the bracket before searching.
    result_lo = evaluator.evaluate(DesignParameters(r_bend=r_lo))
    log.record(DesignParameters(r_bend=r_lo), result_lo)
    result_hi = evaluator.evaluate(DesignParameters(r_bend=r_hi))
    log.record(DesignParameters(r_bend=r_hi), result_hi)

    if not result_lo.feasible:
        return {
            "status": "infeasible",
            "r_max_feasible": None,
            "reason": f"r_lo={r_lo}mm is already infeasible; the crack floor "
                      f"itself violates a constraint -- check the spec",
            "iterations": 2,
        }
    if result_hi.feasible:
        return {
            "status": "trivial",
            "r_max_feasible": r_hi,
            "reason": "r_hi is still feasible; search bracket was not wide enough to be interesting",
            "iterations": 2,
        }

    iteration = 2
    while (r_hi - r_lo) > config.tolerance_mm and iteration < config.max_iterations:
        r_mid = (r_lo + r_hi) / 2.0
        result_mid = evaluator.evaluate(DesignParameters(r_bend=r_mid))
        log.record(DesignParameters(r_bend=r_mid), result_mid)
        iteration += 1

        if result_mid.feasible:
            r_lo = r_mid
        else:
            r_hi = r_mid

    status = "converged" if (r_hi - r_lo) <= config.tolerance_mm else "max_iterations_reached"
    return {
        "status": status,
        "r_max_feasible": r_lo,  # r_lo is always the largest confirmed-feasible value
        "bracket_width_mm": r_hi - r_lo,
        "iterations": iteration,
    }


def main():
    r_floor = bm.crack_floor()
    config = SearchConfig(r_lo=r_floor, r_hi=8.0, tolerance_mm=0.05, max_iterations=30)

    evaluator = get_evaluator(mode="physics", n_samples=500, seed=42)
    log = RadiusSearchLog(Path("logs/search_run.jsonl"))

    result = bisection_search(evaluator, config, log)

    print("Search result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if result["r_max_feasible"] is not None:
        final = evaluator.evaluate(DesignParameters(r_bend=result["r_max_feasible"]))
        print()
        print(f"Final design: r = {result['r_max_feasible']:.3f}mm")
        print(f"  P99 springback: {final.p99_springback_mm:.4f}mm (limit 2.0mm)")
        print(f"  Crack constraint OK: {final.crack_constraint_ok}")


if __name__ == "__main__":
    main()
