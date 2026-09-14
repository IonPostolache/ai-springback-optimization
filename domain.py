"""
Typed domain model + Evaluator protocol for the bend-radius optimization
project.

This is the one interface the rest of the pipeline depends on:
    DesignParameters -> Evaluator.evaluate() -> EvaluationResult

Week 1 implements PhysicsEvaluator (Monte Carlo campaign over the textbook
springback formula in bend_model.py). Week 2 adds SurrogateEvaluator behind
the SAME interface, so the search loop, logging, and plotting code never
change -- only which evaluator is passed in.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, asdict
from typing import Protocol, Literal

import numpy as np

from physics import bend_model as bm


@dataclass(frozen=True)
class DesignParameters:
    """The one design variable in this project: the bend radius."""
    r_bend: float  # mm

    def geometry_hash(self) -> str:
        """Stable hash of the parameters, for provenance/logging."""
        payload = f"r_bend={self.r_bend:.6f}"
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class EvaluationResult:
    p99_springback_mm: float
    max_springback_mm: float
    feasible: bool
    crack_constraint_ok: bool
    springback_constraint_ok: bool
    source: Literal["physics", "surrogate"]
    runtime_s: float
    n_samples: int | None = None  # None for surrogate (single fast call)

    def as_log_dict(self, params: DesignParameters, rationale: str = "") -> dict:
        d = asdict(self)
        d["r_bend_mm"] = params.r_bend
        d["geometry_hash"] = params.geometry_hash()
        d["rationale"] = rationale
        return d


class Evaluator(Protocol):
    def evaluate(self, params: DesignParameters) -> EvaluationResult: ...


SPRINGBACK_LIMIT_MM = 2.0
THICKNESS_TOL = 0.10
YIELD_TOL = 0.05


class PhysicsEvaluator:
    """
    Ground-truth evaluator: runs a Monte Carlo campaign (over thickness and
    yield strength scatter) through the textbook springback formula for a
    given bend radius. This is the "expensive" evaluation -- expensive
    because it's a campaign of n_samples physics calls, not because any
    single call is slow.
    """

    def __init__(
        self,
        n_samples: int = 500,
        seed: int | None = None,
        thickness_nominal_mm: float = bm.T_NOM,
        springback_limit_mm: float = SPRINGBACK_LIMIT_MM,
        thickness_tol: float = THICKNESS_TOL,
        yield_tol: float = YIELD_TOL,
    ):
        self.n_samples = n_samples
        self.seed = seed
        self.thickness_nominal_mm = thickness_nominal_mm
        self.springback_limit_mm = springback_limit_mm
        self.thickness_tol = thickness_tol
        self.yield_tol = yield_tol

    def evaluate(self, params: DesignParameters) -> EvaluationResult:
        t0 = time.perf_counter()

        r_floor = bm.crack_floor(
            t_nom=self.thickness_nominal_mm,
            t_tol=self.thickness_tol,
        )
        crack_ok = params.r_bend >= r_floor

        mc = bm.monte_carlo_p99_springback(
            params.r_bend,
            n_samples=self.n_samples,
            t_tol=self.thickness_tol,
            sy_tol=self.yield_tol,
            seed=self.seed,
        )
        springback_ok = mc["p99"] <= self.springback_limit_mm

        runtime = time.perf_counter() - t0

        return EvaluationResult(
            p99_springback_mm=mc["p99"],
            max_springback_mm=mc["max"],
            feasible=crack_ok and springback_ok,
            crack_constraint_ok=crack_ok,
            springback_constraint_ok=springback_ok,
            source="physics",
            runtime_s=runtime,
            n_samples=self.n_samples,
        )


class SurrogateEvaluator:
    """
    Week-2 stub. Will be trained on (r -> P99 springback) pairs generated
    by running PhysicsEvaluator across a range of radii, then swapped in
    here behind the same interface. Until trained, raises -- this is
    intentional so the pipeline fails loudly rather than silently
    returning nonsense if mode="surrogate" is selected before day 8-9 work
    is done.
    """

    def __init__(self, model=None):
        self.model = model  # trained regressor, e.g. sklearn GradientBoostingRegressor

    def evaluate(self, params: DesignParameters) -> EvaluationResult:
        if self.model is None:
            raise NotImplementedError(
                "SurrogateEvaluator has no trained model yet (week-2 item). "
                "Use PhysicsEvaluator for week-1 development."
            )
        t0 = time.perf_counter()
        p99_pred = float(self.model.predict(np.array([[params.r_bend]]))[0])
        r_floor = bm.crack_floor(t_nom=bm.T_NOM, t_tol=THICKNESS_TOL)
        crack_ok = params.r_bend >= r_floor
        springback_ok = p99_pred <= SPRINGBACK_LIMIT_MM
        runtime = time.perf_counter() - t0
        return EvaluationResult(
            p99_springback_mm=p99_pred,
            max_springback_mm=p99_pred,  # surrogate predicts P99 only; max not modeled separately
            feasible=crack_ok and springback_ok,
            crack_constraint_ok=crack_ok,
            springback_constraint_ok=springback_ok,
            source="surrogate",
            runtime_s=runtime,
            n_samples=None,
        )


def get_evaluator(mode: Literal["physics", "surrogate"] = "physics", **kwargs) -> Evaluator:
    if mode == "physics":
        return PhysicsEvaluator(**kwargs)
    elif mode == "surrogate":
        return SurrogateEvaluator(**kwargs)
    raise ValueError(f"Unknown evaluator mode: {mode}")


if __name__ == "__main__":
    evaluator = get_evaluator(mode="physics", n_samples=500, seed=42)

    print(f"{'r(mm)':>7} {'P99(mm)':>10} {'max(mm)':>10} {'feasible':>9} {'runtime(s)':>11} {'hash':>14}")
    for r in [4.0, 5.0, 5.5, 5.6, 5.7, 6.0, 8.0]:
        params = DesignParameters(r_bend=r)
        result = evaluator.evaluate(params)
        print(f"{r:7.2f} {result.p99_springback_mm:10.4f} {result.max_springback_mm:10.4f} "
              f"{str(result.feasible):>9} {result.runtime_s:11.5f} {params.geometry_hash():>14}")
