"""Tests for search/search_loop.py -- bisection convergence and the stop
conditions that matter for the 'never run unbounded' production framing."""
import tempfile
from pathlib import Path

import pytest

from domain import DesignParameters, PhysicsEvaluator
from physics import bend_model as bm
from search.search_loop import RadiusSearchLog, SearchConfig, bisection_search


@pytest.fixture
def tmp_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield RadiusSearchLog(Path(tmpdir) / "test_run.jsonl")


class TestBisectionSearch:
    def test_converges_within_max_iterations(self, tmp_log):
        evaluator = PhysicsEvaluator(n_samples=300, seed=42)
        config = SearchConfig(r_lo=bm.crack_floor(), r_hi=8.0, tolerance_mm=0.05, max_iterations=30)

        result = bisection_search(evaluator, config, tmp_log)

        assert result["iterations"] <= config.max_iterations
        assert result["status"] in ("converged", "trivial", "max_iterations_reached")

    def test_converges_to_expected_band(self, tmp_log):
        """The maximum feasible radius (springback <= 2.0mm limit) was
        independently verified (physics pre-check + Monte Carlo campaign)
        to sit between r=5.5mm and r=6.0mm. This guards against silent
        drift if the physics model or spec constants ever change."""
        evaluator = PhysicsEvaluator(n_samples=300, seed=42)
        config = SearchConfig(r_lo=bm.crack_floor(), r_hi=8.0, tolerance_mm=0.05, max_iterations=30)

        result = bisection_search(evaluator, config, tmp_log)

        assert result["status"] == "converged"
        assert 5.5 < result["r_max_feasible"] < 6.0

    def test_infeasible_branch_triggers_when_crack_floor_itself_fails(self, tmp_log):
        """If the springback limit were tightened below what's achievable
        even at the crack floor, the search should report 'infeasible'
        rather than silently returning a wrong answer. Simulate this with
        a monkeypatched evaluator whose r_lo always fails."""

        class AlwaysInfeasibleAtFloor:
            def evaluate(self, params):
                from domain import EvaluationResult
                is_floor = params.r_bend == pytest.approx(bm.crack_floor())
                return EvaluationResult(
                    p99_springback_mm=99.0 if is_floor else 1.0,
                    max_springback_mm=99.0 if is_floor else 1.0,
                    feasible=not is_floor,
                    crack_constraint_ok=True,
                    springback_constraint_ok=not is_floor,
                    source="physics",
                    runtime_s=0.0,
                    n_samples=1,
                )

        config = SearchConfig(r_lo=bm.crack_floor(), r_hi=8.0)
        result = bisection_search(AlwaysInfeasibleAtFloor(), config, tmp_log)

        assert result["status"] == "infeasible"
        assert result["r_max_feasible"] is None

    def test_respects_hard_iteration_cap(self, tmp_log):
        """Stop-condition safety net: even with a very tight tolerance
        that would otherwise force many iterations, the loop must never
        exceed max_iterations."""
        evaluator = PhysicsEvaluator(n_samples=50, seed=42)
        config = SearchConfig(r_lo=bm.crack_floor(), r_hi=8.0, tolerance_mm=0.0001, max_iterations=5)

        result = bisection_search(evaluator, config, tmp_log)

        assert result["iterations"] <= 5


class TestSearchLogging:
    def test_every_candidate_is_logged(self, tmp_log):
        evaluator = PhysicsEvaluator(n_samples=100, seed=42)
        config = SearchConfig(r_lo=bm.crack_floor(), r_hi=8.0, max_iterations=10)

        bisection_search(evaluator, config, tmp_log)

        with open(tmp_log.path) as f:
            lines = f.readlines()
        assert len(lines) >= 2  # at minimum, the two bracket-endpoint evaluations

    def test_log_entries_include_geometry_hash_and_rationale(self, tmp_log):
        params = DesignParameters(r_bend=5.5)
        evaluator = PhysicsEvaluator(n_samples=100, seed=42)
        result = evaluator.evaluate(params)

        entry = tmp_log.record(params, result)

        assert "geometry_hash" in entry
        assert "rationale" in entry
        assert entry["geometry_hash"] == params.geometry_hash()
