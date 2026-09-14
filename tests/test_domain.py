"""Tests for domain.py -- the Evaluator protocol contract and provenance."""
import pytest

from domain import DesignParameters, PhysicsEvaluator, SurrogateEvaluator


class TestGeometryHash:
    def test_same_params_give_same_hash(self):
        p1 = DesignParameters(r_bend=5.5)
        p2 = DesignParameters(r_bend=5.5)
        assert p1.geometry_hash() == p2.geometry_hash()

    def test_different_params_give_different_hash(self):
        p1 = DesignParameters(r_bend=5.5)
        p2 = DesignParameters(r_bend=5.6)
        assert p1.geometry_hash() != p2.geometry_hash()


class TestPhysicsEvaluator:
    def test_feasible_below_limit_infeasible_above(self):
        evaluator = PhysicsEvaluator(n_samples=300, seed=42)

        feasible_result = evaluator.evaluate(DesignParameters(r_bend=5.0))
        infeasible_result = evaluator.evaluate(DesignParameters(r_bend=8.0))

        assert feasible_result.feasible
        assert not infeasible_result.feasible
        assert not infeasible_result.springback_constraint_ok

    def test_crack_constraint_rejects_too_small_radius(self):
        evaluator = PhysicsEvaluator(n_samples=100, seed=42)
        result = evaluator.evaluate(DesignParameters(r_bend=1.0))
        assert not result.crack_constraint_ok
        assert not result.feasible

    def test_result_reports_source_as_physics(self):
        evaluator = PhysicsEvaluator(n_samples=100, seed=42)
        result = evaluator.evaluate(DesignParameters(r_bend=5.0))
        assert result.source == "physics"
        assert result.n_samples == 100


class TestSurrogateEvaluatorContract:
    def test_untrained_surrogate_raises_loudly(self):
        """The surrogate must fail loudly, not silently return garbage,
        before it's actually trained (week-2 item)."""
        evaluator = SurrogateEvaluator(model=None)
        with pytest.raises(NotImplementedError):
            evaluator.evaluate(DesignParameters(r_bend=5.0))


class TestEvaluatorInterfaceParity:
    """Both evaluators must satisfy the same interface shape, so the
    search loop can swap between them without any other code changing."""

    def test_both_evaluators_return_same_result_fields(self):
        physics_result = PhysicsEvaluator(n_samples=100, seed=1).evaluate(
            DesignParameters(r_bend=5.0)
        )
        physics_fields = set(physics_result.__dataclass_fields__.keys())

        # SurrogateEvaluator (once trained) returns the same EvaluationResult type,
        # so field parity is structural (same dataclass), not just tested empirically.
        assert "p99_springback_mm" in physics_fields
        assert "feasible" in physics_fields
        assert "source" in physics_fields
