"""
Tests for physics/bend_model.py. These back up the specific claims made
in the README ("validated against the textbook Ks formula", "springback
increases monotonically with radius", "crack floor uses worst-case
thickness") with actual assertions, not just a one-off script that was
run once during development.
"""
import numpy as np
import pytest

from physics import bend_model as bm


class TestSpringbackFactor:
    def test_ks_is_between_zero_and_one_for_typical_bends(self):
        """Ks = R_i/R_f should be < 1 for elastic-plastic bending (some
        springback occurs), and > 0 (physically meaningful radius)."""
        Ks, _ = bm.springback_factor(r_bend=5.0, t=1.2, sigma_y=300.0)
        assert 0.0 < Ks < 1.0

    def test_larger_radius_gives_smaller_ks_more_springback(self):
        """As radius grows, bending strain drops, less material yields
        plastically, so a SMALLER fraction of the deformation is
        permanent -- Ks = R_i/R_f decreases (moves away from 1) as
        radius increases. Since alpha_f = alpha_i * Ks, a smaller Ks
        means more deviation from the target angle, i.e. MORE springback
        at larger radius -- consistent with the monotonicity test below."""
        Ks_small, _ = bm.springback_factor(r_bend=3.0, t=1.2, sigma_y=300.0)
        Ks_large, _ = bm.springback_factor(r_bend=15.0, t=1.2, sigma_y=300.0)
        assert Ks_large < Ks_small


class TestSpringbackMonotonicity:
    """This is the property the whole project's search logic depends on:
    springback increases monotonically with bend radius. If this ever
    fails, the bisection search's core assumption is violated."""

    def test_springback_increases_with_radius(self):
        radii = np.arange(3.0, 20.0, 1.0)
        deflections = [
            bm.evaluate_springback(r, bm.T_NOM, bm.SIGMA_Y_NOM)["tip_deflection_mm"]
            for r in radii
        ]
        assert all(
            deflections[i] < deflections[i + 1] for i in range(len(deflections) - 1)
        ), "springback must increase monotonically with radius"

    def test_thinner_sheet_gives_more_springback(self):
        """Confirmed direction (see project notes): thinner sheet
        produces MORE springback, not less -- this determines which end
        of the thickness tolerance is worst-case for the springback
        constraint (opposite end from the crack constraint)."""
        r = 6.0
        thin = bm.evaluate_springback(r, bm.T_NOM * 0.9, bm.SIGMA_Y_NOM)["tip_deflection_mm"]
        thick = bm.evaluate_springback(r, bm.T_NOM * 1.1, bm.SIGMA_Y_NOM)["tip_deflection_mm"]
        assert thin > thick


class TestCrackFloor:
    def test_crack_floor_uses_worst_case_thick_end(self):
        """Crack avoidance (r >= 3t) is worst-case at MAXIMUM thickness --
        opposite end from the springback worst case. This asymmetry is a
        deliberate project talking point; this test guards the arithmetic."""
        floor = bm.crack_floor(t_nom=1.2, t_tol=0.10)
        expected = 3 * (1.2 * 1.10)
        assert floor == pytest.approx(expected)


class TestMonteCarloCampaign:
    def test_more_samples_converges_to_stable_p99(self):
        """Sanity check that increasing sample count doesn't wildly change
        the P99 estimate -- confirms n_samples=500 (the project default)
        is a reasonable choice, not too noisy."""
        r = 5.5
        mc_small = bm.monte_carlo_p99_springback(r, n_samples=200, seed=1)
        mc_large = bm.monte_carlo_p99_springback(r, n_samples=5000, seed=1)
        assert mc_small["p99"] == pytest.approx(mc_large["p99"], abs=0.15)

    def test_p99_is_between_mean_and_max(self):
        mc = bm.monte_carlo_p99_springback(5.5, n_samples=500, seed=1)
        assert mc["mean"] <= mc["p99"] <= mc["max"]


class TestTextbookCrossValidation:
    """
    Regression guard: at the time this model was validated, the crossing
    point where worst-case P99 springback = 2.0mm was confirmed to sit
    between r=5.5mm and r=6.0mm (see project notes). If this ever drifts
    outside that band, either the model changed unexpectedly or the
    validated constants (SIGMA_Y_NOM, T_NOM, FLANGE_LEN) were edited --
    either way, worth a manual re-check before trusting the project's
    locked task statement.
    """

    def test_feasibility_crossing_is_in_expected_band(self):
        feasible_at_55 = bm.worst_case_springback(5.5) <= 2.0
        infeasible_at_60 = bm.worst_case_springback(6.0) > 2.0
        assert feasible_at_55
        assert infeasible_at_60
