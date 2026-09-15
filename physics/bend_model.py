"""
V-bend springback model for a sheet-metal flange, using the standard
textbook elastic springback-ratio (Ks) formula:

    Ks = R_i / R_f = 4*(R_i*sigma_y/(E*t))**3 - 3*(R_i*sigma_y/(E*t)) + 1

(Marciniak/Duncan/Hu and Boljanovic "Sheet Metal Forming Processes";
standard elastic-recovery estimate for thin-sheet bending.)

R_i = bend radius to neutral axis before unloading (punch/die radius + t/2)
R_f = radius after elastic unloading
sigma_y = material yield strength, E = elastic modulus, t = thickness

Angle relationship (arc length preserved through the bend):
    alpha_i * R_i = alpha_f * R_f  =>  alpha_f = alpha_i * Ks

This file replaced an earlier from-scratch through-thickness integration
model; the two were cross-validated (same shape, same monotonic direction,
constant ~0.57 scale-factor difference), and this textbook formula was
chosen going forward for simplicity and citability.

Units: mm, MPa unless noted. Springback reported as flange-tip linear
deflection (mm), which is what's constrained in this project.
"""
import numpy as np

E = 210_000.0      # MPa, Young's modulus
FLANGE_LEN = 50.0  # mm, fixed flange length used to convert angle -> tip deflection
T_NOM = 1.2        # mm, nominal sheet thickness
SIGMA_Y_NOM = 300.0  # MPa, representative automotive sheet steel yield strength
BEND_ANGLE = np.pi / 2  # 90 degree bend


def springback_factor(r_bend, t, sigma_y, E=E):
    """Ks = R_i/R_f, standard elastic-recovery formula."""
    R_i = r_bend + t / 2.0
    x = (R_i * sigma_y) / (E * t)
    Ks = 4 * x**3 - 3 * x + 1
    return Ks, R_i


def evaluate_springback(r_bend, t, sigma_y, flange_len=FLANGE_LEN, bend_angle=BEND_ANGLE):
    """
    r_bend: punch/die bend radius (mm)
    t: sheet thickness (mm)
    sigma_y: yield strength (MPa)

    Returns dict with Ks, R_i, springback angle change, and flange-tip
    linear deflection (mm) -- the quantity constrained in this project.
    """
    Ks, R_i = springback_factor(r_bend, t, sigma_y)
    alpha_f = bend_angle * Ks
    delta_theta = alpha_f - bend_angle
    tip_deflection = flange_len * abs(delta_theta)
    return {
        "Ks": Ks,
        "R_i": R_i,
        "delta_theta_rad": delta_theta,
        "tip_deflection_mm": tip_deflection,
    }


def scan_radius(r_values, t=T_NOM, sigma_y=SIGMA_Y_NOM):
    rows = []
    for r in r_values:
        res = evaluate_springback(r, t, sigma_y)
        rows.append((r, res["tip_deflection_mm"]))
    return rows


def crack_floor(t_nom=T_NOM, t_tol=0.10):
    """Minimum radius to avoid cracking: r >= 3 * t_max (worst-case thick end)."""
    t_max = t_nom * (1 + t_tol)
    return 3 * t_max


def worst_case_springback(r_bend, t_nom=T_NOM, t_tol=0.10, sigma_y_nom=SIGMA_Y_NOM, sy_tol=0.05):
    """
    Corner-based worst case over thickness (+-t_tol) and yield strength
    (+-sy_tol). Springback is worst at the THIN end of the thickness band
    (opposite direction from the crack constraint, which is worst at the
    thick end) -- this asymmetry is deliberate and worth calling out.
    """
    import itertools
    t_lo, t_hi = t_nom * (1 - t_tol), t_nom * (1 + t_tol)
    sy_lo, sy_hi = sigma_y_nom * (1 - sy_tol), sigma_y_nom * (1 + sy_tol)
    worst = 0.0
    for t, sy in itertools.product([t_lo, t_hi], [sy_lo, sy_hi]):
        res = evaluate_springback(r_bend, t, sy)
        worst = max(worst, res["tip_deflection_mm"])
    return worst


def monte_carlo_p99_springback(r_bend, n_samples=500, t_nom=T_NOM, t_tol=0.10,
                                sigma_y_nom=SIGMA_Y_NOM, sy_tol=0.05, seed=None):
    """
    Monte Carlo campaign at a fixed radius: sample thickness and yield
    strength uniformly within tolerance, return P99 and max springback.
    This is the "expensive" per-candidate evaluation the search loop calls,
    and what the surrogate is trained to replace.
    """
    rng = np.random.default_rng(seed)
    t_samples = rng.uniform(t_nom * (1 - t_tol), t_nom * (1 + t_tol), n_samples)
    sy_samples = rng.uniform(sigma_y_nom * (1 - sy_tol), sigma_y_nom * (1 + sy_tol), n_samples)
    deflections = np.array([
        evaluate_springback(r_bend, t, sy)["tip_deflection_mm"]
        for t, sy in zip(t_samples, sy_samples)
    ])
    return {
        "p99": float(np.percentile(deflections, 99)),
        "max": float(deflections.max()),
        "mean": float(deflections.mean()),
        "n_samples": n_samples,
    }


def main():
    r_vals = np.arange(2.0, 20.1, 1.0)
    r_floor = crack_floor()

    print(f"Crack floor r >= 3*t_max = {r_floor:.2f} mm")
    print()
    print("=== Nominal material, nominal thickness ===")
    print(f"{'r(mm)':>7} {'springback_tip(mm)':>20}")
    for r, defl in scan_radius(r_vals):
        print(f"{r:7.1f} {defl:20.4f}")

    print()
    print("=== Worst-case scan (corner-based): thickness x yield strength, both +-bounds ===")
    print(f"{'r(mm)':>7} {'worst_springback(mm)':>22}")
    for r in np.arange(r_floor, 12.0, 0.25):
        worst = worst_case_springback(r)
        mark = "  <-- crosses 2.0mm here" if worst <= 2.0 else ""
        print(f"{r:7.2f} {worst:22.4f}{mark}")

    print()
    print("=== Monte Carlo P99 check (500 samples) at a few candidate radii ===")
    print(f"{'r(mm)':>7} {'P99(mm)':>10} {'max(mm)':>10}")
    for r in [4.0, 5.0, 5.5, 5.6, 5.7, 6.0, 8.0]:
        mc = monte_carlo_p99_springback(r, seed=42)
        print(f"{r:7.2f} {mc['p99']:10.4f} {mc['max']:10.4f}")


if __name__ == "__main__":
    main()
