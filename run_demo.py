"""
End-to-end demo: ties CAD + physics + search + LLM edges into one runnable
story.

    natural-language requirement
              |
              v
      [LLM] intent_parser  --(falls back to hardcoded spec if LLM unreachable)
              |
              v
      structured EngineeringSpec
              |
              v
      [search] bisection_search  (maximum feasible radius)
              |
              v
      [cad] build_flange  (final winning design -> STEP export)
              |
              v
      [LLM] explainer  --(falls back to a templated summary if LLM unreachable)
              |
              v
      printed result + exported STEP + log file

Run with:  python -m run_demo
Or:        python -m run_demo --requirement "your own sentence here"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cad.flange import build_flange
from domain import get_evaluator
from physics import bend_model as bm
from search.search_loop import RadiusSearchLog, SearchConfig, bisection_search

DEFAULT_REQUIREMENT = (
    "Find the maximum bend radius that keeps springback under 2mm, "
    "accounting for plus or minus 10 percent thickness variation and "
    "plus or minus 5 percent yield strength variation. The flange "
    "length must not change."
)
# DEFAULT_REQUIREMENT = (
#     "Find the maximum bend radius that keeps springback under 3.5mm, "
#     "accounting for plus or minus 20 percent thickness variation and "
#     "plus or minus 10 percent yield strength variation. The flange "
#     "length must not change."
# )


def try_parse_intent(requirement_text: str):
    """
    Attempts the LLM front edge. Falls back to the locked default spec
    (matching the validated project constants) if the LLM backend isn't
    reachable -- so the rest of the pipeline is always demoable even
    without LM Studio running. Prints which path was taken, honestly.
    """
    try:
        from llm.intent_parser import parse_requirement
        spec = parse_requirement(requirement_text)
        print(f"[LLM] Parsed requirement via live backend: {spec}")
        return spec
    except Exception as e:
        print(f"[LLM] Backend unreachable or parsing failed ({type(e).__name__}: {e})")
        print("[LLM] Falling back to the locked default spec.")
        from llm.intent_parser import EngineeringSpec
        return EngineeringSpec(
            objective="maximize_radius",
            springback_limit_mm=2.0,
            thickness_nominal_mm=bm.T_NOM,
            thickness_tolerance_pct=10.0,
            yield_strength_tolerance_pct=5.0,
            frozen_features=["flange_length"],
        )

def try_explain_result(
    search_result: dict,
    final_evaluation: dict,
    springback_limit_mm: float,
    ) -> str:
    """
    Attempts the LLM back edge. Falls back to a templated (non-LLM)
    summary if unreachable -- same honesty-about-fallback pattern as
    try_parse_intent above.
    """
    try:
        from llm.explainer import explain_result
        explanation = explain_result(search_result, final_evaluation)
        print("[LLM] Explanation generated via live backend.")
        return explanation
    except Exception as e:
        print(f"[LLM] Backend unreachable or explanation failed ({type(e).__name__}: {e})")
        print("[LLM] Falling back to a templated summary.")
        r = search_result.get("r_max_feasible")
        p99 = final_evaluation.get("p99_springback_mm")
        return (
            f"[templated fallback] The maximum feasible bend radius is "
            f"{r:.3f}mm, where worst-case (P99) springback reaches "
            f"{p99:.3f}mm -- right at the {springback_limit_mm:.1f}mm limit. "
            f"This is the largest radius the die can use (easier tooling, "
            f"less wear, lower force) before springback exceeds spec "
            f"across the expected material scatter."
        )


def run_demo(requirement_text: str, output_dir: Path = Path("outputs")):
    print("=" * 70)
    print("AI-Assisted Robust Sheet-Metal Bend-Radius Optimization")
    print("=" * 70)
    print()
    print(f"Requirement: {requirement_text}")
    print()

    # 1. Parse intent (LLM front edge, with fallback).
    spec = try_parse_intent(requirement_text)
    print()

    # 2. Search: find the maximum feasible radius.
    r_floor = bm.crack_floor(t_nom=spec.thickness_nominal_mm,
                              t_tol=spec.thickness_tolerance_pct / 100)
    config = SearchConfig(r_lo=r_floor, r_hi=10.0, tolerance_mm=0.05, max_iterations=30)

    # Use the physics evaluator for the final design decision.
    # The surrogate is intended as an acceleration strategy: in a real CAE
    # workflow, expensive simulations would be used to generate a DOE,
    # train and validate the surrogate, and promising candidates would then
    # be verified against the high-fidelity solver.
    evaluator = get_evaluator(
        mode="physics",
        # mode="surrogate",
        n_samples=500,
        seed=42,
        thickness_nominal_mm=spec.thickness_nominal_mm,
        springback_limit_mm=spec.springback_limit_mm,
        thickness_tol=spec.thickness_tolerance_pct / 100,
        yield_tol=spec.yield_strength_tolerance_pct / 100,
    )

    log_path = output_dir / "logs" / "demo_run.jsonl"
    log = RadiusSearchLog(log_path)

    print(f"Searching for maximum feasible radius (crack floor: {r_floor:.3f}mm)...")
    search_result = bisection_search(evaluator, config, log)
    print(f"Search result: {search_result}")
    print()

    if search_result["r_max_feasible"] is None:
        print("No feasible design found -- check the spec (see search_result 'reason').")
        return 1

    r_final = search_result["r_max_feasible"]
    from domain import DesignParameters
    final_eval = evaluator.evaluate(DesignParameters(r_bend=r_final))

    # 3. Generate final CAD for the winning design.
    step_path = output_dir / "cad" / f"final_flange_r{r_final:.3f}.step"
    geometry = build_flange(r_final, export_path=step_path)
    print(f"Final CAD exported: {geometry.step_path}")
    print(f"  Geometry hash: {geometry.geometry_hash}")
    print(f"  Volume: {geometry.volume_mm3:.1f} mm^3")
    print()

    # 4. Explain the result (LLM back edge, with fallback).
    final_eval_dict = final_eval.as_log_dict(DesignParameters(r_bend=r_final))
    explanation = try_explain_result(
        search_result,
        final_eval_dict,
        spec.springback_limit_mm,
    )
    print()
    print("-" * 70)
    print("EXPLANATION")
    print("-" * 70)
    print(explanation)
    print()

    print("=" * 70)
    print(f"Final design: r = {r_final:.3f}mm")
    print(f"  P99 springback: {final_eval.p99_springback_mm:.4f}mm (limit: {spec.springback_limit_mm}mm)")
    print(f"  Crack constraint OK: {final_eval.crack_constraint_ok}")
    print(f"  Full search log: {log_path}")
    print(f"  Final CAD: {step_path}")
    print("=" * 70)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Run the bend-radius optimization demo end-to-end.")
    parser.add_argument(
        "--requirement",
        type=str,
        default=DEFAULT_REQUIREMENT,
        help="Natural-language engineering requirement to parse and solve.",
    )
    args = parser.parse_args()

    sys.exit(run_demo(args.requirement))


if __name__ == "__main__":
    main()
