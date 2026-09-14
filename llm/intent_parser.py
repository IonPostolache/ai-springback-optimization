"""
LLM front edge: parse a natural-language engineering requirement into a
structured spec. This is one of exactly two places the LLM touches this
project (see explainer.py for the other) -- it never proposes candidates
or influences the physics; it only translates intent into structured
inputs that the deterministic search loop then consumes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from llm.client import get_client

SYSTEM_PROMPT = """You are an engineering-requirement parser for a sheet-metal \
bending process. Given a natural-language requirement, extract a structured \
specification. Respond with ONLY a JSON object, no other text, in exactly \
this shape:

{
  "objective": "maximize_radius" | "minimize_radius",
  "springback_limit_mm": <float>,
  "thickness_nominal_mm": <float>,
  "thickness_tolerance_pct": <float>,
  "yield_strength_tolerance_pct": <float>,
  "frozen_features": [<string>, ...]
}

If a value isn't mentioned, use these defaults: thickness_nominal_mm=1.2, \
thickness_tolerance_pct=10, yield_strength_tolerance_pct=5, \
frozen_features=["flange_length"]."""


@dataclass(frozen=True)
class EngineeringSpec:
    objective: str
    springback_limit_mm: float
    thickness_nominal_mm: float
    thickness_tolerance_pct: float
    yield_strength_tolerance_pct: float
    frozen_features: list[str]


def parse_requirement(requirement_text: str) -> EngineeringSpec:
    """
    Parse a natural-language requirement into an EngineeringSpec.
    Every field is validated/clamped after parsing -- raw LLM output never
    reaches the physics or CAD layers unchecked.
    """
    client, model = get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": requirement_text},
        ],
        temperature=0.0,
        max_tokens=300,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences some local models add despite instructions.
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()

    parsed = json.loads(raw)

    return _validate_spec(parsed)


def _validate_spec(parsed: dict) -> EngineeringSpec:
    """
    Validation/clamping gate. Never trust raw LLM output -- clamp to
    physically sensible bounds before it can influence the search or CAD.
    """
    objective = parsed.get("objective", "maximize_radius")
    if objective not in ("maximize_radius", "minimize_radius"):
        objective = "maximize_radius"

    springback_limit = float(parsed.get("springback_limit_mm", 2.0))
    springback_limit = max(0.1, min(springback_limit, 20.0))  # sane physical bounds

    thickness_nom = float(parsed.get("thickness_nominal_mm", 1.2))
    thickness_nom = max(0.3, min(thickness_nom, 10.0))

    thickness_tol = float(parsed.get("thickness_tolerance_pct", 10.0))
    thickness_tol = max(0.0, min(thickness_tol, 30.0))

    yield_tol = float(parsed.get("yield_strength_tolerance_pct", 5.0))
    yield_tol = max(0.0, min(yield_tol, 20.0))

    frozen = parsed.get("frozen_features", ["flange_length"])
    if not isinstance(frozen, list):
        frozen = ["flange_length"]

    return EngineeringSpec(
        objective=objective,
        springback_limit_mm=springback_limit,
        thickness_nominal_mm=thickness_nom,
        thickness_tolerance_pct=thickness_tol,
        yield_strength_tolerance_pct=yield_tol,
        frozen_features=frozen,
    )


if __name__ == "__main__":
    example = (
        "Find the maximum bend radius that keeps springback under 2mm, "
        "accounting for plus or minus 10 percent thickness variation and "
        "plus or minus 5 percent yield strength variation. The flange "
        "length must not change."
    )
    spec = parse_requirement(example)
    print(spec)
