
 AI Springback Optimization

### The task statement

> Given a 50mm flange in nominal 1.2mm sheet steel, find the minimum bend radius `r` such that `r ≥ 3×t_max` (crack avoidance) and the flange-tip springback stays ≤ 2.0mm under ±10% thickness and ±5% yield-strength variation, sampled via Monte Carlo.

### Validated physics

- **Model:** standard textbook elastic springback-ratio formula, `Ks = R_i/R_f = 4x³-3x+1` where `x = R_i·σy/(E·t)` — citable (Marciniak/Duncan/Hu, Boljanovic), not a from-scratch derivation you'd have to defend line-by-line.
- **Cross-validated** against an independent through-thickness-integration model: same monotonic shape, same direction, constant ~0.57 scale factor — confirms the physics is sound, not a coincidence of curve-fitting.
- **Crack floor:** `r ≥ 3×1.32mm = 3.96mm` (correctly uses worst-case *thick* end).
- **Springback crossing:** confirmed numerically at **r≈5.5-5.7mm** — comfortably above the crack floor, a genuinely non-trivial search, not dominated by either constraint trivially.
- **Nice detail for the README:** the two constraints bind at *opposite* ends of the thickness tolerance (crack at thick, springback at thin) — shows you're not throwing uncertainty in decoratively.
- **Force dropped entirely** — in this model it doesn't depend on radius at all, so it structurally can't create a trade-off; keeping it would've been decorative. Friction dropped with it.

---

## Day-by-day plan

**Day 1 — CAD + typed interfaces.** build123d parametric flange (fixed 50mm length, variable radius `r`), STEP export, geometry validation, geometry hash. `DesignParameters`/`EvaluationResult` dataclasses, `Evaluator` protocol with `mode="physics"/"surrogate"` stub from the start.

**Day 2 — Physics evaluator + Monte Carlo campaign.** Implement `evaluate_springback` (already validated above) as `PhysicsEvaluator.evaluate(r)`: sample N draws (start ~200-500) over thickness (±10%) and yield strength (±5%), return P99 and max springback. This is your honest per-radius cost.

**Day 3 — Search loop.** Simple bisection/scan over `r`, starting at the crack floor, using the physics evaluator. Log every candidate: `r, P99_springback, max_springback, feasible, geometry_hash`. Rejected/infeasible candidates get a suggested-fix message (vcad-style: "SpringbackExceedsLimit at r=5.0mm → try r≥5.6mm").

**Day 4 — LLM at the edges.** Front: parse "find the minimum bend radius keeping springback under 2mm across ±10% thickness variation" into structured constraints JSON. Back: explain the final result from logged numbers only. No LLM inside the search.

**Day 5 — Robustness pass.** Repeated runs, edge-case radii, confirm the bisection converges cleanly and rejected-candidate logging is complete.

**Day 6 — Surrogate + validation reporting.** Generate training data (`r → P99` pairs from campaigns across the search range), train a small regressor (GradientBoosting or similar), report R²/MAPE against held-out physics evaluations. Plot: springback vs. radius (physics vs. surrogate overlay), campaign-time vs. surrogate-time comparison using your actual measured numbers.

**Day 7 — README.** Task statement up top. PLM interface-contract framing. Manual-vs-automated timing. Stop-condition rationale. Rejected-candidates examples. Data-provenance sentence. Precise surrogate-honesty paragraph (parameter-vector surrogate vs. NC's geometry-native deep learning). Citations to LaDEEP/DDACS/ENSIMA as evidence this problem class is real. One line on why you chose closed-form physics over full FEA (execution-risk trade-off for a one-week solo build, workflow pattern transfers).

**Week 2 (optional, additive only):** active-learning refinement, small API boundary, visualization polish — in that priority order, only if week 1 finishes early.