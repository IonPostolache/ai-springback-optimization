# AI-Assisted Robust Sheet-Metal Bend-Radius Optimization

## Problem

A stamping engineer has a released bracket design. The mounting interface — flange length, bolt/mating geometry — is frozen in PLM and cannot change. A new requirement arrives: tool the die to as generous a bend radius as possible (easier tooling, less die wear, lower forming force), without exceeding the part's springback tolerance across normal incoming material scatter.

Given a 50 mm flange in nominal 1.2 mm sheet steel, find the **maximum bend radius** `r` such that:

- `r ≥ 3 × t_max` (crack avoidance — binds at the *thick* end of the tolerance band)
- flange-tip springback ≤ 2.0 mm under Monte Carlo sampling of:
  - thickness: ±10%
  - yield strength: ±5%
  (springback binds at the *thin* end of the tolerance band — the opposite direction from the crack constraint; the design must satisfy both simultaneously)

The objective is deliberately framed as a constrained robustness search, not a single-point calculation — a real stamping engineer asks "will this radius hold across the material I'll actually receive," not "what's the springback at nominal thickness."

**Why closed-form physics instead of full FEA:** mesh generation and CalculiX boundary-condition setup were the highest execution-risk, least-familiar part of the available toolchain. A validated closed-form elastic-recovery model was chosen deliberately to keep the *workflow* (physics campaign → surrogate → accelerated search → validation) reliable and demoable; the same interface (`Evaluator.evaluate()`) would accept a full FEA solver or a predictor without changing the search, logging, or LLM code.

**Data provenance:** all inference in this project runs locally against an LM Studio endpoint on the local network — no customer geometry, simulation data, or requirement text leaves the local environment.

---

## How to run

### 1. Install

```bash
git clone https://github.com/IonPostolache/ai-springback-optimization.git

cd ai-springback-optimization

python -m venv .venv

source .venv/bin/activate

pip install -e .
```

### 2. Configure the LLM backend

Copy `.env.example` to `.env` and point it at any OpenAI-compatible endpoint (LM Studio, Ollama, vLLM). The pipeline runs and produces a complete result even without a reachable LLM — see [LLM Integration](#llm-integration).


### 3. Run the end-to-end demo

```bash
python run_demo.py
```

The demo:

1. Parses a natural-language engineering requirement into a structured specification.
2. Runs the robust optimization search using the physics evaluator.
3. Generates the corresponding parametric CAD geometry with build123d.
4. Exports the final design as a STEP file.
5. Generates a plain-English explanation of the result.

Outputs are written to:

```
outputs/
├── cad/
│   └── final_flange_r*.step
└── logs/
    └── demo_run.jsonl
```

### 4. Run the tests

```bash
pytest
```

### 5. Run individual components

```bash
python cad/flange.py                        # generate example CAD geometries
python surrogate/generate_training_data.py  # generate surrogate training data
python surrogate/train_surrogate.py         # train the surrogate model
python plots/make_plots.py                  # generate plots
python llm/client.py                        # test the LLM connection
```

---

## Architecture

```
Natural-language engineering requirement
                  │
                  ▼
          LLM intent parser
                  │
                  ▼
        Structured engineering spec
                  │
                  ▼
        Robust optimization search
                  │
          ┌───────┴────────┐
          ▼                ▼
   Physics evaluator   Surrogate evaluator
          │                │
          └───────┬────────┘
                  ▼
          Final candidate
                  │
                  ▼
       Parametric build123d CAD
                  │
                  ▼
              STEP export
                  │
                  ▼
          LLM result explainer
```

The LLM is intentionally kept at the edges of the system. It does not propose optimization candidates, modify the physics model, or decide whether a design is feasible — the numerical search stays reproducible and physics-grounded regardless of language-model variability. If the LLM backend is unreachable, the pipeline falls back to a locked default specification and a templated explanation rather than failing (see `run_demo.py`) — the search, CAD generation, and validation always complete.

**Production mapping** (what each stage would be in a real customer deployment):

| This project | Production equivalent |
|---|---|
| build123d / STEP | NX Open, CATIA CAA, or the customer's native CAD API |
| Physics evaluator | Customer's Ansys/Abaqus/Star-CCM+ job, or a trained predictor |
| Local search loop | Monitored job on a cloud/HPC job scheduler |
| Local LLM (LM Studio) | Enterprise-hosted inference, still respecting data locality constraints |
| JSON-lines log | PLM-integrated provenance/audit trail |

### Validated physics

- **Model:** standard textbook elastic springback-ratio formula, `Ks = R_i/R_f = 4x^3 - 3x + 1` where `x = R_i*sigma_y/(E*t)` (Marciniak/Duncan/Hu; Boljanovic, *Sheet Metal Forming Processes*).
- **Cross-checked** against an independently derived through-thickness-integration model: both exhibit the same monotonic trend and direction, with an approximately constant ~0.57 scaling difference. This is an internal consistency check, not validation against experimental data or production FEA — stated explicitly rather than implied.
- **Crack floor:** `r >= 3 x 1.32mm = 3.96mm` (correctly uses worst-case *thick* end of the thickness tolerance).
- **Springback crossing:** confirmed numerically at **r ~ 5.5-5.7mm** — comfortably above the crack floor, a genuinely non-trivial search, not trivially dominated by either constraint alone.
- Monotonicity, the crack-floor calculation, and the feasibility crossing band are all covered by automated tests (`tests/test_bend_model.py`), not just a one-off validation script.

### CAD generation

The flange is generated parametrically in build123d: a filleted centerline (`FilletPolyline`, radius = the design variable `r`) offset by sheet thickness to produce a constant-thickness cross-section, extruded to the bend width. Every generated geometry is validated (non-degenerate, positive volume) and hashed for provenance before export.

### Robust optimization

The search starts from the physically determined crack-avoidance floor and bisects toward the maximum radius at which the springback constraint still holds, using the deterministic `Evaluator` interface. Each evaluation records:

- bend radius
- P99 springback
- maximum sampled springback
- constraint status (crack / springback, independently)
- evaluator source (physics / surrogate)
- runtime
- geometry hash

This produces a traceable optimization history rather than an opaque "the optimizer found this value" result. Infeasible candidates are logged with a named violation and a concrete suggested correction (e.g. `SpringbackExceedsLimit at r=6.00mm -> try decreasing r`), not a bare pass/fail.

**Stop condition:** the search terminates on convergence (bracket width below tolerance) or a hard iteration cap, whichever comes first. The iteration cap exists specifically so an automated loop can never run unbounded in a real deployment — an unbounded search against a real HPC-scheduled solver has a real cost, and a customer-facing automation tool should never be able to consume unbounded compute silently.

### Surrogate model

A machine-learning surrogate approximates `bend radius -> P99 springback`, trained on data generated by running the physics evaluator's full Monte Carlo campaign at each of 200 radii. It is trained on the *campaign output*, not on individual `(r, thickness, yield strength)` samples — deliberately, so the surrogate replaces the expensive campaign as a whole, which is the actual workflow acceleration a fast predictor provides in production, rather than a per-sample speedup alone.

The surrogate is evaluated against held-out physics campaigns using R^2 and MAPE, and is used only to accelerate candidate exploration; the final candidate is always confirmed against the physics evaluator, never accepted on the surrogate's prediction alone.

**On what this surrogate is, honestly:** this is a parameter-vector surrogate (`r -> P99`), not a geometry-native model. Production surrogates operate directly on 3D geometry or mesh representations, which is what lets them generalize across topologies and predict full physical fields rather than a single scalar from a hand-picked parameter. This project's surrogate demonstrates the same *workflow pattern* — expensive simulation campaign replaced by a fast, validated predictor; the representation underneath is deliberately simpler.

The pattern itself is not a toy: published work on stretch-bending surrogates ([LaDEEP](https://github.com/therontau0054/LaDEEP)) reports roughly five orders of magnitude speedup over FEA and has been deployed in an industrial system; datasets like [DDACS](https://github.com/BaumSebastian/DDACS) (32,000+ deep-drawing simulations) exist specifically to train this class of model; and Bayesian-optimization/surrogate approaches to forming simulation (e.g. TU Darmstadt's [ENSIMA](https://github.com/tuda-hpclab/ensima)) are active research directions. This project is a small, self-contained synthesis in that space, built independently rather than adapted from any of the above.

### LLM integration

The LLM has exactly two responsibilities:

- **Front edge:** convert a natural-language engineering requirement into a validated, structured specification. Every field is clamped to physically sensible bounds before it can influence the search or CAD generation — raw model output never reaches the physics layer unchecked.
- **Back edge:** explain the optimization result in plain language, using only the numbers the pipeline actually produced.

The LLM is deliberately excluded from the optimization loop and the physics evaluation itself — candidate generation is a plain, deterministic bisection. This keeps the numerical search reproducible regardless of language-model variability, and avoids the more fundamental issue of asking a language model to make or verify physics claims it cannot ground.

The current implementation targets any OpenAI-compatible local endpoint (developed against LM Studio / LM Link running on a separate machine on the local network), keeping inference local rather than dependent on a cloud API.

### Validation and limitations

This is a workflow demonstrator, not a production stamping solver. Explicit, deliberate simplifications:

- Linear-elastic springback recovery only (no anisotropy, no strain-path effects, no multi-stage forming).
- A single load case / bend geometry (90 degree V-bend, fixed flange length).
- Two uncertain parameters (thickness, yield strength) rather than a full material/process uncertainty budget.
- Candidate search is a 1D bisection over a single design variable; this is the appropriate scope for a simple build, not a claim that the underlying method generalizes to a high-dimensional design space without further work.

---

## Results

For the nominal requirement — thickness 1.2mm +/-10%, yield strength +/-5%, springback limit 2.0mm, objective: maximize bend radius — the robust search converges to:

| Metric | Result |
|---|---:|
| Maximum feasible radius | 5.706 mm |
| P99 springback | 1.9992 mm |
| Crack constraint | Pass |
| Search iterations | 9 |
| Monte Carlo samples/evaluation | 500 |
| Automated tests | 21/21 passed |

### Surrogate performance

Trained on 200 physics campaigns (160 training / 40 held-out test):

| Metric | Result |
|---|---:|
| Held-out R^2 | 0.99953 |
| Held-out MAPE | 0.4442% |
| Training time | 0.025 s |
| 200-radius surrogate sweep | 0.24 ms |
| 200-radius physics campaign sweep | ~9.4 s |

This is a measured ~28,500x speedup for the search-relevant comparison (a full radius sweep), not an invented figure. Note this compares *sweep* time, not single-call time — a single physics evaluation with this closed-form model is already fast; the honest expensive cost is running many campaigns across a design space, which is exactly the cost a production surrogate is built to remove.

### Optimization result

![Springback vs bend radius](plots/output/springback_vs_radius.png)

Springback rises monotonically with bend radius; the feasible region is bounded below by the crack-avoidance floor and above by the springback constraint.

### Physics model vs. surrogate

![Physics vs surrogate](plots/output/physics_vs_surrogate.png)

The surrogate closely reproduces the physics evaluator across the sampled radius range (held-out R^2 = 0.99953, MAPE = 0.44%).

### Evaluation speed

![Physics vs surrogate timing](plots/output/timing_comparison.png)

The surrogate substantially reduces sweep time, making it suitable for accelerating candidate exploration ahead of a final physics-based verification.

---

## Next steps

- **Active-learning refinement:** rather than training the surrogate on a uniform radius sweep, use it to identify the region near the feasibility boundary and concentrate additional physics campaigns there — spending expensive evaluations where they change the answer, not uniformly across the whole range.
