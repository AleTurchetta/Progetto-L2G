# L2G Stress-Testing Project — Full Plan

> PI Controller / Preferential BO · Phases 1–3

---

## 1. Test Case Selection

### Recommendation: Keep the PI Controller, Extend It

The PI tuning case is **the right choice** for this project. The reasons are solid:

- **2D parameter space** (Kp, Ki) is fully visualizable — you can plot the entire ground-truth preference landscape and verify the model is learning the right thing.
- **Metrics are physically interpretable** — overshoot, settling time, MSE map directly to intuitive human language ("too bouncy", "too slow"), making the oracle's behaviour believable.
- **The true optimum is known** — you can use Ziegler-Nichols or pole-placement as a ground truth to compute regret.
- **Switching to abstract benchmarks (Branin, Rosenbrock)** loses the connection to human feedback semantics, which is the whole point of L2G.

### Suggested Extensions (not replacements)

| Extension | Why it's useful for stress-testing |
|-----------|-------------------------------------|
| **PID (3D: Kp, Ki, Kd)** | Tests scaling — does L2G handle a higher-dimensional space? The Kd term adds derivative-based damping, which maps cleanly to "reduce oscillations". |
| **Underdamped plant (zeta=0.1)** | Forces large overshoots at initialization, making feedback about damping more discriminating and stress-testable. |
| **Two conflicting plants** | Run the same oracle on a stiff plant and a slow plant. Tests whether the constraint extraction generalizes. |

**Concrete plan:** Run Phase 1–2 on the existing `zeta=0.7` plant, then repeat on `zeta=0.1` (underdamped) as your second test scenario.

---

## 2. What "Stress-Test" Really Means — Scenarios & Metrics

### 2.1 Stress Scenarios (Oracle Personas)

| ID | Persona Name | Description | What it tests |
|----|--------------|-------------|---------------|
| S1 | **Monotone** | "I prefer whichever has lower MSE" every iteration | Baseline: does L2G converge without noise? |
| S2 | **Noisy** | 20% random flip in preference label | Robustness to human inconsistency |
| S3 | **Contradictory** | Prefers B but adds a constraint that B already violates | Does the system degrade gracefully or break? |
| S4 | **Drastic-Absolute** | "No overshoot" = hard threshold at 0% | Can the model find a valid candidate under tight constraints? |
| S5 | **Drastic-Relative** | "Make it better than A in everything" | Tests multi-constraint feasibility |
| S6 | **Direction-Only** | Never gives a preference label, only directions ("reduce overshoot") | Tests the `direction_only` feedback_type path end-to-end |
| S7 | **Late Switcher** | Sends consistent feedback for 3 iterations, then reverses all preferences | Tests recovery / catastrophic forgetting in the PairwiseGP |
| S8 | **Ambiguous-Satisfaction** | Gives positive feedback even when B is worse than A | Tests whether the system exploits false positives |

### 2.2 Quantitative Metrics (what to measure per run)

#### Primary
- **Simple Regret** at iteration T: `SR_T = f(x*) - f(x_best_T)` where `f` is the oracle's hidden utility function. This requires you to know the true optimum — run a dense grid evaluation once at the start.
- **Cumulative Constraint Violations**: number of iterations where the selected candidate fails `check_real_constraints`.
- **Convergence Iteration**: first T where `SR_T < epsilon` (e.g., 5% of initial regret).

#### Secondary
- **Preference Alignment Rate**: fraction of iterations where the optimizer's proposed B actually aligns with what the oracle would have preferred (before the oracle speaks). Measures how well the acquisition function anticipates preference.
- **Constraint Extraction Accuracy**: for oracle runs where the ground truth constraints are known, check whether the `L2GOutput` constraints match the oracle's declared targets (threshold delta, correct subfunction_id).
- **Feasibility Rate of Proposed Candidates**: how often the first `optimize_acqf` attempt passes `check_real_constraints` without retries.

#### Aggregation
Run each scenario with **5–10 seeds** and report mean ± std of each metric. This directly addresses the reproducibility challenge.

---

## 3. Model Modifications — Feedback on Your Ideas + New Ones

### 3.1 Acquisition Function Changes ✅ (Your Idea — Good)

**What to try:**

| Variant | Implementation | Hypothesis |
|---------|---------------|------------|
| `qUCB` with fixed β | Current baseline | — |
| `qUCB` with **annealed β** (high → low) | `beta = beta_max * (1 - t/T)` | More exploration early, exploitation late |
| `qUCB` with **feedback-driven β** | Increase β when oracle is contradictory (detected by preference reversal), decrease when consistent | Adapts to human reliability |
| `qEHVI` (Expected Hypervolume Improvement) | BoTorch's `qExpectedHypervolumeImprovement` over the 3 metric GPs | Treats this as a proper multi-objective problem; no need for a preference model in early iterations |
| `ThompsonSampling` | Sample from PairwiseGP posterior, pick argmax | Less aggressive, avoids over-committing to uncertain regions |
| `qKG` (Knowledge Gradient) | BoTorch's `qKnowledgeGradient` | One-step optimal; heavier compute but better in low-budget settings |

**Concretely testable:** compare Fixed-β UCB vs. Annealed-β UCB vs. qEHVI across Scenarios S1 and S4. This is publishable.

### 3.2 Constraint Function Computation ✅ (Your Idea — Good)

The current `build_nonlinear_inequality_constraints` uses the **surrogate mean** as the constraint function. This is optimistic and can lead to constraint violations in the real simulation. Two alternatives:

**Option A — Conservative (UCB-Constraint):** Instead of `limit - model(X)`, use `limit - (mean(X) + k*std(X))`. This adds a safety margin proportional to surrogate uncertainty.

```python
# In _create_constraint_func, pass the full posterior not just mean:
def constraint(X):
    posterior = model_gp.posterior(X_2d)
    mu = posterior.mean
    sigma = posterior.variance.sqrt()
    pessimistic_val = mu + k * sigma  # pessimistic estimate (assumes worst case)
    return limit - pessimistic_val
```

**Option B — Probabilistic Feasibility Filter:** Instead of a hard constraint, compute `P(model(X) <= limit)` from the GP posterior (a Gaussian CDF). Use it as a soft penalty in the acquisition function. This is closer to what SCBO (Scalable Constrained BO) does.

**Option C — Constraint Tightening Schedule:** Start with `limit * 1.2` (20% relaxed), tighten to `limit` by iteration 3. Helps the model find a feasible region before enforcing hard constraints.

### 3.3 New Ideas Worth Pursuing

**Idea A — Preference Confidence Decay:** The PairwiseGP currently weights all historical comparisons equally. But early comparisons may have been made when the human had less information. Implement an exponential decay: weight comparison at iteration `t` by `gamma^(T-t)` where `gamma < 1`. This lets the model "forget" old inconsistent preferences.

**Idea B — Abstention / "I don't know" Handling:** The oracle (and real humans) sometimes genuinely can't tell. Adding an abstention option (feedback_type="none") that contributes zero gradient to the preference model but still adds points to the metric surrogates is worth testing.

**Idea C — Active Query Selection:** Instead of always showing the best surrogate point as A, actively select A to maximally disambiguate the preference model (similar to BALD/information-gain query selection for active learning). This is a bigger contribution but fits your Phase 3.

---

## 4. Oracle Function Design

### Architecture

The oracle replaces the LLM + human. It has:
1. A **declared target profile** at initialization (the "hidden utility")
2. A **deterministic `query()` method** that takes metrics A and metrics B and returns a valid `L2GOutput` JSON
3. A **noise parameter** to simulate human inconsistency
4. A **persona** enum to select stress scenarios

### Oracle Utility Function

The oracle's internal utility is a weighted sum of normalized metrics:

```
U(x) = - w_ov * overshoot_pct/100 - w_ts * settling_time/T_max - w_mse * log(tracking_mse+1)
```

A prefers B when `U(B) > U(A)`. The weights + hard thresholds define the persona.

### The oracle code is in `oracle.py` (see companion file).

---

## 5. GitHub Repository Structure

```
L2G-stress-test/
│
├── README.md                    # Project overview + how to run
├── requirements.txt             
│
├── src/                         # Core source (your existing code, lightly refactored)
│   ├── __init__.py
│   ├── control_utils.py         # Plant simulation, metrics, plotting
│   ├── L2Gengine.py             # L2G engine (unchanged)
│   └── oracle.py                # NEW: deterministic pseudo-human oracle
│
├── experiments/
│   ├── configs/                 # YAML experiment configs (scenario, seeds, hyperparams)
│   │   ├── s1_monotone.yaml
│   │   ├── s4_drastic_absolute.yaml
│   │   └── ...
│   ├── run_experiment.py        # Entry point: loads config, runs N seeds, saves results
│   └── results/                 # Auto-created: JSON files with per-seed metrics
│       └── .gitkeep
│
├── analysis/
│   ├── plot_regret.py           # Convergence curves (SR vs iteration)
│   ├── plot_heatmap.py          # Constraint violation heatmap per scenario
│   ├── compare_acqf.py          # Side-by-side comparison of acquisition functions
│   └── notebooks/
│       └── exploration.ipynb    # Interactive exploration
│
├── tests/
│   ├── test_oracle.py           # Unit tests: oracle returns valid L2GOutput
│   ├── test_l2g_engine.py       # Unit tests: constraint building, preference update
│   └── test_control_utils.py    # Sanity checks on plant/metrics
│
└── docs/
    ├── METHODOLOGY.md           # Explanation of L2G, PBO, oracle design
    └── RESULTS.md               # Filled in Phase 3
```

### Branch Strategy

- `main` — stable, tagged versions only
- `dev` — working branch
- `exp/acqf-comparison` — acquisition function experiments
- `exp/constraint-variants` — constraint computation modifications

### Commit Convention

```
feat: add oracle S3 contradictory persona
fix: constraint tightening schedule off-by-one
exp: run S1 baseline, 5 seeds, results in experiments/results/s1/
```

---

## Phase Roadmap

### Phase 1 (now)
- [ ] Finalize oracle.py (see companion file)
- [ ] Write `run_experiment.py` with YAML configs for S1 and S4
- [ ] Compute ground-truth optimum grid for PI plant (for regret calculation)
- [ ] Set up GitHub repo with the structure above
- [ ] Unit tests for oracle and L2G engine

### Phase 2 (stress-testing)
- [ ] Run all 8 scenarios, 5 seeds each, fixed warm-start n=200 (faster than 1000)
- [ ] Compare fixed-β vs annealed-β UCB on S1 and S4
- [ ] Compare surrogate-mean vs UCB-constraint on S4 (drastic absolute)
- [ ] Document failure modes

### Phase 3 (baselines + presentation)
- [ ] Implement vanilla PBO baseline (PairwiseGP + EI, no L2G constraints)
- [ ] Implement Random Search baseline
- [ ] Plot convergence curves + box plots of final regret
- [ ] Write RESULTS.md
