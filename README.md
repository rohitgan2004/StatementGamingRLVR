# LEANDRIFT

**Proving the Wrong Theorem: RLVR Induces Statement Gaming in Lean 4 Autoformalization Pipelines**

Reference implementation for the paper. LEANDRIFT is a Lean 4 / Mathlib training
environment that measures **statement gaming**: the failure mode in which
reinforcement learning with verifiable rewards (RLVR) teaches a prover to *weaken
the theorem it formalizes* until an easy proof passes, rather than to prove the
intended theorem — even though the proof checker is formally sound throughout.

The headline metric is the **hacking gap**

```
Δhack = weak_pass_rate(ŝ, π)  −  strict_pass_rate(ŝ, π; s⋆)
        └ what training optimizes ┘   └ what we actually wanted ┘
```

which quantifies, at every RL step, how much of the model's apparent competence is
illusory.

---

## What this repository gives you

| Paper artifact | Where |
|---|---|
| Autoformalize-then-prove environment, pinned-header protocol | `src/leandrift/core/`, `src/leandrift/rl/prompt.py` |
| Difficulty-calibrated corpus (families **D** divisibility / **I** inequalities) | `src/leandrift/corpus/` |
| Five statement-weakening move families (Table 1) | `src/leandrift/corpus/weakenings.py` |
| Weak verifier (in reward path) + strict audit verifier | `src/leandrift/verifiers/verify.py` |
| **DRIFTCLASS** structural faithfulness detector (Section 4.6) | `src/leandrift/verifiers/driftclass.py` |
| **MICROPROOF** exact schematic-equivalence oracle + validation (Appendix A) | `src/leandrift/microproof/` |
| Warm pool of Lean 4 REPL workers + memoization (Section 4.7) | `src/leandrift/lean/` |
| Per-arm GRPO reward shaping B / E / H / H′ (Table 2, Appendix B) | `src/leandrift/rl/reward.py` |
| GRPO training: **Tinker**, **TRL/QLoRA**, and a local simulator | `src/leandrift/rl/train_*.py` |
| Baselines: pre-RL, SFT-on-genuine-proofs (Section 5.2) | `src/leandrift/rl/baselines.py`, `sft.py` |
| Metrics (Δhack), taxonomy, cross-family transfer | `src/leandrift/eval/` |
| `make reproduce` → Figure 2, Tables 4 & 5 | `src/leandrift/eval/figures.py`, `Makefile` |

### Two ways to run it

1. **Local simulation (no GPU, no Lean).** A tabular, reward-driven policy plus a
   semantic mock Lean backend exercise the *entire* pipeline — verifiers, per-arm
   reward, GRPO-style update with a KL term, metrics, taxonomy, figures — on a
   laptop in seconds. The gaming dynamics **emerge from the real reward
   structure** (reinforced under E, gated out under H, redirected under H′); they
   are **not** scripted. Use this to validate the environment before spending
   GPU-hours. Outputs are clearly marked `(simulation)`.

2. **Real runs (cloud GPU + Lean).** Train Qwen2.5-{Coder,Math}-1.5B with GRPO
   against the unmodified Lean kernel over Mathlib, on the Thinking Machines
   **Tinker** API or a single **A10G** via TRL/QLoRA. See
   [`docs/CLOUD.md`](docs/CLOUD.md). **These runs produce the real numbers that
   replace the paper's placeholders.**

> ⚠️ The tables/figures produced by `make reproduce` are a **simulation** of the
> pipeline for validation. They reproduce the paper's *structure* (which arms open
> a gap, transfer, mitigation ordering), not measured LLM results. Fill in the
> paper's numbers by running the real pipeline in `docs/CLOUD.md`.

---

## Quickstart (local, no GPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # env / reward-loop deps only (no torch)

make reproduce              # DRIFTCLASS validation + all arms + baselines + figures
#   -> figures/figure2_dynamics.png
#   -> outputs/table4_results.md, outputs/table5_transfer.md

make test                   # unit tests (oracle, DRIFTCLASS, verifiers, reward, sim)
```

Individual steps:

```bash
leandrift gen-corpus                              # data/corpus/corpus.json
leandrift validate-driftclass                     # DRIFTCLASS vs exact oracle (Appendix A)
leandrift run-sim --config configs/arm_E.yaml     # one arm, local simulation
leandrift figures                                 # regenerate figures/tables
```

Representative simulation output (mechanism check — *not* measured LLM results):

| Condition | Δhack (sim) | Δhack (paper) |
|---|---|---|
| Arm B (locked statement) | ~0.00 | 0.004 |
| Arm E (exploitable) | ~0.57 | 0.593 |
| Arm H (faithfulness-gated) | ~0.02 | 0.019 |
| Arm H′ (pattern penalty) | ~0.28 | 0.339 |

The gap transfers from the trained family D to the never-trained family I, and
neither the pre-RL base nor SFT exhibits it — the same qualitative structure the
paper reports.

---

## The experimental arms (Table 2)

| Arm | Statement channel | Training task reward | Config |
|---|---|---|---|
| **B** Baseline | closed (`ŝ ≡ s⋆`) | `V(s⋆, π)·Hyg` | `configs/arm_B.yaml` |
| **E** Exploitable | open (model emits `ŝ`) | `V_weak = V(ŝ,π)·Hyg·Related` | `configs/arm_E.yaml` |
| **H** Hardened | open, faithfulness-gated | `V_weak · DRIFTCLASS-Faithful` | `configs/arm_H.yaml` |
| **H′** Ablation | open, pattern-penalized | `V_weak − 0.2·ExploitPat` | `configs/arm_Hp.yaml` |

All arms share the corpus, base model, GRPO configuration, and format-shaping
terms; only the task reward differs. In Arm E the strict verifier is a **pure
audit**: it is logged every checkpoint but contributes zero reward, so the model
games a check it has never seen.

---

## How to get the *real* (non-placeholder) numbers

```bash
# 0. one-time: build the pinned Lean 4 + Mathlib + REPL environment (on the GPU box)
make lean-build

# 1. calibrate the corpus against the real base model (selects the honest band)
make setup-trl          # or: make setup-tinker
make calibrate CONFIG=configs/base.yaml

# 2. switch the verifier to real Lean (configs: lean.backend: repl) and train each arm
make train-tinker ARM=B        # or: make train-trl ARM=B
make train-tinker ARM=E
make train-tinker ARM=H
make train-tinker ARM=Hp
make sft                        # SFT-on-genuine-proofs baseline

# 3. regenerate Figure 2 + Tables 4/5 from the logged runs
make figures
```

Every driver logs `runs/<arm>/metrics.jsonl` in one schema, so `leandrift figures`
produces the same Figure 2 / Tables 4–5 regardless of which backend generated the
data. Total compute is ≈ 140 GPU-hours (≈ \$180 on-demand); see `docs/CLOUD.md`.

---

## Design notes

- **Why schematic (not extensional) equivalence.** Extensional equivalence over
  the standard model is degenerate on true theorems: `P→Q` and `P∧Q→Q` are both
  identically true, so it would certify vacuous weakenings as faithful. The oracle
  (`microproof/oracle.py`) abstracts each atomic comparison and quantified
  subformula to a propositional variable (ground atoms pre-evaluated) and checks
  equivalence over all Boolean assignments — keeping proof obligations distinct.
- **DRIFTCLASS is honest about being approximate.** It is a *structural*
  slot-comparison detector (a syntactic approximation of the oracle), because full
  Lean offers no equivalence oracle and deployed pipelines face the same
  constraint. Its error is *measured*, not assumed: `make validate` scores it
  against the exact oracle (Appendix A), and both error directions are tracked
  because they bias Δhack in opposite ways.
- **Hygiene is enforced everywhere.** `sorry`/`admit`, new axioms, and
  model-supplied imports are rejected by both verifiers, isolating the statement
  channel as the only planted vulnerability.
- **The mock backend is a simulation, not Lean.** It decides acceptance by tactic
  adequacy + truth (sampled) for the template fragments; every `LeanResult` it
  returns carries `simulated=True`. Real runs use `lean.backend: repl`.

---

## Repository layout

```
configs/                 base + per-arm YAML (extends base.yaml)
lean/                    pinned Lean 4 project (Mathlib + REPL) for real runs
src/leandrift/
  core/                  Prop/Term AST, Statement, Episode, codec, parser
  corpus/                templates (D, I), weakenings (Table 1), generator, calibrate
  verifiers/             related filter, hygiene, DRIFTCLASS, weak/strict verify
  microproof/            exact oracle + DRIFTCLASS validation harness
  lean/                  backend interface, REPL worker + warm pool, semantic mock
  rl/                    reward, rollout, sim policy, train_sim/train_tinker/train_trl, sft, baselines
  eval/                  metrics (Δhack), taxonomy, evaluate, figures
tests/                   unit + end-to-end simulation tests
docs/CLOUD.md            cloud GPU guide (Tinker + A10G/TRL)
```

## Citation

If you use this code, please cite the paper (Rajagopalan, *Proving the Wrong
Theorem*). Code, corpus, and training logs are released under the MIT license.
