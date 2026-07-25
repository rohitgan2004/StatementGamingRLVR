# Running LEANDRIFT on a cloud GPU

This guide takes you from an empty cloud box to the **real, non-placeholder**
numbers behind Figure 2 and Tables 4–5. Two supported paths:

- **Path A — Thinking Machines Tinker** (recommended). A managed API does the
  distributed LoRA training/sampling of Qwen2.5; your Lean reward loop stays on a
  cheap CPU driver box. No local GPU to babysit, and it maps 1:1 onto the paper's
  GRPO description.
- **Path B — single A10G via TRL/QLoRA**. Everything (training + Lean) on one
  24 GB GPU. This is the paper's stated single-GPU configuration.

Either way the key architectural point is the paper's threat model: **the Lean
kernel is never modified; only the harness's *statement channel* is
exploitable.** So the reward loop — parse → weak/strict verify against real Lean
→ per-arm reward — always runs on *your* machine, next to the warm REPL pool.

---

## 0. The compute budget

Per arm: ~400 GRPO steps × (8 prompts × 8 samples) ≈ 25k rollouts, each needing a
Lean check (memoized + pooled). Four arms (B/E/H/H′) × 3 seeds + SFT + calibration:

| Item | Tinker (Path A) | A10G (Path B) |
|---|---|---|
| Wall-clock / arm-seed | ~2–3 h | ~8–10 h |
| GPU | managed | 1× A10G (24 GB) |
| Driver box | 8 vCPU / 16 GB (Lean pool) | same box as GPU |
| Total | ~140 GPU-h ≈ **\$180** on-demand | ~120 GPU-h ≈ **\$150** spot |

Start with **1 seed × 150 steps** to sanity-check the gap opens before committing
to the full 3-seed × 400-step sweep.

---

## 1. One-time: build the pinned Lean 4 + Mathlib + REPL environment

Do this on the machine that will run the reward loop (the driver box for Path A,
the GPU box for Path B). Lean/Mathlib are pinned by `lean/lean-toolchain` and
`lean/lakefile.lean`.

```bash
# install elan (Lean version manager)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
source ~/.elan/env

# fetch Mathlib's prebuilt cache + the REPL, then build (cache get avoids ~1h compile)
make lean-build          # == cd lean && lake update && lake exe cache get && lake build

# smoke test: the REPL pool should verify a trivial theorem
python -c "from leandrift.utils.config import load_config; \
from leandrift.lean.backend import get_backend; \
cfg=load_config('configs/base.yaml'); cfg['lean']['backend']='repl'; \
b=get_backend(cfg); print(b.check_theorem('theorem t : 1 = 1 := by rfl','')); b.close()"
```

`lake exe cache get` pulls Mathlib's prebuilt `.olean`s so you don't compile
Mathlib from scratch. Budget ~15–30 min for this step, mostly download.

> To switch every verifier from the mock to real Lean, set `lean.backend: repl`
> (edit `configs/base.yaml` or pass `--config` overrides). The mock backend is for
> local development only and stamps every result `simulated=True`.

---

## 2. Path A — Thinking Machines Tinker (recommended)

### 2a. Provision + install

A small CPU box is enough for the driver (Tinker holds the GPUs):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[tinker]"
export TINKER_API_KEY=...          # from the Tinker console
```

### 2b. Calibrate the corpus against the real base model

Calibration selects theorems in the honest "difficulty band" and confirms each has
an exploitable weakening (drift advantage), so the gap we later measure is real and
not an artifact of trivial/impossible items.

```bash
make calibrate CONFIG=configs/base.yaml     # -> data/corpus/corpus.json
```

### 2c. Train each arm

```bash
make train-tinker ARM=B CORPUS=data/corpus/corpus.json
make train-tinker ARM=E
make train-tinker ARM=H
make train-tinker ARM=Hp
make sft                                     # SFT-on-genuine-proofs baseline
```

Each run streams `[tinker] step N mean_reward=... datums=...` and writes
`runs/arm_<ARM>_tinker_seed<seed>/metrics.jsonl` (weak/strict pass, Δhack,
faithfulness, taxonomy — logged every `eval_every` steps). For seeds, override the
config: `... --config configs/arm_E.yaml` then set `seed` per run, or loop:

```bash
for s in 0 1 2; do
  python -m leandrift.rl.train_tinker --config configs/arm_E.yaml \
    --corpus data/corpus/corpus.json --steps 400 \
    # set seed via a config override file or env; see configs/base.yaml
done
```

What the driver does each step (`src/leandrift/rl/train_tinker.py`): sample a group
of completions from Tinker → parse `FORMAL_STATEMENT`/`PROOF` → verify against the
**local** Lean pool → compute the arm's reward → send GRPO advantages
(`loss_fn="importance_sampling"`) back to Tinker's `forward_backward`/`optim_step`.
The strict verifier is logged every checkpoint but (in Arm E) contributes **zero**
reward — the model games a check it never sees.

---

## 3. Path B — single A10G via TRL/QLoRA

```bash
# on the GPU box (A10G / 24 GB), after step 1's Lean build
python -m venv .venv && source .venv/bin/activate
pip install -e ".[trl]"

make calibrate CONFIG=configs/base.yaml
make train-trl ARM=B      # TRL GRPOTrainer + QLoRA (4-bit) Qwen2.5-1.5B
make train-trl ARM=E
make train-trl ARM=H
make train-trl ARM=Hp
make sft
```

`train_trl.py` wraps TRL's `GRPOTrainer` with a custom reward function that calls
the same `rollout()` → Lean pool → per-arm reward path. QLoRA (4-bit base + LoRA
rank from the config) keeps Qwen2.5-1.5B + GRPO groups inside 24 GB. If you OOM,
lower `grpo.group_size` or `model.max_completion_tokens` in the arm config.

> Keep the REPL pool sized to your CPU cores (`lean.pool_workers`). Lean checking,
> not the GPU, is usually the throughput bottleneck; memoization on
> `(statement, proof)` hashes absorbs most of the duplicate rollouts.

---

## 4. Collect results → Figure 2 + Tables 4/5

Both paths write the **same** `runs/*/metrics.jsonl` schema, so one command turns
the logs into the paper's artifacts:

```bash
make figures         # leandrift figures --config configs/base.yaml
#   figures/figure2_dynamics.png   (weak vs strict vs Δhack over training, per arm)
#   outputs/table4_results.md      (final-checkpoint results, all conditions)
#   outputs/table5_transfer.md     (Arm E cross-family D→I transfer)
```

Averaging over seeds and confidence intervals is done by `eval/figures.py` from all
matching `runs/arm_*` directories. **These are the numbers that replace the paper's
placeholders.** (For real runs, drop the "(simulation)" caption that the local
`make reproduce` path emits.)

---

## 5. Cost-saving tips

- **Prove the mechanism cheaply first.** `make reproduce` (local, no GPU) confirms
  the whole pipeline and the expected arm ordering in seconds.
- **Short pilot before the sweep.** `--steps 150`, 1 seed, arms B and E only — you
  should already see E's Δhack climbing above ~0.3 while B stays flat.
- **Spot/preemptible GPUs** for Path B: checkpoints are written every
  `training.checkpoint_every` steps, so resume on eviction.
- **Right-size the Lean pool.** More REPL workers than CPU cores just thrashes;
  memoization already removes most duplicate checks.
- **Cache the Mathlib build** into your machine image so `lake exe cache get`
  only runs once across many launches.
