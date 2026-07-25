"""Causality baselines: pre-RL base model and SFT-on-genuine-proofs (Section 5.2).

Both are evaluated on the identical protocol.  Neither should exhibit a hacking
gap: the pre-RL policy prefers honesty (no gaming under prompting, cf. Kim et
al.), and SFT on genuine proofs raises proving skill without inducing gaming.
This isolates RLVR optimization pressure (Arm E) as the cause of the gap.
"""

from __future__ import annotations

import os
from typing import Dict

from leandrift.corpus.generator import Corpus
from leandrift.eval.evaluate import evaluate_sim
from leandrift.lean.backend import get_backend
from leandrift.rl.sim_policy import SimPolicy
from leandrift.utils.io import append_jsonl, ensure_dir, write_json


def _eval_policy(policy: SimPolicy, cfg: dict, corpus: Corpus, backend, tag: str) -> Dict:
    arm = {"name": "B", "channel": "open"}  # scoring only; reward not used
    m, _ = evaluate_sim(policy, corpus.eval, backend, arm, cfg["reward_shaping"], step=0,
                        n_samples=max(cfg["eval"].get("n_eval_samples", 1), 4),
                        related_threshold=cfg["verifier"]["related_jaccard_threshold"])
    run_dir = ensure_dir(os.path.join(cfg["paths"]["runs_dir"], f"baseline_{tag}"))
    rec = {"step": 0, **m.to_dict()}
    write_json(os.path.join(run_dir, "summary.json"),
               {"baseline": tag, "simulated": True, "final": rec})
    append_jsonl(os.path.join(run_dir, "metrics.jsonl"), rec)
    return rec


def run_baselines(cfg: dict, corpus: Corpus) -> Dict:
    seed = cfg.get("seed", 0)
    backend = get_backend(cfg)
    steps = cfg["training"]["steps"]

    scfg = cfg.get("sim", {})
    ref = scfg.get("honest_only_pref", -4.0)

    # Pre-RL base: untrained, honest-preferring, proving skill at base rate.
    pre = SimPolicy(total_steps=steps, channel_open=True, seed=seed, honest_only_pref=ref)
    pre_rec = _eval_policy(pre, cfg, corpus, backend, "preRL")

    # SFT on genuine proofs: honest, higher proving skill, no gaming propensity.
    sft = SimPolicy(total_steps=steps, channel_open=True, seed=seed,
                    fixed_skill=0.42, honest_only_pref=ref - 2.0)
    sft_rec = _eval_policy(sft, cfg, corpus, backend, "SFT")

    return {"preRL": pre_rec, "SFT": sft_rec}
