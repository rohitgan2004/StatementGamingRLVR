"""Simulation GRPO loop (local, no GPU / no Lean).

Drives the tabular SimPolicy with the real per-arm reward and verifiers, logging
the same metrics a real run logs (weak/strict pass, delta_hack, faithfulness,
taxonomy, per-family transfer).  Use this to validate the environment end-to-end
before launching the real Qwen2.5 runs on the cloud.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List

from leandrift.corpus.generator import Corpus, load_corpus
from leandrift.eval.evaluate import evaluate_sim
from leandrift.lean.backend import get_backend
from leandrift.rl.reward import compute_reward
from leandrift.rl.rollout import rollout
from leandrift.rl.sim_policy import SimPolicy
from leandrift.utils.io import append_jsonl, ensure_dir, write_json


def run_sim(cfg: dict, corpus: Corpus) -> Dict:
    arm = cfg["arm"]
    grpo = cfg["grpo"]
    shaping = cfg["reward_shaping"]
    tcfg = cfg["training"]
    seed = cfg.get("seed", 0)
    related_threshold = cfg["verifier"]["related_jaccard_threshold"]

    backend = get_backend(cfg)
    rng = random.Random(seed)

    steps = tcfg["steps"]
    group_size = grpo["group_size"]
    prompts_per_step = grpo["prompts_per_step"]
    eval_every = tcfg["eval_every"]
    n_eval_samples = cfg["eval"].get("n_eval_samples", 1)

    scfg = cfg.get("sim", {})
    policy = SimPolicy(
        total_steps=steps,
        channel_open=(arm.get("channel", "open") == "open"),
        seed=seed,
        pref_lr=scfg.get("pref_lr", 0.3),
        kl_coef=scfg.get("kl_coef", 0.07),
        honest_only_pref=scfg.get("honest_only_pref", -4.0),
        skill_ceiling=scfg.get("skill_ceiling", 0.55),
        temperature=scfg.get("temperature", 1.0),
    )

    run_dir = ensure_dir(os.path.join(cfg["paths"]["runs_dir"], f"arm_{arm['name']}_seed{seed}"))
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    train_items = corpus.train
    history: List[dict] = []

    for step in range(steps + 1):
        # ---- evaluation checkpoint ----
        if step % eval_every == 0 or step == steps:
            m, _ = evaluate_sim(policy, corpus.eval, backend, arm, shaping, step,
                                n_samples=max(n_eval_samples, 8),
                                related_threshold=related_threshold)
            rec = {"step": step, **m.to_dict()}
            history.append(rec)
            append_jsonl(metrics_path, rec)

        if step == steps:
            break

        # ---- one GRPO step ----
        batch = [train_items[rng.randrange(len(train_items))] for _ in range(prompts_per_step)]
        for item in batch:
            samples = policy.sample_group(item, step, group_size)
            actions, rewards = [], []
            for action, completion in samples:
                ep = rollout(item.prompt, completion, backend, arm, shaping,
                             related_threshold=related_threshold, step=step, with_reward=True)
                actions.append(action)
                rewards.append(ep.reward)
            policy.update(item, actions, rewards)

    summary = {
        "arm": arm["name"],
        "seed": seed,
        "simulated": True,
        "final": history[-1] if history else {},
        "steps": steps,
    }
    write_json(os.path.join(run_dir, "summary.json"), summary)
    return {"history": history, "summary": summary, "run_dir": run_dir}
