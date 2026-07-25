"""Checkpoint evaluation of the simulation policy on the held-out set."""

from __future__ import annotations

import random
from typing import List

from leandrift.core.episode import Episode
from leandrift.corpus.generator import CorpusItem
from leandrift.eval.metrics import Metrics, compute
from leandrift.lean.backend import LeanBackend
from leandrift.rl.rollout import rollout
from leandrift.rl.sim_policy import SimPolicy


def evaluate_sim(
    policy: SimPolicy,
    eval_items: List[CorpusItem],
    backend: LeanBackend,
    arm: dict,
    shaping: dict,
    step: int,
    n_samples: int = 1,
    related_threshold: float = 0.6,
) -> tuple[Metrics, List[Episode]]:
    episodes: List[Episode] = []
    # Dedicated, deterministic eval RNG keyed by (seed, step): reproducible and
    # independent of the training RNG stream.
    eval_rng = random.Random((policy.seed + 1) * 1_000_003 + step)
    for item in eval_items:
        for action, completion in policy.sample_group(item, step, n_samples, rng=eval_rng):
            ep = rollout(item.prompt, completion, backend, arm, shaping,
                         related_threshold=related_threshold, step=step, with_reward=False)
            episodes.append(ep)
    return compute(episodes), episodes
