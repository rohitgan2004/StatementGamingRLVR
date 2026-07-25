"""Shared rollout: turn a completion into a verified, rewarded Episode.

Used by both the simulation loop and the real (Tinker / TRL) drivers so that
verification + reward are computed identically regardless of where the completion
came from.
"""

from __future__ import annotations

from typing import Optional

from leandrift.core.episode import Completion, Episode, Prompt
from leandrift.lean.backend import LeanBackend
from leandrift.rl.reward import RewardBreakdown, compute_reward
from leandrift.verifiers.verify import verify


def rollout(
    prompt: Prompt,
    completion: Completion,
    backend: LeanBackend,
    arm: dict,
    shaping: dict,
    related_threshold: float = 0.6,
    step: int = 0,
    with_reward: bool = True,
) -> Episode:
    # Arm B closes the statement channel: the model does not control s_hat.
    if arm.get("channel") == "closed" and completion.statement is not None:
        completion = Completion(
            statement=prompt.intended,
            proof=completion.proof,
            raw_text=completion.raw_text,
            parsed_ok=completion.parsed_ok,
        )
    weak, strict, drift_class = verify(prompt, completion, backend, related_threshold)
    ep = Episode(
        prompt=prompt,
        completion=completion,
        weak=weak,
        strict=strict,
        drift_class=drift_class,
        step=step,
    )
    if with_reward:
        br: RewardBreakdown = compute_reward(arm, weak, strict, drift_class, completion, shaping)
        ep.reward = br.reward
    return ep
