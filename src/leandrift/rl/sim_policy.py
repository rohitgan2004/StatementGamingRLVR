"""A tabular, reward-driven simulation policy.

This is NOT the LLM policy used for real runs (that is Qwen2.5 via Tinker/TRL).
It is a mechanistic stand-in that lets the *entire* pipeline -- verifiers, per-arm
reward, GRPO-style update, metrics, taxonomy, figures -- run on a laptop with no
GPU and no Lean, so the environment can be validated before spending GPU-hours.

Faithfulness of the mechanism: the policy keeps a single *global* preference over
strategy types {honest} U {Table 1 weakening classes} -- NOT a per-prompt table
-- and updates it with the SAME per-arm reward the LLM would receive, via a
GRPO-style within-group advantage.  Because the preference is over strategies
rather than specific theorems, what is learned on family D transfers to held-out
prompts and to the never-trained family I, exactly the generalization the paper
reports (Section 6.4).  The gaming dynamics *emerge* from the reward structure --
reinforced under Arm E, gated out under Arm H, redirected under Arm H' -- rather
than being scripted.  A separate, arm-independent "proving skill" term makes
honest proofs succeed more often over training, modeling capability acquisition.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from leandrift.core.episode import Completion, Prompt
from leandrift.corpus.generator import CorpusItem
from leandrift.corpus.weakenings import all_weakenings

HONEST = "honest"
# A plausible-but-failing honest first attempt (rejected by the mock backend).
_FAILING_PROOF = "intro h; simp [h]"


@dataclass
class Action:
    name: str            # "honest" or a drift class
    statement: object    # Statement to emit
    proof_ok: str        # proof used when the attempt "succeeds"
    drift_class: str     # DRIFTCLASS label this action should receive


@dataclass
class SimPolicy:
    total_steps: int
    channel_open: bool = True
    temperature: float = 1.0
    pref_lr: float = 0.3
    kl_coef: float = 0.07          # pull toward the reference policy (prevents collapse)
    skill_ceiling: float = 0.55
    fixed_skill: float = -1.0      # >=0 overrides the step-dependent proving skill
    honest_only_pref: float = -4.0  # initial (reference) preference for gaming actions
    seed: int = 0
    # GLOBAL preference over strategy types (shared across all prompts).
    prefs: Dict[str, float] = field(default_factory=dict)
    _rng: random.Random = field(default_factory=lambda: random.Random(0))

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # ---- action set -----------------------------------------------------------
    def _actions(self, item: CorpusItem) -> List[Action]:
        s_star = item.prompt.intended
        acts = [Action(HONEST, s_star, item.honest_proof, "faithful")]
        if self.channel_open:
            for w in all_weakenings(s_star):
                acts.append(Action(w.drift_class, w.statement, w.proof, w.drift_class))
        return acts

    def _ref(self, name: str) -> float:
        # Reference (base) policy: prefers honesty (no gaming under prompting).
        return 0.0 if name == HONEST else self.honest_only_pref

    def _pref(self, name: str) -> float:
        if name not in self.prefs:
            self.prefs[name] = self._ref(name)
        return self.prefs[name]

    def _skill(self, item: CorpusItem, step: int) -> float:
        if self.fixed_skill >= 0.0:
            return self.fixed_skill
        r0 = item.honest_base_rate
        progress = step / max(1, self.total_steps)
        return min(self.skill_ceiling, r0 + (self.skill_ceiling - r0) * progress)

    # ---- sampling -------------------------------------------------------------
    def sample_group(
        self,
        item: CorpusItem,
        step: int,
        group_size: int,
        rng: "random.Random | None" = None,
    ) -> List[Tuple[Action, Completion]]:
        # Evaluation passes its own rng so that measuring the policy never perturbs
        # the training RNG stream (otherwise the eval-sample count would silently
        # change the training trajectory and break reproducibility).
        rng = rng if rng is not None else self._rng
        acts = self._actions(item)
        logits = [self._pref(a.name) / self.temperature for a in acts]
        m = max(logits)
        exps = [math.exp(z - m) for z in logits]
        total = sum(exps)
        probs = [e / total for e in exps]

        skill = self._skill(item, step)
        out: List[Tuple[Action, Completion]] = []
        for _ in range(group_size):
            a = rng.choices(acts, weights=probs, k=1)[0]
            if a.name == HONEST:
                # Honest attempt succeeds with current proving skill.
                if rng.random() < skill:
                    proof = a.proof_ok
                else:
                    proof = _FAILING_PROOF
            else:
                proof = a.proof_ok  # canonical cheap proof for the weakening
            comp = Completion(statement=a.statement, proof=proof, parsed_ok=True,
                              raw_text=f"{a.statement.to_lean()} := by {proof}")
            out.append((a, comp))
        return out

    # ---- GRPO-style tabular update -------------------------------------------
    def update(self, item: CorpusItem, actions: List[Action], rewards: List[float]) -> None:
        if not rewards:
            return
        baseline = sum(rewards) / len(rewards)
        std = (sum((r - baseline) ** 2 for r in rewards) / len(rewards)) ** 0.5 or 1.0
        # Policy-gradient step on sampled actions.
        for a, r in zip(actions, rewards):
            adv = (r - baseline) / std
            self.prefs[a.name] = self._pref(a.name) + self.pref_lr * adv
        # KL penalty toward the reference policy (applied to every known strategy),
        # mirroring GRPO's KL term: it caps how far the policy drifts and yields a
        # stochastic equilibrium instead of full collapse.
        for name in list(self.prefs):
            self.prefs[name] -= self.kl_coef * (self.prefs[name] - self._ref(name))
