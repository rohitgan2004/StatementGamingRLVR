"""Episode and verdict records for the training / audit loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from leandrift.core.statement import Statement


@dataclass
class Prompt:
    """A natural-language theorem prompt plus its (hidden) intended statement."""

    id: str
    family: str            # "D" or "I"
    informal: str          # natural-language statement shown to the policy
    intended: Statement    # locked s_star(x), hidden from the policy
    template: str          # template id it was generated from

    @property
    def pinned_header(self) -> str:
        return self.intended.pinned_header()


@dataclass
class Completion:
    """What the policy emits: a formal statement s_hat and a tactic proof pi."""

    statement: Optional[Statement]   # None if unparseable
    proof: str
    raw_text: str = ""
    parsed_ok: bool = True


@dataclass
class Verdict:
    """A single verifier's decision plus its factored components."""

    passed: bool
    kernel_ok: bool = False        # V(s_hat, pi): proof closes
    hygiene_ok: bool = False       # no sorry / no new axioms / safe imports
    related_ok: bool = False       # relatedness filter (weak verifier)
    faithful: bool = False         # DRIFTCLASS-Faithful (strict verifier)
    detail: str = ""


@dataclass
class Episode:
    """A full rollout: prompt, completion, and both verifier verdicts."""

    prompt: Prompt
    completion: Completion
    weak: Verdict
    strict: Verdict
    drift_class: str = "faithful"   # DRIFTCLASS output / taxonomy label
    reward: float = 0.0
    step: int = 0
