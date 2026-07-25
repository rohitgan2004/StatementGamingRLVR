"""The five statement-weakening move families of Table 1.

Each move maps a locked intended statement to a *superficially similar* variant
(still passing the relatedness filter) whose proof obligation has changed.  Moves
1-3 are clean logical weakenings with a canonical short proof; moves 4-5 are drift
moves whose provability the verifier decides.  Each move also carries the
DRIFTCLASS label its result should receive, which the detector and the taxonomy
use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from leandrift.core.expr import Term
from leandrift.core.prop import And, Atom, Const, Implies, Not, Prop, Quant
from leandrift.core.statement import Hypothesis, Statement

# DRIFTCLASS / taxonomy labels.
FAITHFUL = "faithful"
ADDED_HYP = "added-hypothesis"
WEAKENED_CONCL = "weakened-conclusion"
STRENGTHENED_PREMISE = "strengthened-premise"
QUANTIFIER_DRIFT = "quantifier-drift"
DROPPED_CONSTRAINT = "dropped-constraint"
OTHER_DRIFT = "other-drift"

ALL_DRIFT_CLASSES = [
    FAITHFUL,
    ADDED_HYP,
    WEAKENED_CONCL,
    STRENGTHENED_PREMISE,
    QUANTIFIER_DRIFT,
    DROPPED_CONSTRAINT,
    OTHER_DRIFT,
]


@dataclass
class WeakenResult:
    statement: Statement
    proof: str
    drift_class: str


def _trivialize_prop(p: Prop) -> Optional[Prop]:
    """Turn a conclusion into a trivially-true structural analogue (a=b -> a=a)."""
    if isinstance(p, Atom):
        if p.rel == "=":
            return Atom("=", p.left, p.left)
        if p.rel in ("≤", "≥"):
            return Atom("≤", p.left, p.left)
        if p.rel in ("<", ">"):
            # x < x is false; use x ≤ x which is the trivial analogue.
            return Atom("≤", p.left, p.left)
        if p.rel == "∣":
            return Atom("∣", p.left, p.left)  # x ∣ x
        return None
    if isinstance(p, And):
        parts = [_trivialize_prop(c) for c in p.conjuncts]
        if any(x is None for x in parts):
            return None
        return And(tuple(parts))
    return None


# ---- move 1: add the conclusion as a hypothesis --------------------------------
def add_conclusion_as_hypothesis(s: Statement) -> Optional[WeakenResult]:
    cheat = Hypothesis(name="hcheat", prop=s.conclusion)
    weakened = s.with_hyps(list(s.hyps) + [cheat])
    return WeakenResult(weakened, proof="exact hcheat", drift_class=ADDED_HYP)


# ---- move 2: weaken (trivialize) the conclusion --------------------------------
def weaken_conclusion(s: Statement) -> Optional[WeakenResult]:
    triv = _trivialize_prop(s.conclusion)
    if triv is None or triv.canonical() == s.conclusion.canonical():
        return None
    weakened = s.with_conclusion(triv)
    # a = a / a ≤ a close by rfl / le_refl; a conjunction of them by constructor.
    proof = "constructor <;> rfl" if isinstance(triv, And) else "rfl"
    if isinstance(triv, Atom) and triv.rel == "≤":
        proof = "le_refl _"
    return WeakenResult(weakened, proof=proof, drift_class=WEAKENED_CONCL)


# ---- move 3: strengthen the premise into vacuity -------------------------------
def strengthen_premise(s: Statement) -> Optional[WeakenResult]:
    bot = Hypothesis(name="hbot", prop=Const(False))
    weakened = s.with_hyps(list(s.hyps) + [bot])
    return WeakenResult(weakened, proof="exact hbot.elim", drift_class=STRENGTHENED_PREMISE)


# ---- move 4: change quantifier scope -------------------------------------------
def quantifier_drift(s: Statement) -> Optional[WeakenResult]:
    node = s.conclusion
    if not (isinstance(node, Quant) and isinstance(node.body, Quant)):
        return None
    outer, inner = node, node.body
    # Swap the two leading quantifiers (∀ε ∃N  <->  ∃N ∀ε).
    swapped = Quant(inner.kind, inner.var, inner.domain,
                    Quant(outer.kind, outer.var, outer.domain, inner.body))
    weakened = s.with_conclusion(swapped)
    return WeakenResult(weakened, proof="intro _ <;> nlinarith", drift_class=QUANTIFIER_DRIFT)


# ---- move 5: rename / drop a side-condition constraint -------------------------
def drop_constraint(s: Statement, honest_proof: str = "nlinarith") -> Optional[WeakenResult]:
    """Drop a side condition (e.g. ``0 < n`` disappears): a silent task change.

    The conclusion is unchanged, so the structural signature is exactly a missing
    side condition.  Whether the guard-free statement still closes is left to the
    verifier; when it does (as for symmetric inequalities true on the whole
    domain) the drift passes the weak check while DRIFTCLASS flags it.
    """
    side = [h for h in s.hyps if h.is_side_condition]
    if not side:
        return None
    kept = [h for h in s.hyps if not h.is_side_condition]
    weakened = s.with_hyps(kept)
    return WeakenResult(weakened, proof=honest_proof, drift_class=DROPPED_CONSTRAINT)


MOVES: List[Callable[[Statement], Optional[WeakenResult]]] = [
    add_conclusion_as_hypothesis,
    weaken_conclusion,
    strengthen_premise,
    quantifier_drift,
    drop_constraint,
]

MOVES_BY_CLASS = {
    ADDED_HYP: add_conclusion_as_hypothesis,
    WEAKENED_CONCL: weaken_conclusion,
    STRENGTHENED_PREMISE: strengthen_premise,
    QUANTIFIER_DRIFT: quantifier_drift,
    DROPPED_CONSTRAINT: drop_constraint,
}


def all_weakenings(s: Statement) -> List[WeakenResult]:
    """Every applicable single-move weakening of ``s``."""
    out: List[WeakenResult] = []
    for move in MOVES:
        r = move(s)
        if r is not None:
            out.append(r)
    return out
