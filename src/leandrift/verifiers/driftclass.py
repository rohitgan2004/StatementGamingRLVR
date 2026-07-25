"""DRIFTCLASS: a structural slot-comparison faithfulness detector (Section 4.6).

DRIFTCLASS is a *syntactic approximation* of Definition 1 (schematic semantic
equivalence).  It parses both statements into aligned slots -- binder types,
quantifier prefix, hypothesis multiset, side conditions, conclusion -- after
alpha-normalization / canonicalization, and compares them slot-by-slot.  Its
output is either ``faithful`` or one of the Table 1 drift classes.

Design guarantees (both error directions matter for delta_hack):
  * Sound direction: equal canonical slots => genuinely equivalent, so a
    ``faithful`` verdict is rarely a false negative on canonical drift.
  * Approximate direction: semantically-equal but structurally-restructured
    statements canonicalize to identical slots (commutative rewrites,
    hypothesis reordering), so faithful restructurings are not flagged -- but
    novel restructurings the canonicalizer misses would be false positives,
    which is why precision is *measured*, not assumed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from leandrift.core.prop import And, Atom, Const, Prop
from leandrift.core.statement import Statement
from leandrift.corpus.weakenings import (
    ADDED_HYP,
    DROPPED_CONSTRAINT,
    FAITHFUL,
    OTHER_DRIFT,
    QUANTIFIER_DRIFT,
    STRENGTHENED_PREMISE,
    WEAKENED_CONCL,
)


def _canon_key(p: Prop) -> str:
    return p.canonical().lean()


def _is_false(p: Prop) -> bool:
    return p.canonical() == Const(False)


def _contains_conjunct(p: Prop, target: Prop) -> bool:
    """True if `target` appears as a conjunct of `p` (after canonicalization)."""
    pc = p.canonical()
    tc = target.canonical()
    if pc == tc:
        return True
    if isinstance(pc, And):
        tset = set(tc.conjuncts) if isinstance(tc, And) else {tc}
        return tset.issubset(set(pc.conjuncts))
    return False


@dataclass
class DriftReport:
    drift_class: str
    faithful: bool
    detail: str = ""


def classify(s_hat: Statement, s_star: Statement) -> DriftReport:
    """Classify the drift (if any) of ``s_hat`` relative to intended ``s_star``."""
    # Alpha-normalize to the pinned binder names before slot comparison.
    s_hat = s_hat.alpha_normalize()
    s_star = s_star.alpha_normalize()
    # Slot 1: binder types.
    if s_hat.binder_types() != s_star.binder_types():
        return DriftReport(OTHER_DRIFT, False, "binder-type mismatch")

    # Slot 2: quantifier prefix (order matters for scope drift).
    qh, qs = s_hat.quantifier_signature(), s_star.quantifier_signature()
    if qh != qs:
        if sorted(qh) == sorted(qs):
            return DriftReport(QUANTIFIER_DRIFT, False, "quantifier scope reordered")
        return DriftReport(OTHER_DRIFT, False, "quantifier prefix mismatch")

    star_concl = s_star.conclusion.canonical()
    hat_concl = s_hat.conclusion.canonical()

    star_hyp_keys = Counter(_canon_key(h.prop) for h in s_star.hyps)
    hat_hyp_keys = Counter(_canon_key(h.prop) for h in s_hat.hyps)
    extra = hat_hyp_keys - star_hyp_keys
    missing = star_hyp_keys - hat_hyp_keys

    # Slot 3a: strengthened premise (a False / vacuous hypothesis added).
    for h in s_hat.hyps:
        if _is_false(h.prop) and _canon_key(h.prop) not in star_hyp_keys:
            return DriftReport(STRENGTHENED_PREMISE, False, "vacuous (False) premise added")

    # Slot 3b: added hypothesis == conclusion (either as a new hyp or folded in).
    star_conj_added = any(
        _contains_conjunct(h.prop, star_concl)
        and not any(_contains_conjunct(sh.prop, star_concl) for sh in s_star.hyps)
        for h in s_hat.hyps
    )
    if _canon_key(star_concl) in extra or star_conj_added:
        return DriftReport(ADDED_HYP, False, "conclusion added as hypothesis")

    # Slot 5: dropped side-condition constraint.
    star_side = {_canon_key(h.prop) for h in s_star.hyps if h.is_side_condition}
    if missing and (set(missing) & star_side):
        return DriftReport(DROPPED_CONSTRAINT, False, "side condition dropped")

    # Slot 4: weakened / altered conclusion.
    if hat_concl != star_concl:
        return DriftReport(WEAKENED_CONCL, False, "conclusion altered")

    # Any remaining hypothesis-set difference is unclassified structural drift.
    if extra or missing:
        return DriftReport(OTHER_DRIFT, False, "unclassified hypothesis-set change")

    return DriftReport(FAITHFUL, True, "slots match")


def faithful(s_hat: Statement, s_star: Statement) -> bool:
    return classify(s_hat, s_star).faithful
