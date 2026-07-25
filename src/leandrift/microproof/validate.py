"""Validate DRIFTCLASS against the exact schematic-equivalence oracle (Appendix A).

Protocol: for each template we enumerate the intended statement, all single- and
double-move weakenings from Table 1, and several random faithful restructurings
(hypothesis reordering, commutative rewrites, bound renamings); then we score
DRIFTCLASS against the oracle over every candidate.

Caveat (reported by the paper): candidates are generated from the same move
families DRIFTCLASS detects, so perfect agreement here bounds detector error only
on *canonical* drift; the hand-labeled Lean estimates carry the weight for messy
real outputs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from leandrift.core.expr import BinOp, Term
from leandrift.core.prop import And, Atom, Or, Prop, Implies, Not, Quant
from leandrift.core.statement import Binder, Hypothesis, Statement
from leandrift.corpus.templates import Instance, enumerate_instances
from leandrift.corpus.weakenings import MOVES, WeakenResult
from leandrift.microproof.oracle import faithful_oracle
from leandrift.verifiers.driftclass import faithful as drift_faithful


# ---- faithful restructurings ---------------------------------------------------
def _swap_commutative_term(t: Term) -> Tuple[Term, bool]:
    if isinstance(t, BinOp):
        if t.op in ("+", "*"):
            return BinOp(t.op, t.right, t.left), True
        left, done = _swap_commutative_term(t.left)
        if done:
            return BinOp(t.op, left, t.right), True
        right, done = _swap_commutative_term(t.right)
        return BinOp(t.op, t.left, right), done
    return t, False


def _swap_commutative_prop(p: Prop) -> Tuple[Prop, bool]:
    if isinstance(p, Atom):
        left, done = _swap_commutative_term(p.left)
        if done:
            return Atom(p.rel, left, p.right), True
        right, done = _swap_commutative_term(p.right)
        return Atom(p.rel, p.left, right), done
    if isinstance(p, And):
        parts = list(p.conjuncts)
        for i, c in enumerate(parts):
            nc, done = _swap_commutative_prop(c)
            if done:
                parts[i] = nc
                return And(tuple(parts)), True
        return p, False
    return p, False


def faithful_restructurings(s: Statement, rng: random.Random, k: int = 5) -> List[Statement]:
    out: List[Statement] = []
    # 1. hypothesis reordering
    if len(s.hyps) >= 2:
        hyps = list(s.hyps)
        rng.shuffle(hyps)
        out.append(s.with_hyps(hyps))
    # 2. commutative rewrite in conclusion
    swapped, done = _swap_commutative_prop(s.conclusion)
    if done:
        out.append(s.with_conclusion(swapped))
    # 3. commutative rewrite in a hypothesis
    for i, h in enumerate(s.hyps):
        nh, done = _swap_commutative_prop(h.prop)
        if done:
            hyps = list(s.hyps)
            hyps[i] = Hypothesis(h.name, nh, h.is_side_condition)
            out.append(s.with_hyps(hyps))
            break
    # 4. bound renaming (rename first binder)
    if s.binders:
        mapping = {s.binders[0].name: "z"}
        renamed = Statement(
            s.name,
            (Binder("z", s.binders[0].type),) + s.binders[1:],
            tuple(Hypothesis(h.name, h.prop.subst(mapping), h.is_side_condition) for h in s.hyps),
            s.conclusion.subst(mapping),
        )
        out.append(renamed)
    return out[:k]


@dataclass
class ValidationReport:
    n_instances: int = 0
    n_candidates: int = 0
    agreement: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    disagreements: List[dict] = field(default_factory=list)


def validate(seed: int = 0, families: Optional[List[str]] = None) -> ValidationReport:
    rng = random.Random(seed)
    families = families or ["D", "I"]
    instances: List[Instance] = [
        i for i in enumerate_instances() if i.family in families
    ]
    rep = ValidationReport()
    for inst in instances:
        rep.n_instances += 1
        s_star = inst.statement
        candidates: List[Statement] = [s_star]

        singles: List[WeakenResult] = []
        for move in MOVES:
            r = move(s_star)
            if r is not None:
                singles.append(r)
                candidates.append(r.statement)
        # double-move compositions
        for r in singles:
            for move2 in MOVES:
                r2 = move2(r.statement)
                if r2 is not None:
                    candidates.append(r2.statement)
        # faithful restructurings
        candidates.extend(faithful_restructurings(s_star, rng))

        for cand in candidates:
            try:
                oracle_faithful = faithful_oracle(cand, s_star)
            except ValueError:
                continue  # too large to enumerate exactly
            det_faithful = drift_faithful(cand, s_star)
            rep.n_candidates += 1
            # "positive" = drift (not faithful)
            oracle_drift = not oracle_faithful
            det_drift = not det_faithful
            if oracle_drift and det_drift:
                rep.tp += 1
            elif oracle_drift and not det_drift:
                rep.fn += 1
                rep.disagreements.append(
                    {"kind": "missed-drift", "stmt": cand.to_lean(), "intended": s_star.to_lean()}
                )
            elif not oracle_drift and det_drift:
                rep.fp += 1
                rep.disagreements.append(
                    {"kind": "false-flag", "stmt": cand.to_lean(), "intended": s_star.to_lean()}
                )
            else:
                rep.tn += 1

    total = rep.tp + rep.fp + rep.tn + rep.fn
    rep.agreement = (rep.tp + rep.tn) / total if total else 1.0
    rep.precision = rep.tp / (rep.tp + rep.fp) if (rep.tp + rep.fp) else 1.0
    rep.recall = rep.tp / (rep.tp + rep.fn) if (rep.tp + rep.fn) else 1.0
    return rep
