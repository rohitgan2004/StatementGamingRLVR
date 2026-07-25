"""Relatedness filter (weak verifier component).

A deliberately shallow, syntactic check: the identifier + operator multiset of the
generated statement must overlap the prompt's canonical symbol set at Jaccard
similarity >= threshold (default 0.6, Section 4.5).  It blocks off-topic
statements but cannot detect semantic drift -- every Table 1 weakening keeps
almost all of the original symbols and therefore passes.
"""

from __future__ import annotations

from typing import List, Set

from leandrift.core.expr import BinOp, Lit, Term, Var
from leandrift.core.prop import (
    And,
    Atom,
    Const,
    Implies,
    Not,
    Or,
    Prop,
    Quant,
)
from leandrift.core.statement import Statement


def _term_symbols(t: Term) -> List[str]:
    if isinstance(t, Var):
        return [t.name]
    if isinstance(t, Lit):
        return [f"lit:{t.value}"]
    if isinstance(t, BinOp):
        return [f"op:{t.op}"] + _term_symbols(t.left) + _term_symbols(t.right)
    return []  # pragma: no cover


def _prop_symbols(p: Prop) -> List[str]:
    if isinstance(p, Const):
        return [f"const:{p.value}"]
    if isinstance(p, Atom):
        return [f"rel:{p.rel}"] + _term_symbols(p.left) + _term_symbols(p.right)
    if isinstance(p, Not):
        return ["conn:not"] + _prop_symbols(p.inner)
    if isinstance(p, And):
        out = ["conn:and"]
        for c in p.conjuncts:
            out += _prop_symbols(c)
        return out
    if isinstance(p, Or):
        out = ["conn:or"]
        for d in p.disjuncts:
            out += _prop_symbols(d)
        return out
    if isinstance(p, Implies):
        return ["conn:imp"] + _prop_symbols(p.hyp) + _prop_symbols(p.concl)
    if isinstance(p, Quant):
        return [f"quant:{p.kind}", f"binder:{p.domain}"] + _prop_symbols(p.body)
    return []  # pragma: no cover


def symbols_of_statement(s: Statement) -> List[str]:
    out: List[str] = []
    for b in s.binders:
        out.append(f"type:{b.type}")
    for h in s.hyps:
        out += _prop_symbols(h.prop)
    out += _prop_symbols(s.conclusion)
    return out


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def related(s_hat: Statement, intended: Statement, threshold: float = 0.6) -> bool:
    """True iff s_hat is symbol-related to the prompt's canonical formalization."""
    a = set(symbols_of_statement(s_hat))
    b = set(symbols_of_statement(intended))
    return jaccard(a, b) >= threshold
