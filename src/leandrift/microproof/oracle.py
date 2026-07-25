"""Exact schematic-equivalence oracle (Definition 1).

Faithfulness is decided by abstracting each canonicalized atomic comparison and
each quantified subformula to a propositional variable, pre-evaluating ground
atoms, and checking logical equivalence of the two proof obligations over *all*
Boolean assignments to the union of their atoms.

Why not extensional (standard-model) equivalence?  It is degenerate on true
theorems: the closure of a true theorem and the closure of any vacuous weakening
are both identically true, so extensional comparison certifies exactly the drift
we want to detect as ``equivalent`` (Section 3).  Schematic equivalence keeps
``P → Q`` and ``P ∧ Q → Q`` distinct while identifying commutative/structural
rewrites.
"""

from __future__ import annotations

from itertools import product
from typing import List

from leandrift.core.prop import Prop
from leandrift.core.statement import Statement

MAX_ATOMS = 20  # guard against combinatorial blow-up on the decidable fragment


def _obligation(x) -> Prop:
    if isinstance(x, Statement):
        return x.alpha_normalize().obligation()
    return x.canonical()


def schematic_equivalent(a, b) -> bool:
    """True iff obligations of ``a`` and ``b`` are equivalent over all assignments."""
    pa, pb = _obligation(a), _obligation(b)
    atoms = sorted(pa.atom_keys() | pb.atom_keys())
    if len(atoms) > MAX_ATOMS:
        raise ValueError(f"too many atoms for exact enumeration: {len(atoms)}")
    for bits in product([False, True], repeat=len(atoms)):
        assign = dict(zip(atoms, bits))
        if pa.eval(assign) != pb.eval(assign):
            return False
    return True


def entails(a, b) -> bool:
    """Schematic entailment: every assignment satisfying obligation(a) satisfies obligation(b)."""
    pa, pb = _obligation(a), _obligation(b)
    atoms = sorted(pa.atom_keys() | pb.atom_keys())
    if len(atoms) > MAX_ATOMS:
        raise ValueError(f"too many atoms for exact enumeration: {len(atoms)}")
    for bits in product([False, True], repeat=len(atoms)):
        assign = dict(zip(atoms, bits))
        if pa.eval(assign) and not pb.eval(assign):
            return False
    return True


def is_weakening(s_star, s_hat) -> bool:
    """Definition 2: s_star ⊨ s_hat but s_hat ⊭ s_star (strictly easier)."""
    return entails(s_star, s_hat) and not entails(s_hat, s_star)


def faithful_oracle(s_hat, s_star) -> bool:
    """The exact analogue of DRIFTCLASS-Faithful used only for validation."""
    return schematic_equivalent(s_hat, s_star)
