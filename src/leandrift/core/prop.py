"""Propositional AST for statement bodies.

This is the representation Definition 1 (schematic faithfulness) operates on:
each *canonicalized atomic comparison* and each *quantified subformula* abstracts
to a propositional variable, ground atoms are pre-evaluated, and two formulas are
faithful iff logically equivalent over all Boolean assignments to the union of
their atoms.  The exact oracle (``leandrift.microproof.oracle``) enumerates those
assignments; DRIFTCLASS (``leandrift.verifiers.driftclass``) is a *structural*
approximation over the same node types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from leandrift.core.expr import Term

# Relations that are symmetric under operand swap.
_SYMMETRIC = {"=", "≠"}
# Normalise > / >= to < / <= with swapped operands.
_FLIP = {">": "<", "≥": "≤"}


class Prop:
    """Base class for propositions."""

    def canonical(self) -> "Prop":  # pragma: no cover - overridden
        raise NotImplementedError

    def subst(self, mapping: Dict[str, str]) -> "Prop":  # pragma: no cover - overridden
        raise NotImplementedError

    def atom_keys(self) -> Set[str]:
        """Schematic atoms: comparison atoms + whole quantified subformulas."""
        raise NotImplementedError  # pragma: no cover

    def eval(self, assign: Dict[str, bool]) -> bool:
        """Evaluate under a Boolean assignment to (non-ground) atom keys."""
        raise NotImplementedError  # pragma: no cover

    def lean(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True)
class Const(Prop):
    value: bool

    def canonical(self) -> "Prop":
        return self

    def subst(self, mapping: Dict[str, str]) -> "Prop":
        return self

    def atom_keys(self) -> Set[str]:
        return set()

    def eval(self, assign: Dict[str, bool]) -> bool:
        return self.value

    def lean(self) -> str:
        return "True" if self.value else "False"


@dataclass(frozen=True)
class Atom(Prop):
    """An atomic comparison, e.g. ``n % 6 = 0`` or ``a * a ≥ 0``."""

    rel: str  # = ≠ < ≤ > ≥ ∣
    left: Term
    right: Term

    def canonical(self) -> "Prop":
        lhs = self.left.canonical()
        rhs = self.right.canonical()
        rel = self.rel
        if rel in _FLIP:
            rel = _FLIP[rel]
            lhs, rhs = rhs, lhs
        if rel in _SYMMETRIC and rhs < lhs:
            lhs, rhs = rhs, lhs
        return Atom(rel, lhs, rhs)

    def subst(self, mapping: Dict[str, str]) -> "Prop":
        return Atom(self.rel, self.left.subst(mapping), self.right.subst(mapping))

    def _ground_value(self) -> Optional[bool]:
        lv = self.left.eval_ground()
        rv = self.right.eval_ground()
        if lv is None or rv is None:
            return None
        return _rel_eval(self.rel, lv, rv)

    def atom_keys(self) -> Set[str]:
        # Ground atoms are pre-evaluated and contribute no free variable.
        if self._ground_value() is not None:
            return set()
        return {self.key()}

    def key(self) -> str:
        c = self.canonical()
        assert isinstance(c, Atom)
        return f"{c.left.lean()} {c.rel} {c.right.lean()}"

    def eval(self, assign: Dict[str, bool]) -> bool:
        g = self._ground_value()
        if g is not None:
            return g
        return assign[self.key()]

    def lean(self) -> str:
        if self.rel == "∣":
            return f"{self.left.lean()} ∣ {self.right.lean()}"
        return f"{self.left.lean()} {self.rel} {self.right.lean()}"


@dataclass(frozen=True)
class Not(Prop):
    inner: Prop

    def canonical(self) -> "Prop":
        return Not(self.inner.canonical())

    def subst(self, mapping: Dict[str, str]) -> "Prop":
        return Not(self.inner.subst(mapping))

    def atom_keys(self) -> Set[str]:
        return self.inner.atom_keys()

    def eval(self, assign: Dict[str, bool]) -> bool:
        return not self.inner.eval(assign)

    def lean(self) -> str:
        return f"¬({self.inner.lean()})"


@dataclass(frozen=True)
class And(Prop):
    conjuncts: tuple

    def canonical(self) -> "Prop":
        flat: List[Prop] = []
        for c in self.conjuncts:
            cc = c.canonical()
            if isinstance(cc, And):
                flat.extend(cc.conjuncts)
            elif cc == Const(True):
                continue
            else:
                flat.append(cc)
        if not flat:
            return Const(True)
        flat = sorted(set(flat), key=lambda p: p.lean())
        if len(flat) == 1:
            return flat[0]
        return And(tuple(flat))

    def subst(self, mapping: Dict[str, str]) -> "Prop":
        return And(tuple(c.subst(mapping) for c in self.conjuncts))

    def atom_keys(self) -> Set[str]:
        return set().union(*(c.atom_keys() for c in self.conjuncts)) if self.conjuncts else set()

    def eval(self, assign: Dict[str, bool]) -> bool:
        return all(c.eval(assign) for c in self.conjuncts)

    def lean(self) -> str:
        return " ∧ ".join(f"({c.lean()})" for c in self.conjuncts)


@dataclass(frozen=True)
class Or(Prop):
    disjuncts: tuple

    def canonical(self) -> "Prop":
        flat: List[Prop] = []
        for d in self.disjuncts:
            dd = d.canonical()
            if isinstance(dd, Or):
                flat.extend(dd.disjuncts)
            elif dd == Const(False):
                continue
            else:
                flat.append(dd)
        if not flat:
            return Const(False)
        flat = sorted(set(flat), key=lambda p: p.lean())
        if len(flat) == 1:
            return flat[0]
        return Or(tuple(flat))

    def subst(self, mapping: Dict[str, str]) -> "Prop":
        return Or(tuple(d.subst(mapping) for d in self.disjuncts))

    def atom_keys(self) -> Set[str]:
        return set().union(*(d.atom_keys() for d in self.disjuncts)) if self.disjuncts else set()

    def eval(self, assign: Dict[str, bool]) -> bool:
        return any(d.eval(assign) for d in self.disjuncts)

    def lean(self) -> str:
        return " ∨ ".join(f"({d.lean()})" for d in self.disjuncts)


@dataclass(frozen=True)
class Implies(Prop):
    hyp: Prop
    concl: Prop

    def canonical(self) -> "Prop":
        return Implies(self.hyp.canonical(), self.concl.canonical())

    def subst(self, mapping: Dict[str, str]) -> "Prop":
        return Implies(self.hyp.subst(mapping), self.concl.subst(mapping))

    def atom_keys(self) -> Set[str]:
        return self.hyp.atom_keys() | self.concl.atom_keys()

    def eval(self, assign: Dict[str, bool]) -> bool:
        return (not self.hyp.eval(assign)) or self.concl.eval(assign)

    def lean(self) -> str:
        return f"({self.hyp.lean()}) → ({self.concl.lean()})"


@dataclass(frozen=True)
class Quant(Prop):
    """A quantified subformula, abstracted to a single propositional variable.

    Per Definition 1 we do *not* expand quantifiers; the whole ``∀x, body`` (or
    ``∃x, body``) is one schematic atom.  The binder order is part of the key, so
    ``∀ε ∃N`` and ``∃N ∀ε`` produce distinct atoms (quantifier-scope drift).
    """

    kind: str  # "forall" or "exists"
    var: str
    domain: str
    body: Prop

    def canonical(self) -> "Prop":
        return Quant(self.kind, self.var, self.domain, self.body.canonical())

    def subst(self, mapping: Dict[str, str]) -> "Prop":
        new_var = mapping.get(self.var, self.var)
        return Quant(self.kind, new_var, self.domain, self.body.subst(mapping))

    def atom_keys(self) -> Set[str]:
        return {self.key()}

    def key(self) -> str:
        c = self.canonical()
        assert isinstance(c, Quant)
        q = "∀" if c.kind == "forall" else "∃"
        return f"{q}{c.var}:{c.domain}.{c.body.lean()}"

    def eval(self, assign: Dict[str, bool]) -> bool:
        return assign[self.key()]

    def lean(self) -> str:
        q = "∀" if self.kind == "forall" else "∃"
        return f"{q} {self.var} : {self.domain}, {self.body.lean()}"


def _rel_eval(rel: str, a: int, b: int) -> bool:
    """Evaluate a ground atomic relation over integer values (for pre-evaluation)."""
    if rel == "=":
        return a == b
    if rel == "≠":
        return a != b
    if rel == "<":
        return a < b
    if rel == "≤":
        return a <= b
    if rel == ">":
        return a > b
    if rel == "≥":
        return a >= b
    if rel == "∣":
        return (b % a == 0) if a != 0 else (b == 0)
    raise ValueError(f"unknown relation {rel}")


def conj(props: List[Prop]) -> Prop:
    """Build a canonical conjunction (``True`` when empty)."""
    if not props:
        return Const(True)
    return And(tuple(props)).canonical()
