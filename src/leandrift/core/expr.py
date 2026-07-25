"""Arithmetic term sublanguage used inside atomic comparisons.

Terms are the operands of the atomic relations that appear in LEANDRIFT and
MICROPROOF statements (e.g. ``n % 6`` in ``n % 6 = 0``). Keeping terms
structured (rather than as opaque strings) lets us:

  * canonicalize commutative operators so ``a + b`` and ``b + a`` collapse to one
    atom (required for the faithful-restructuring case of DRIFTCLASS / the
    oracle), and
  * evaluate *ground* terms (all-literal operands) so ``2 = 2`` can be
    pre-evaluated to ``True`` per Definition 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# Operators whose operands may be reordered without changing meaning.
COMMUTATIVE = {"+", "*", "gcd"}


class Term:
    """Base class for arithmetic terms."""

    def canonical(self) -> "Term":  # pragma: no cover - overridden
        raise NotImplementedError

    def subst(self, mapping: Dict[str, str]) -> "Term":  # pragma: no cover - overridden
        raise NotImplementedError

    def eval_ground(self) -> Optional[int]:
        """Return the integer value if the term is ground (no variables), else None."""
        raise NotImplementedError  # pragma: no cover

    def lean(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def __lt__(self, other: "Term") -> bool:
        # Total order used to sort commutative operands deterministically.
        return self.canonical().lean() < other.canonical().lean()


@dataclass(frozen=True)
class Var(Term):
    name: str

    def canonical(self) -> "Term":
        return self

    def subst(self, mapping: Dict[str, str]) -> "Term":
        return Var(mapping.get(self.name, self.name))

    def eval_ground(self) -> Optional[int]:
        return None

    def lean(self) -> str:
        return self.name


@dataclass(frozen=True)
class Lit(Term):
    value: int

    def canonical(self) -> "Term":
        return self

    def subst(self, mapping: Dict[str, str]) -> "Term":
        return self

    def eval_ground(self) -> Optional[int]:
        return self.value

    def lean(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class BinOp(Term):
    op: str  # one of + - * % / gcd
    left: Term
    right: Term

    def canonical(self) -> "Term":
        lhs = self.left.canonical()
        rhs = self.right.canonical()
        # Constant-fold ground subterms.
        lv, rv = lhs.eval_ground(), rhs.eval_ground()
        if lv is not None and rv is not None:
            folded = _apply(self.op, lv, rv)
            if folded is not None:
                return Lit(folded)
        if self.op in COMMUTATIVE and rhs < lhs:
            lhs, rhs = rhs, lhs
        return BinOp(self.op, lhs, rhs)

    def subst(self, mapping: Dict[str, str]) -> "Term":
        return BinOp(self.op, self.left.subst(mapping), self.right.subst(mapping))

    def eval_ground(self) -> Optional[int]:
        lv = self.left.eval_ground()
        rv = self.right.eval_ground()
        if lv is None or rv is None:
            return None
        return _apply(self.op, lv, rv)

    def lean(self) -> str:
        if self.op == "gcd":
            return f"Nat.gcd {_paren(self.left)} {_paren(self.right)}"
        return f"{_paren(self.left)} {self.op} {_paren(self.right)}"


def _paren(t: Term) -> str:
    if isinstance(t, BinOp) and t.op != "gcd":
        return f"({t.lean()})"
    return t.lean()


def _apply(op: str, a: int, b: int) -> Optional[int]:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "%":
        return a % b if b != 0 else None
    if op == "/":
        return a // b if b != 0 else None
    if op == "gcd":
        from math import gcd

        return gcd(a, b)
    return None  # pragma: no cover
