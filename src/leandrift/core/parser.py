"""A small recursive-descent parser for the pinned-header statement fragment.

Real model output is Lean 4 text; DRIFTCLASS and the relatedness filter operate on
structured statements, so we parse the emitted ``theorem ... : ...`` back into a
``Statement``.  The pinned-header protocol constrains the grammar to the fragment
below (arithmetic terms, comparisons, propositional connectives, and leading
quantifiers), which keeps parsing tractable; anything outside it yields ``None``
(the completion is then treated as unparseable and fails both verifiers).

Deployed pipelines would use Lean's elaborator output instead; this parser is the
lightweight equivalent for the constrained template grammar.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from leandrift.core.expr import BinOp, Lit, Term, Var
from leandrift.core.prop import And, Atom, Const, Implies, Not, Or, Prop, Quant
from leandrift.core.statement import Binder, Hypothesis, Statement

# Normalize ASCII aliases to the canonical unicode operators.
_ALIASES = [
    ("/\\", " ∧ "), ("\\/", " ∨ "), ("->", " → "), ("→", " → "),
    ("<=", " ≤ "), (">=", " ≥ "), ("!=", " ≠ "), ("≠", " ≠ "),
    ("¬", " ¬ "), ("∧", " ∧ "), ("∨", " ∨ "), ("∀", " ∀ "), ("∃", " ∃ "),
    ("∣", " ∣ "), ("≤", " ≤ "), ("≥", " ≥ "),
]

_TOKEN_RE = re.compile(
    r"\s*(∀|∃|∧|∨|¬|→|≤|≥|≠|∣|<|>|=|\+|\-|\*|%|/|\(|\)|,|:|True|False|"
    r"Nat\.gcd|[A-Za-z_][A-Za-z0-9_']*|\d+)"
)


def _tokenize(s: str) -> List[str]:
    for a, b in _ALIASES:
        s = s.replace(a, b)
    toks, i = [], 0
    while i < len(s):
        m = _TOKEN_RE.match(s, i)
        if not m:
            if s[i].isspace():
                i += 1
                continue
            raise ValueError(f"cannot tokenize at: {s[i:][:20]!r}")
        toks.append(m.group(1))
        i = m.end()
    return toks


class _Parser:
    def __init__(self, toks: List[str]) -> None:
        self.toks = toks
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> str:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, t: str) -> None:
        if self.peek() != t:
            raise ValueError(f"expected {t!r}, got {self.peek()!r}")
        self.next()

    # prop := implication
    def parse_prop(self) -> Prop:
        return self._implies()

    def _implies(self) -> Prop:
        left = self._or()
        if self.peek() == "→":
            self.next()
            right = self._implies()
            return Implies(left, right)
        return left

    def _or(self) -> Prop:
        parts = [self._and()]
        while self.peek() == "∨":
            self.next()
            parts.append(self._and())
        return parts[0] if len(parts) == 1 else Or(tuple(parts))

    def _and(self) -> Prop:
        parts = [self._not()]
        while self.peek() == "∧":
            self.next()
            parts.append(self._not())
        return parts[0] if len(parts) == 1 else And(tuple(parts))

    def _not(self) -> Prop:
        if self.peek() == "¬":
            self.next()
            return Not(self._not())
        return self._quant_or_atom()

    def _quant_or_atom(self) -> Prop:
        t = self.peek()
        if t in ("∀", "∃"):
            self.next()
            var = self.next()
            self.expect(":")
            domain = self.next()
            self.expect(",")
            body = self.parse_prop()
            return Quant("forall" if t == "∀" else "exists", var, domain, body)
        if t == "(":
            # Could be a parenthesized prop or a parenthesized term comparison.
            save = self.i
            try:
                self.next()
                inner = self.parse_prop()
                self.expect(")")
                return inner
            except ValueError:
                self.i = save
        if t == "True":
            self.next()
            return Const(True)
        if t == "False":
            self.next()
            return Const(False)
        return self._atom()

    def _atom(self) -> Prop:
        left = self._term()
        rel = self.peek()
        if rel not in ("=", "≠", "<", "≤", ">", "≥", "∣"):
            raise ValueError(f"expected relation, got {rel!r}")
        self.next()
        right = self._term()
        return Atom(rel, left, right)

    # term := add
    def _term(self) -> Term:
        return self._add()

    def _add(self) -> Term:
        left = self._mul()
        while self.peek() in ("+", "-"):
            op = self.next()
            left = BinOp(op, left, self._mul())
        return left

    def _mul(self) -> Term:
        left = self._factor()
        while self.peek() in ("*", "%", "/"):
            op = self.next()
            left = BinOp(op, left, self._factor())
        return left

    def _factor(self) -> Term:
        t = self.peek()
        if t == "(":
            self.next()
            inner = self._term()
            self.expect(")")
            return inner
        if t == "Nat.gcd":
            self.next()
            a = self._factor()
            b = self._factor()
            return BinOp("gcd", a, b)
        if t is not None and t.isdigit():
            self.next()
            return Lit(int(t))
        if t is not None and re.match(r"[A-Za-z_]", t):
            self.next()
            return Var(t)
        raise ValueError(f"unexpected term token {t!r}")


def _split_binders(header: str) -> Tuple[List[Tuple[str, str, bool]], str]:
    """Split '(a : T) (h : P) ... : CONCL' into binder specs and conclusion text.

    Returns list of (name, body, is_binder_type) plus the conclusion string.
    """
    # Grab balanced (...) groups, then the trailing ': CONCL'.
    specs: List[Tuple[str, str]] = []
    i, n = 0, len(header)
    while i < n:
        if header[i] == "(":
            depth, j = 1, i + 1
            while j < n and depth:
                depth += (header[j] == "(") - (header[j] == ")")
                j += 1
            group = header[i + 1:j - 1]
            if ":" in group:
                name, body = group.split(":", 1)
                specs.append((name.strip(), body.strip()))
            i = j
        elif header[i] == ":":
            return specs, header[i + 1:].strip()
        else:
            i += 1
    return specs, ""


_TYPE_TOKENS = {"Int", "Nat", "ℝ", "ℚ", "ℤ", "ℕ", "Real", "Rat"}


def parse_statement(text: str) -> Optional[Statement]:
    text = text.strip().rstrip()
    m = re.match(r"theorem\s+([A-Za-z_][A-Za-z0-9_']*)\s*(.*)", text, re.DOTALL)
    if not m:
        return None
    name, rest = m.group(1), m.group(2)
    # Strip a trailing ':= ...' if present.
    rest = re.split(r":=", rest, maxsplit=1)[0].strip()
    try:
        specs, concl_text = _split_binders(rest)
        binders: List[Binder] = []
        hyps: List[Hypothesis] = []
        for nm, body in specs:
            names = nm.split()
            if body in _TYPE_TOKENS:
                for one in names:
                    binders.append(Binder(one, body))
            else:
                prop = _Parser(_tokenize(body)).parse_prop()
                for one in names:
                    hyps.append(Hypothesis(one, prop))
        concl = _Parser(_tokenize(concl_text)).parse_prop()
        return Statement(name, tuple(binders), tuple(hyps), concl)
    except (ValueError, IndexError):
        return None
