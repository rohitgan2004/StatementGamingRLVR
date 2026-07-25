"""Structured Lean theorem statements under the pinned-header protocol.

A ``Statement`` fixes everything the pinned header fixes (theorem name, binder
names, binder order, hypothesis-name skeleton) and exposes the semantic content
as structured slots so that the *only* free channel is the meaning of the
hypotheses and conclusion.  This is what makes any divergence between the
generated statement ``s_hat`` and the locked intended statement ``s_star``
semantic (Section 4.1), which is exactly what DRIFTCLASS is built to classify.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Tuple

from leandrift.core.prop import Prop, Quant, Implies, conj


@dataclass(frozen=True)
class Binder:
    name: str
    type: str  # e.g. "Int", "Nat", "ℝ"

    def lean(self) -> str:
        return f"({self.name} : {self.type})"


@dataclass(frozen=True)
class Hypothesis:
    name: str
    prop: Prop
    is_side_condition: bool = False  # e.g. `0 < n` positivity constraints

    def lean(self) -> str:
        return f"({self.name} : {self.prop.lean()})"


@dataclass(frozen=True)
class Statement:
    name: str
    binders: Tuple[Binder, ...]
    hyps: Tuple[Hypothesis, ...]
    conclusion: Prop

    # ---- construction helpers -------------------------------------------------
    @staticmethod
    def make(name: str, binders, hyps, conclusion: Prop) -> "Statement":
        return Statement(
            name=name,
            binders=tuple(binders),
            hyps=tuple(hyps),
            conclusion=conclusion,
        )

    def with_hyps(self, hyps) -> "Statement":
        return replace(self, hyps=tuple(hyps))

    def with_conclusion(self, concl: Prop) -> "Statement":
        return replace(self, conclusion=concl)

    def alpha_normalize(self) -> "Statement":
        """Rename binder variables to canonical positional names (v0, v1, ...).

        This is the alpha-normalization DRIFTCLASS applies before slot comparison
        so that bound-renamed-but-otherwise-identical statements compare faithful.
        """
        mapping = {b.name: f"v{i}" for i, b in enumerate(self.binders)}
        new_binders = tuple(Binder(f"v{i}", b.type) for i, b in enumerate(self.binders))
        new_hyps = tuple(
            Hypothesis(h.name, h.prop.subst(mapping), h.is_side_condition) for h in self.hyps
        )
        new_concl = self.conclusion.subst(mapping)
        return Statement(self.name, new_binders, new_hyps, new_concl)

    # ---- semantics ------------------------------------------------------------
    def obligation(self) -> Prop:
        """The proof obligation as a single proposition: (∧ hyps) → conclusion."""
        if not self.hyps:
            return self.conclusion.canonical()
        antecedent = conj([h.prop for h in self.hyps])
        return Implies(antecedent, self.conclusion).canonical()

    # ---- DRIFTCLASS slots -----------------------------------------------------
    def binder_types(self) -> Tuple[str, ...]:
        return tuple(b.type for b in self.binders)

    def quantifier_signature(self) -> Tuple[str, ...]:
        """Ordered kinds of leading quantifiers in the conclusion (for scope drift)."""
        sig: List[str] = []
        node = self.conclusion
        while isinstance(node, Quant):
            sig.append(node.kind)
            node = node.body
        return tuple(sig)

    def hypothesis_keys(self) -> Tuple[str, ...]:
        return tuple(h.prop.canonical().lean() for h in self.hyps if not h.is_side_condition)

    def side_condition_keys(self) -> Tuple[str, ...]:
        return tuple(h.prop.canonical().lean() for h in self.hyps if h.is_side_condition)

    def conclusion_key(self) -> str:
        return self.conclusion.canonical().lean()

    # ---- serialization --------------------------------------------------------
    def to_lean(self) -> str:
        parts = [f"theorem {self.name}"]
        for b in self.binders:
            parts.append(b.lean())
        for h in self.hyps:
            parts.append(h.lean())
        head = " ".join(parts)
        return f"{head} : {self.conclusion.lean()}"

    def pinned_header(self) -> str:
        """The header handed to the policy: names/binders fixed, propositions blank."""
        parts = [f"theorem {self.name}"]
        for b in self.binders:
            parts.append(b.lean())
        for h in self.hyps:
            parts.append(f"({h.name} : _)")
        head = " ".join(parts)
        return f"{head} : _ := by"

    def symbol_multiset(self) -> List[str]:
        """Identifier + operator multiset used by the relatedness filter."""
        from leandrift.verifiers.related import symbols_of_statement

        return symbols_of_statement(self)
