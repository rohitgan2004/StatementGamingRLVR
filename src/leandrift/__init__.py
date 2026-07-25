"""LEANDRIFT: statement gaming in RLVR autoformalize-then-prove pipelines.

A Lean 4 / Mathlib training environment that measures ``statement gaming``: the
failure mode in which RLVR teaches a prover to weaken the theorem it formalizes
until an easy proof passes, rather than to prove the intended theorem.

See the paper ``Proving the Wrong Theorem`` (Rajagopalan) for the full design.
"""

__version__ = "0.1.0"

from leandrift.core.prop import (
    Atom,
    And,
    Or,
    Not,
    Implies,
    Const,
    Quant,
    Prop,
)
from leandrift.core.statement import Statement, Binder, Hypothesis

__all__ = [
    "__version__",
    "Atom",
    "And",
    "Or",
    "Not",
    "Implies",
    "Const",
    "Quant",
    "Prop",
    "Statement",
    "Binder",
    "Hypothesis",
]
