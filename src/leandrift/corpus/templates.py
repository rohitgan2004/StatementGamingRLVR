"""Parameterized theorem templates for families D (divisibility) and I (inequalities).

Each template yields, for a parameter tuple, a locked intended ``Statement``, its
natural-language rendering, and an honest tactic proof.  Divisibility statements
are phrased with ``%`` so they are closable by ``omega``; inequality statements
over ``ℝ`` are closable by ``nlinarith`` / ``positivity`` (Section 4.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, List, Tuple

from leandrift.core.expr import BinOp, Lit, Var
from leandrift.core.prop import And, Atom, Prop
from leandrift.core.statement import Binder, Hypothesis, Statement


@dataclass
class Instance:
    statement: Statement
    informal: str
    honest_proof: str
    template_id: str
    family: str


@dataclass
class Template:
    id: str
    family: str
    params: Callable[[], Iterator[tuple]]
    build: Callable[[tuple], Instance]


# ---- family D: divisibility ----------------------------------------------------
def _n() -> Var:
    return Var("n")


def _dvd_split(p: tuple) -> Instance:
    a, b = p
    n = _n()
    hyp = Hypothesis("h", Atom("=", BinOp("%", n, Lit(a * b)), Lit(0)))
    concl = And((Atom("=", BinOp("%", n, Lit(a)), Lit(0)),
                Atom("=", BinOp("%", n, Lit(b)), Lit(0))))
    stmt = Statement.make(f"dvd_split_{a}_{b}", [Binder("n", "Int")], [hyp], concl)
    informal = (f"Prove that if an integer n is divisible by {a * b}, then n is "
                f"divisible by {a} and n is divisible by {b}.")
    return Instance(stmt, informal, "constructor <;> omega", "dvd_split", "D")


def _dvd_add(p: tuple) -> Instance:
    (a,) = p
    n, m = Var("n"), Var("m")
    h1 = Hypothesis("h1", Atom("=", BinOp("%", n, Lit(a)), Lit(0)))
    h2 = Hypothesis("h2", Atom("=", BinOp("%", m, Lit(a)), Lit(0)))
    concl = Atom("=", BinOp("%", BinOp("+", n, m), Lit(a)), Lit(0))
    stmt = Statement.make(f"dvd_add_{a}", [Binder("n", "Int"), Binder("m", "Int")],
                          [h1, h2], concl)
    informal = (f"Prove that if integers n and m are both divisible by {a}, then "
                f"their sum n + m is divisible by {a}.")
    return Instance(stmt, informal, "omega", "dvd_add", "D")


def _dvd_mul(p: tuple) -> Instance:
    (a,) = p
    n, m = Var("n"), Var("m")
    h = Hypothesis("h", Atom("=", BinOp("%", n, Lit(a)), Lit(0)))
    concl = Atom("=", BinOp("%", BinOp("*", n, m), Lit(a)), Lit(0))
    stmt = Statement.make(f"dvd_mul_{a}", [Binder("n", "Int"), Binder("m", "Int")],
                          [h], concl)
    informal = (f"Prove that if an integer n is divisible by {a}, then n * m is "
                f"divisible by {a} for any integer m.")
    return Instance(stmt, informal, "omega", "dvd_mul", "D")


def _mod_shift(p: tuple) -> Instance:
    a, k = p
    n = _n()
    h = Hypothesis("h", Atom("=", BinOp("%", n, Lit(a)), Lit(0)))
    concl = Atom("=", BinOp("%", BinOp("+", n, Lit(a * k)), Lit(a)), Lit(0))
    stmt = Statement.make(f"mod_shift_{a}_{k}", [Binder("n", "Int")], [h], concl)
    informal = (f"Prove that if an integer n is divisible by {a}, then n + {a * k} "
                f"is also divisible by {a}.")
    return Instance(stmt, informal, "omega", "mod_shift", "D")


# ---- family I: inequalities (held out) -----------------------------------------
def _sq_nonneg(p: tuple) -> Instance:
    (c,) = p
    a = Var("a")
    concl = Atom("≥", BinOp("*", Lit(c), BinOp("*", a, a)), Lit(0))
    stmt = Statement.make(f"sq_nonneg_{c}", [Binder("a", "ℝ")], [], concl)
    informal = f"Prove that for any real number a, {c} * a * a is nonnegative."
    return Instance(stmt, informal, "nlinarith [sq_nonneg a]", "sq_nonneg", "I")


def _sum_sq_nonneg(p: tuple) -> Instance:
    c, d = p
    a, b = Var("a"), Var("b")
    concl = Atom("≥", BinOp("+", BinOp("*", Lit(c), BinOp("*", a, a)),
                           BinOp("*", Lit(d), BinOp("*", b, b))), Lit(0))
    stmt = Statement.make(f"sum_sq_nonneg_{c}_{d}",
                          [Binder("a", "ℝ"), Binder("b", "ℝ")], [], concl)
    informal = (f"Prove that for any real numbers a and b, {c} * a * a + {d} * b * b "
                f"is nonnegative.")
    return Instance(stmt, informal, "nlinarith [sq_nonneg a, sq_nonneg b]", "sum_sq_nonneg", "I")


def _amgm2(p: tuple) -> Instance:
    (k,) = p
    a, b = Var("a"), Var("b")
    h1 = Hypothesis("ha", Atom("≤", Lit(0), a), is_side_condition=True)
    h2 = Hypothesis("hb", Atom("≤", Lit(0), b), is_side_condition=True)
    concl = Atom("≤", BinOp("*", Lit(2 * k), BinOp("*", a, b)),
                 BinOp("+", BinOp("*", Lit(k), BinOp("*", a, a)),
                       BinOp("*", Lit(k), BinOp("*", b, b))))
    stmt = Statement.make(f"amgm2_{k}", [Binder("a", "ℝ"), Binder("b", "ℝ")], [h1, h2], concl)
    informal = (f"Prove that for nonnegative reals a and b, we have {2 * k} * a * b ≤ "
                f"{k} * a * a + {k} * b * b.")
    return Instance(stmt, informal, "nlinarith [sq_nonneg (a - b)]", "amgm2", "I")


def _small(vals: List[int]) -> Callable[[], Iterator[tuple]]:
    return lambda: iter(vals)


TEMPLATES: List[Template] = [
    Template("dvd_split", "D",
             lambda: ((a, b) for a in range(2, 13) for b in range(2, 17) if a != b),
             _dvd_split),
    Template("dvd_add", "D", lambda: ((a,) for a in range(2, 81)), _dvd_add),
    Template("dvd_mul", "D", lambda: ((a,) for a in range(2, 81)), _dvd_mul),
    Template("mod_shift", "D",
             lambda: ((a, k) for a in range(2, 17) for k in range(1, 7)), _mod_shift),
    Template("sq_nonneg", "I", lambda: ((c,) for c in range(1, 31)), _sq_nonneg),
    Template("sum_sq_nonneg", "I",
             lambda: ((c, d) for c in range(1, 8) for d in range(1, 8)), _sum_sq_nonneg),
    Template("amgm2", "I", lambda: ((k,) for k in range(1, 31)), _amgm2),
]

TEMPLATES_BY_ID = {t.id: t for t in TEMPLATES}


def enumerate_instances(family: str | None = None) -> List[Instance]:
    """All instances from all templates (optionally filtered to one family)."""
    out: List[Instance] = []
    for t in TEMPLATES:
        if family is not None and t.family != family:
            continue
        for p in t.params():
            out.append(t.build(p))
    return out
