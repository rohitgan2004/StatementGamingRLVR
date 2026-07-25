"""Semantic mock Lean backend.

Reproduces Lean's accept/reject behavior for the LEANDRIFT template fragments
*without* Lean, so the entire pipeline runs on a laptop.  Acceptance =
(the tactic is adequate for the goal shape) AND (the proof obligation is actually
true), plus the two context-based shortcuts a real Lean would also accept:
``exact <h>`` when a hypothesis provides the goal, and ``exact hbot.elim`` when a
``False`` hypothesis is in scope.  Truth is decided by sampling concrete values
for the binders -- exact for our decidable arithmetic fragments, and clearly a
simulation for anything else.
"""

from __future__ import annotations

import random
import time
from typing import Dict, Optional

from leandrift.core.expr import BinOp, Lit, Term, Var
from leandrift.core.prop import And, Atom, Const, Implies, Not, Or, Prop, Quant
from leandrift.core.statement import Statement
from leandrift.lean.backend import LeanBackend, LeanResult

_N_SAMPLES = 200


def _term_value(t: Term, env: Dict[str, float]) -> float:
    if isinstance(t, Var):
        return env[t.name]
    if isinstance(t, Lit):
        return t.value
    if isinstance(t, BinOp):
        a, b = _term_value(t.left, env), _term_value(t.right, env)
        if t.op == "+":
            return a + b
        if t.op == "-":
            return a - b
        if t.op == "*":
            return a * b
        if t.op == "%":
            return a % b if b != 0 else float("nan")
        if t.op == "/":
            return a // b if b != 0 else float("nan")
        if t.op == "gcd":
            from math import gcd

            return gcd(int(a), int(b))
    raise ValueError(f"cannot evaluate term: {t}")  # pragma: no cover


def _rel_holds(rel: str, a: float, b: float) -> bool:
    if a != a or b != b:  # NaN from div/mod by zero
        return False
    if rel == "=":
        return abs(a - b) < 1e-9
    if rel == "≠":
        return abs(a - b) >= 1e-9
    if rel == "<":
        return a < b - 1e-12
    if rel == "≤":
        return a <= b + 1e-9
    if rel == ">":
        return a > b + 1e-12
    if rel == "≥":
        return a >= b - 1e-9
    if rel == "∣":
        return b % a == 0 if a != 0 else (b == 0)
    raise ValueError(f"unknown relation {rel}")  # pragma: no cover


def _prop_holds(p: Prop, env: Dict[str, float]) -> bool:
    if isinstance(p, Const):
        return p.value
    if isinstance(p, Atom):
        return _rel_holds(p.rel, _term_value(p.left, env), _term_value(p.right, env))
    if isinstance(p, Not):
        return not _prop_holds(p.inner, env)
    if isinstance(p, And):
        return all(_prop_holds(c, env) for c in p.conjuncts)
    if isinstance(p, Or):
        return any(_prop_holds(d, env) for d in p.disjuncts)
    if isinstance(p, Implies):
        return (not _prop_holds(p.hyp, env)) or _prop_holds(p.concl, env)
    if isinstance(p, Quant):
        # Sample the bound variable; universal must hold for all samples,
        # existential for some. Approximate but adequate for the mock.
        vals = [random.uniform(-8, 8) for _ in range(24)]
        holds = [_prop_holds(p.body, {**env, p.var: v}) for v in vals]
        return all(holds) if p.kind == "forall" else any(holds)
    raise ValueError(f"cannot evaluate prop: {p}")  # pragma: no cover


def _sample_env(stmt: Statement, rng: random.Random) -> Dict[str, float]:
    env: Dict[str, float] = {}
    for b in stmt.binders:
        if b.type == "Nat":
            env[b.name] = rng.randint(0, 60)
        elif b.type == "Int":
            env[b.name] = rng.randint(-60, 60)
        else:  # ℝ or other ordered field
            env[b.name] = rng.uniform(-10, 10)
    return env


class MockLeanBackend(LeanBackend):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._valid_cache: Dict[str, bool] = {}

    # ---- truth checking -------------------------------------------------------
    def _is_valid(self, stmt: Statement) -> bool:
        key = stmt.obligation().lean()
        if key in self._valid_cache:
            return self._valid_cache[key]
        obligation = stmt.obligation()
        ok = True
        for _ in range(_N_SAMPLES):
            env = _sample_env(stmt, self._rng)
            if not _prop_holds(obligation, env):
                ok = False
                break
        self._valid_cache[key] = ok
        return ok

    # ---- tactic adequacy ------------------------------------------------------
    @staticmethod
    def _is_reflexive(p: Prop) -> bool:
        pc = p.canonical()
        if isinstance(pc, Atom) and pc.rel in ("=", "≤", "≥"):
            return pc.left.canonical() == pc.right.canonical()
        if isinstance(pc, And):
            return all(MockLeanBackend._is_reflexive(c) for c in pc.conjuncts)
        return False

    @staticmethod
    def _is_int_arith(stmt: Statement) -> bool:
        return all(b.type in ("Int", "Nat") for b in stmt.binders)

    @staticmethod
    def _is_real_arith(stmt: Statement) -> bool:
        return any(b.type not in ("Int", "Nat") for b in stmt.binders)

    def _closes(self, proof: str, stmt: Statement) -> bool:
        p = proof.strip()
        concl = stmt.conclusion

        # Context shortcuts (accepted regardless of goal truth).
        if p.startswith("exact "):
            arg = p[len("exact "):].strip()
            if ".elim" in arg or arg.startswith("absurd") or "False.elim" in arg:
                return any(h.prop.canonical() == Const(False) for h in stmt.hyps)
            base = arg.split(".")[0].split()[0]
            for h in stmt.hyps:
                if h.name == base:
                    return self._hyp_provides(h.prop, arg, concl)
            return False

        valid = self._is_valid(stmt)

        if p == "rfl":
            return self._is_reflexive(concl)
        if p in ("trivial", "simp", "norm_num", "decide", "tauto") or p.startswith("simp"):
            # These only dispatch structurally-trivial goals here.
            return self._is_reflexive(concl) or concl.canonical() == Const(True)
        if "omega" in p:
            return valid and self._is_int_arith(stmt)
        if "nlinarith" in p or "positivity" in p or "polyrith" in p:
            return valid and self._is_real_arith(stmt)
        if "linarith" in p:
            return valid
        if p.startswith("constructor"):
            # e.g. `constructor <;> omega` / `constructor <;> rfl`
            inner = p.split("<;>")[-1].strip() if "<;>" in p else ""
            if not isinstance(concl.canonical(), And):
                return False
            if "omega" in inner:
                return valid and self._is_int_arith(stmt)
            if "rfl" in inner:
                return self._is_reflexive(concl)
            if "nlinarith" in inner or "linarith" in inner:
                return valid
            return False
        if p.startswith("intro") and "simp" in p:
            # A common *failing* honest first attempt (cf. the paper's example).
            return self._is_reflexive(concl)
        # Unknown / inadequate tactic.
        return False

    @staticmethod
    def _hyp_provides(hyp_prop: Prop, arg: str, concl: Prop) -> bool:
        hc = hyp_prop.canonical()
        cc = concl.canonical()
        if arg.endswith(".right") or arg.endswith(".2"):
            return isinstance(hc, And) and hc.conjuncts[-1] == cc
        if arg.endswith(".left") or arg.endswith(".1"):
            return isinstance(hc, And) and hc.conjuncts[0] == cc
        return hc == cc

    # ---- backend API ----------------------------------------------------------
    def check(
        self,
        statement_source: str,
        proof_source: str,
        statement: Optional[Statement] = None,
    ) -> LeanResult:
        t0 = time.time()
        if statement is None:
            # Without structure the mock cannot reason; treat as elaboration fail.
            return LeanResult(kernel_ok=False, error="mock: no structured statement",
                              elapsed_s=time.time() - t0, simulated=True)
        ok = self._closes(proof_source, statement)
        return LeanResult(
            kernel_ok=ok,
            sorries=[],
            axioms=[] if ok else None,
            elapsed_s=time.time() - t0,
            simulated=True,
        )
