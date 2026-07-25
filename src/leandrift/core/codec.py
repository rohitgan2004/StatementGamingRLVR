"""(De)serialization of the AST, statements, and prompts to plain dicts/JSON."""

from __future__ import annotations

from typing import Any, Dict

from leandrift.core.expr import BinOp, Lit, Term, Var
from leandrift.core.prop import And, Atom, Const, Implies, Not, Or, Prop, Quant
from leandrift.core.episode import Prompt
from leandrift.core.statement import Binder, Hypothesis, Statement


# ---- terms ---------------------------------------------------------------------
def term_to_dict(t: Term) -> Dict[str, Any]:
    if isinstance(t, Var):
        return {"t": "var", "name": t.name}
    if isinstance(t, Lit):
        return {"t": "lit", "value": t.value}
    if isinstance(t, BinOp):
        return {"t": "bin", "op": t.op, "l": term_to_dict(t.left), "r": term_to_dict(t.right)}
    raise TypeError(t)  # pragma: no cover


def term_from_dict(d: Dict[str, Any]) -> Term:
    k = d["t"]
    if k == "var":
        return Var(d["name"])
    if k == "lit":
        return Lit(d["value"])
    if k == "bin":
        return BinOp(d["op"], term_from_dict(d["l"]), term_from_dict(d["r"]))
    raise ValueError(d)  # pragma: no cover


# ---- props ---------------------------------------------------------------------
def prop_to_dict(p: Prop) -> Dict[str, Any]:
    if isinstance(p, Const):
        return {"p": "const", "value": p.value}
    if isinstance(p, Atom):
        return {"p": "atom", "rel": p.rel, "l": term_to_dict(p.left), "r": term_to_dict(p.right)}
    if isinstance(p, Not):
        return {"p": "not", "x": prop_to_dict(p.inner)}
    if isinstance(p, And):
        return {"p": "and", "xs": [prop_to_dict(c) for c in p.conjuncts]}
    if isinstance(p, Or):
        return {"p": "or", "xs": [prop_to_dict(d) for d in p.disjuncts]}
    if isinstance(p, Implies):
        return {"p": "imp", "h": prop_to_dict(p.hyp), "c": prop_to_dict(p.concl)}
    if isinstance(p, Quant):
        return {"p": "quant", "kind": p.kind, "var": p.var,
                "domain": p.domain, "body": prop_to_dict(p.body)}
    raise TypeError(p)  # pragma: no cover


def prop_from_dict(d: Dict[str, Any]) -> Prop:
    k = d["p"]
    if k == "const":
        return Const(d["value"])
    if k == "atom":
        return Atom(d["rel"], term_from_dict(d["l"]), term_from_dict(d["r"]))
    if k == "not":
        return Not(prop_from_dict(d["x"]))
    if k == "and":
        return And(tuple(prop_from_dict(x) for x in d["xs"]))
    if k == "or":
        return Or(tuple(prop_from_dict(x) for x in d["xs"]))
    if k == "imp":
        return Implies(prop_from_dict(d["h"]), prop_from_dict(d["c"]))
    if k == "quant":
        return Quant(d["kind"], d["var"], d["domain"], prop_from_dict(d["body"]))
    raise ValueError(d)  # pragma: no cover


# ---- statements ----------------------------------------------------------------
def statement_to_dict(s: Statement) -> Dict[str, Any]:
    return {
        "name": s.name,
        "binders": [{"name": b.name, "type": b.type} for b in s.binders],
        "hyps": [{"name": h.name, "prop": prop_to_dict(h.prop),
                  "side": h.is_side_condition} for h in s.hyps],
        "conclusion": prop_to_dict(s.conclusion),
    }


def statement_from_dict(d: Dict[str, Any]) -> Statement:
    return Statement(
        name=d["name"],
        binders=tuple(Binder(b["name"], b["type"]) for b in d["binders"]),
        hyps=tuple(Hypothesis(h["name"], prop_from_dict(h["prop"]),
                              h.get("side", False)) for h in d["hyps"]),
        conclusion=prop_from_dict(d["conclusion"]),
    )


# ---- prompts -------------------------------------------------------------------
def prompt_to_dict(p: Prompt) -> Dict[str, Any]:
    return {
        "id": p.id,
        "family": p.family,
        "informal": p.informal,
        "intended": statement_to_dict(p.intended),
        "template": p.template,
    }


def prompt_from_dict(d: Dict[str, Any]) -> Prompt:
    return Prompt(
        id=d["id"],
        family=d["family"],
        informal=d["informal"],
        intended=statement_from_dict(d["intended"]),
        template=d["template"],
    )
