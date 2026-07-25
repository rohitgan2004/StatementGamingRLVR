from leandrift.core.expr import BinOp, Lit, Var
from leandrift.core.prop import And, Atom, Const, Implies, Or
from leandrift.core.statement import Binder, Hypothesis, Statement
from leandrift.microproof.oracle import (
    entails,
    is_weakening,
    schematic_equivalent,
)


def n():
    return Var("n")


def atom_mod(k):
    return Atom("=", BinOp("%", n(), Lit(k)), Lit(0))


def test_ground_atom_preevaluated():
    # 2 = 2 is ground/true; 3 = 4 is ground/false.
    assert Atom("=", Lit(2), Lit(2)).atom_keys() == set()
    assert Atom("=", Lit(2), Lit(2)).eval({}) is True
    assert Atom("=", Lit(3), Lit(4)).eval({}) is False


def test_commutative_canonicalization():
    a, b = Var("a"), Var("b")
    p1 = Atom("=", BinOp("+", a, b), Lit(0))
    p2 = Atom("=", BinOp("+", b, a), Lit(0))
    assert p1.key() == p2.key()


def test_implication_not_tautology():
    # P -> Q is NOT equivalent to P & Q -> Q (the latter is a tautology).
    P, Q = atom_mod(6), atom_mod(2)
    s1 = Implies(P, Q)
    s2 = Implies(And((P, Q)), Q)
    assert not schematic_equivalent(s1, s2)


def test_faithful_restructuring_equivalent():
    P, Q, R = atom_mod(6), atom_mod(2), atom_mod(3)
    s1 = Implies(P, And((Q, R)))
    s2 = Implies(P, And((R, Q)))  # reordered conjunction
    assert schematic_equivalent(s1, s2)


def test_add_conclusion_is_weakening():
    P, Q = atom_mod(6), atom_mod(2)
    star = Statement.make("t", [Binder("n", "Int")], [Hypothesis("h", P)], Q)
    weak = star.with_hyps([Hypothesis("h", P), Hypothesis("hc", Q)])
    assert is_weakening(star, weak)
    assert not schematic_equivalent(star, weak)


def test_false_premise_is_weakening():
    P, Q = atom_mod(6), atom_mod(2)
    star = Statement.make("t", [Binder("n", "Int")], [Hypothesis("h", P)], Q)
    vac = star.with_hyps([Hypothesis("h", P), Hypothesis("hbot", Const(False))])
    assert is_weakening(star, vac)


def test_entails_reflexive():
    P, Q = atom_mod(6), atom_mod(2)
    star = Statement.make("t", [Binder("n", "Int")], [Hypothesis("h", P)], Q)
    assert entails(star, star)
    assert not is_weakening(star, star)
