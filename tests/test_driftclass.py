from leandrift.corpus.templates import enumerate_instances
from leandrift.corpus.weakenings import (
    ADDED_HYP,
    FAITHFUL,
    STRENGTHENED_PREMISE,
    WEAKENED_CONCL,
    add_conclusion_as_hypothesis,
    strengthen_premise,
    weaken_conclusion,
)
from leandrift.microproof.validate import validate
from leandrift.verifiers import driftclass


def _dvd_split():
    return [i for i in enumerate_instances() if i.statement.name == "dvd_split_2_3"][0].statement


def test_faithful_on_self():
    s = _dvd_split()
    assert driftclass.classify(s, s).drift_class == FAITHFUL
    assert driftclass.faithful(s, s)


def test_added_hypothesis_detected():
    s = _dvd_split()
    w = add_conclusion_as_hypothesis(s).statement
    r = driftclass.classify(w, s)
    assert r.drift_class == ADDED_HYP and not r.faithful


def test_weakened_conclusion_detected():
    s = _dvd_split()
    w = weaken_conclusion(s).statement
    assert driftclass.classify(w, s).drift_class == WEAKENED_CONCL


def test_strengthened_premise_detected():
    s = _dvd_split()
    w = strengthen_premise(s).statement
    assert driftclass.classify(w, s).drift_class == STRENGTHENED_PREMISE


def test_alpha_renaming_is_faithful():
    s = _dvd_split()
    renamed = s.alpha_normalize()  # renames n -> v0
    assert driftclass.faithful(renamed, s)


def test_validation_agreement_high():
    rep = validate(seed=0)
    assert rep.n_candidates > 500
    assert rep.agreement >= 0.99
    assert rep.precision >= 0.95
    assert rep.recall >= 0.90
