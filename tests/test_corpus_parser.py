from leandrift.core.parser import parse_statement
from leandrift.corpus.generator import generate
from leandrift.corpus.templates import enumerate_instances
from leandrift.corpus.weakenings import all_weakenings
from leandrift.rl.prompt import parse_completion
from leandrift.utils.config import load_config


def test_parser_roundtrip_all_instances():
    bad = 0
    for inst in enumerate_instances():
        cands = [inst.statement] + [w.statement for w in all_weakenings(inst.statement)]
        for s in cands:
            p = parse_statement(s.to_lean())
            if p is None or p.obligation() != s.obligation():
                bad += 1
    assert bad == 0


def test_parse_completion_blocks():
    raw = ("FORMAL_STATEMENT:\ntheorem t (n : Int) (h : n % 6 = 0) : n % 2 = 0 /\\ n % 3 = 0\n"
           "PROOF:\nconstructor <;> omega")
    c = parse_completion(raw)
    assert c.parsed_ok and c.statement is not None
    assert "constructor" in c.proof


def test_parse_completion_malformed():
    c = parse_completion("I could not solve this problem.")
    assert not c.parsed_ok and c.statement is None


def test_corpus_calibration_band():
    cfg = load_config("configs/base.yaml")
    corpus = generate(cfg)
    lo, hi = cfg["corpus"]["honest_band"]
    assert len(corpus.train) == cfg["corpus"]["n_train"]
    for it in corpus.train:
        assert lo <= it.honest_base_rate <= hi
        assert it.drift_advantage >= cfg["corpus"]["drift_advantage_threshold"]
    # training family is D only
    assert all(it.prompt.family == "D" for it in corpus.train)
    # eval set is stratified across D and I
    fams = {it.prompt.family for it in corpus.eval}
    assert "I" in fams
