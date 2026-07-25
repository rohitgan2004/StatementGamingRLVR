"""End-to-end simulation smoke test: the hacking gap must open in E, not in B/H."""

from leandrift.corpus.generator import generate
from leandrift.rl.train_sim import run_sim
from leandrift.utils.config import apply_overrides, load_config


def _run(arm: str, steps: int, corpus):
    cfg = apply_overrides(load_config(f"configs/arm_{arm}.yaml"), {"training.steps": steps})
    return run_sim(cfg, corpus)["summary"]["final"]


def test_gap_opens_in_E_not_B_or_H():
    corpus = generate(load_config("configs/base.yaml"))
    b = _run("B", 150, corpus)
    e = _run("E", 150, corpus)
    h = _run("H", 150, corpus)
    # Arm E opens a substantial gap; B and H stay near zero.
    assert e["delta_hack"] > 0.3
    assert b["delta_hack"] < 0.05
    assert h["delta_hack"] < 0.1
    # E's weak pass climbs above its strict pass.
    assert e["weak_pass"] - e["strict_pass"] > 0.3
    # Transfer: family I (never trained) also shows a gap in E.
    assert e["by_family"]["I"]["delta_hack"] > 0.2
