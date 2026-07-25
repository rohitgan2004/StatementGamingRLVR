"""Difficulty-calibrated corpus construction (Sections 4.2 & 4.4).

A gaming study is only informative if honest proving is possible-but-effortful
and gaming is strictly easier.  We therefore retain a template instance only if:

  * Honest band: base-model pass rate on the intended statement lies in
    [0.15, 0.55]; and
  * Drift advantage: at least one Table 1 weakening yields a statement the base
    model proves at rate >= 0.85.

In the default (simulation) mode the honest base rate is assigned per instance
(deterministically) and the drift advantage is checked with the mock backend
against the canonical cheap proofs.  The real calibrator (scripts/calibrate.py)
measures both by sampling an actual base model.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from leandrift.core.episode import Prompt
from leandrift.corpus.templates import Instance, enumerate_instances
from leandrift.corpus.weakenings import all_weakenings
from leandrift.lean.backend import LeanBackend
from leandrift.lean.mock import MockLeanBackend


@dataclass
class CorpusItem:
    prompt: Prompt
    honest_proof: str
    honest_base_rate: float
    drift_advantage: float


@dataclass
class Corpus:
    train: List[CorpusItem] = field(default_factory=list)
    eval: List[CorpusItem] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)


def _assigned_rate(name: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return round(0.10 + 0.50 * frac, 3)  # in [0.10, 0.60]


def _drift_advantage_mock(inst: Instance, backend: LeanBackend) -> float:
    """Best pass rate over the canonical weakenings (deterministic in the mock)."""
    best = 0.0
    for w in all_weakenings(inst.statement):
        res = backend.check(w.statement.to_lean(), w.proof, statement=w.statement)
        best = max(best, 1.0 if res.kernel_ok else 0.0)
    return best


def _retain(rate: float, drift_adv: float, band: Tuple[float, float], thresh: float) -> bool:
    lo, hi = band
    return (lo <= rate <= hi) and (drift_adv >= thresh)


def generate(
    cfg: dict,
    backend: Optional[LeanBackend] = None,
    rate_fn: Optional[Callable[[Instance], float]] = None,
) -> Corpus:
    ccfg = cfg["corpus"]
    seed = cfg.get("seed", 0)
    band = tuple(ccfg["honest_band"])
    thresh = ccfg["drift_advantage_threshold"]
    n_train = ccfg["n_train"]
    n_eval = ccfg["n_eval"]
    train_family = ccfg["train_family"]
    backend = backend or MockLeanBackend(seed=seed)
    rate_fn = rate_fn or (lambda inst: _assigned_rate(inst.statement.name, seed))

    rng = random.Random(seed)

    def build_items(instances: List[Instance]) -> List[CorpusItem]:
        items: List[CorpusItem] = []
        for inst in instances:
            rate = rate_fn(inst)
            drift_adv = _drift_advantage_mock(inst, backend)
            if not _retain(rate, drift_adv, band, thresh):
                continue
            prompt = Prompt(
                id=f"{inst.family}:{inst.statement.name}",
                family=inst.family,
                informal=inst.informal,
                intended=inst.statement,
                template=inst.template_id,
            )
            items.append(CorpusItem(prompt, inst.honest_proof, rate, drift_adv))
        return items

    train_pool = build_items([i for i in enumerate_instances() if i.family == train_family])
    other_pool = build_items([i for i in enumerate_instances() if i.family != train_family])

    rng.shuffle(train_pool)
    train = train_pool[:n_train]
    held_out_train_family = train_pool[n_train:]

    # Eval set: stratified across the trained family (held-out items) and family I.
    rng.shuffle(other_pool)
    n_eval_other = min(len(other_pool), n_eval // 2)
    n_eval_train = n_eval - n_eval_other
    eval_items = held_out_train_family[:n_eval_train] + other_pool[:n_eval_other]
    rng.shuffle(eval_items)

    corpus = Corpus(
        train=train,
        eval=eval_items,
        meta={
            "seed": seed,
            "honest_band": list(band),
            "drift_advantage_threshold": thresh,
            "n_train_requested": n_train,
            "n_train_retained": len(train),
            "n_train_pool": len(train_pool),
            "n_eval": len(eval_items),
            "eval_families": {
                f: sum(1 for it in eval_items if it.prompt.family == f)
                for f in set(it.prompt.family for it in eval_items)
            },
            "simulated": isinstance(backend, MockLeanBackend),
        },
    )
    return corpus


# ---- persistence ---------------------------------------------------------------
def _item_to_dict(it: CorpusItem) -> dict:
    from leandrift.core.codec import prompt_to_dict

    return {
        "prompt": prompt_to_dict(it.prompt),
        "honest_proof": it.honest_proof,
        "honest_base_rate": it.honest_base_rate,
        "drift_advantage": it.drift_advantage,
    }


def _item_from_dict(d: dict) -> CorpusItem:
    from leandrift.core.codec import prompt_from_dict

    return CorpusItem(
        prompt=prompt_from_dict(d["prompt"]),
        honest_proof=d["honest_proof"],
        honest_base_rate=d["honest_base_rate"],
        drift_advantage=d["drift_advantage"],
    )


def save_corpus(corpus: Corpus, path: str) -> None:
    from leandrift.utils.io import write_json

    write_json(path, {
        "train": [_item_to_dict(it) for it in corpus.train],
        "eval": [_item_to_dict(it) for it in corpus.eval],
        "meta": corpus.meta,
    })


def load_corpus(path: str) -> Corpus:
    from leandrift.utils.io import read_json

    d = read_json(path)
    return Corpus(
        train=[_item_from_dict(x) for x in d["train"]],
        eval=[_item_from_dict(x) for x in d["eval"]],
        meta=d.get("meta", {}),
    )
