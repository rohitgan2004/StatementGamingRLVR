"""Real base-model corpus calibration (Section 4.4).

Measures, for each candidate template instance, the base model's honest pass rate
on the intended statement (must land in [0.15, 0.55]) and the best pass rate over
the Table 1 weakenings (drift advantage, must reach >= 0.85), then builds a corpus
retaining only calibrated items.  This is the real analogue of the assigned-rate
simulation calibrator in ``generator.py``.

The fallback ladder (few-shot exemplars -> restricted tactic subset -> Lean-exposed
base model) is applied by re-running with different flags; the first rung that
puts enough items in band is fixed for all arms and reported in the meta block.

Run:
    pip install -e ".[trl]"
    python -m leandrift.corpus.calibrate --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from typing import Callable, List

from leandrift.corpus.templates import Instance, enumerate_instances
from leandrift.corpus.weakenings import all_weakenings
from leandrift.lean.backend import LeanBackend, get_backend
from leandrift.rl.prompt import build_prompt, parse_completion
from leandrift.utils.config import load_config


def _hf_sampler(model_name: str, cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb,
                                                 device_map="auto")

    def sample(prompt: str, n: int) -> List[str]:
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=cfg["model"]["max_prompt_tokens"]).to(model.device)
        outs = model.generate(
            **inputs, max_new_tokens=cfg["model"]["max_completion_tokens"],
            do_sample=True, num_return_sequences=n,
            temperature=cfg["sampling"]["temperature"], top_p=cfg["sampling"]["top_p"],
            pad_token_id=tok.eos_token_id,
        )
        return [tok.decode(o[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                for o in outs]

    return sample


def honest_pass_rate(inst: Instance, sampler, backend: LeanBackend, cfg: dict,
                     n: int = 8) -> float:
    """Fraction of base-model attempts that close the *intended* statement."""
    from leandrift.core.episode import Prompt

    prompt = Prompt(id=inst.statement.name, family=inst.family, informal=inst.informal,
                    intended=inst.statement, template=inst.template_id)
    texts = sampler(build_prompt(prompt, few_shot=True), n)
    passes = 0
    for t in texts:
        comp = parse_completion(t)
        if comp.statement is None:
            continue
        # Honest = the model's statement must equal the intended one.
        if comp.statement.obligation() != inst.statement.obligation():
            continue
        res = backend.check(inst.statement.to_lean(), comp.proof, statement=inst.statement)
        passes += int(res.kernel_ok)
    return passes / n


def drift_advantage(inst: Instance, backend: LeanBackend) -> float:
    best = 0.0
    for w in all_weakenings(inst.statement):
        res = backend.check(w.statement.to_lean(), w.proof, statement=w.statement)
        best = max(best, 1.0 if res.kernel_ok else 0.0)
    return best


def calibrate(cfg: dict, out: str) -> dict:
    from leandrift.corpus.generator import generate, save_corpus

    backend = get_backend(cfg)
    sampler = _hf_sampler(cfg["model"]["base_model"], cfg)
    cache = {}

    def rate_fn(inst: Instance) -> float:
        if inst.statement.name not in cache:
            cache[inst.statement.name] = honest_pass_rate(inst, sampler, backend, cfg)
        return cache[inst.statement.name]

    corpus = generate(cfg, backend=backend, rate_fn=rate_fn)
    corpus.meta["simulated"] = False
    corpus.meta["base_model"] = cfg["model"]["base_model"]
    save_corpus(corpus, out)
    backend.close()
    return corpus.meta


def main() -> None:
    ap = argparse.ArgumentParser(description="LEANDRIFT real corpus calibration")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="data/corpus/corpus.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    meta = calibrate(cfg, args.out)
    print(f"[calibrate] {meta}")


if __name__ == "__main__":
    main()
