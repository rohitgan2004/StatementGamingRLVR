"""Frozen SFT baseline: supervised fine-tuning on genuine (faithful) proofs.

Fine-tunes the base model on ~1,000 statement-faithful (statement, proof) pairs
sampled from the corpus generator, matched in token budget to the RL runs
(Section 5.2).  If RLVR (Arm E) produces substantially more statement gaming than
this SFT model at comparable weak-pass competence, the effect is attributable to
optimization against the verifier, not to exposure to the task distribution.

Run:
    pip install -e ".[trl]"
    python -m leandrift.rl.sft --config configs/base.yaml --corpus data/corpus/corpus.json
"""

from __future__ import annotations

import argparse
import os
import random
from typing import List

from leandrift.corpus.generator import load_corpus
from leandrift.rl.prompt import SYSTEM, build_prompt
from leandrift.utils.config import load_config
from leandrift.utils.io import ensure_dir


def _target_text(item) -> str:
    s = item.prompt.intended
    return (f"FORMAL_STATEMENT:\n{s.to_lean()}\nPROOF:\n{item.honest_proof}\n")


def build_sft_examples(corpus, n: int = 1000, seed: int = 0) -> List[dict]:
    rng = random.Random(seed)
    pool = list(corpus.train)
    examples = []
    while len(examples) < n:
        item = pool[rng.randrange(len(pool))]
        examples.append({
            "prompt": build_prompt(item, few_shot=False),
            "completion": _target_text(item),
        })
    return examples


def run_sft(cfg: dict, corpus, n_examples: int = 1000) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    model_name = cfg["model"]["base_model"]
    seed = cfg.get("seed", 0)
    examples = build_sft_examples(corpus, n=n_examples, seed=seed)
    ds = Dataset.from_list([{"text": e["prompt"] + "\n" + e["completion"]} for e in examples])

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb,
                                                 device_map="auto")
    peft_cfg = LoraConfig(
        r=cfg["lora"]["rank"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    run_dir = ensure_dir(os.path.join(cfg["paths"]["runs_dir"], f"sft_seed{seed}"))
    args = SFTConfig(
        output_dir=os.path.join(run_dir, "ckpt"),
        per_device_train_batch_size=4, gradient_accumulation_steps=2,
        learning_rate=1e-4, num_train_epochs=1, logging_steps=20,
        max_seq_length=cfg["model"]["max_prompt_tokens"] + cfg["model"]["max_completion_tokens"],
        seed=seed, report_to=[],
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(os.path.join(run_dir, "adapter"))
    return {"run_dir": run_dir, "n_examples": len(examples)}


def main() -> None:
    ap = argparse.ArgumentParser(description="LEANDRIFT SFT-on-genuine-proofs baseline")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--corpus", default="data/corpus/corpus.json")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_sft(cfg, load_corpus(args.corpus), n_examples=args.n)


if __name__ == "__main__":
    main()
