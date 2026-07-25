"""Real GRPO training with TRL's GRPOTrainer + QLoRA (the paper's single-A10G path).

Trains Qwen2.5-{Coder,Math}-1.5B with 4-bit QLoRA and TRL GRPO.  The reward
function parses each completion, verifies it against Lean (warm REPL pool), and
applies the LEANDRIFT per-arm reward.  A callback logs weak/strict/delta_hack on
the held-out set every `eval_every` steps to ``metrics.jsonl`` (same schema the
figures script consumes).

Run:
    pip install -e ".[trl]"
    python -m leandrift.rl.train_trl --config configs/arm_E.yaml \
        --corpus data/corpus/corpus.json
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

from leandrift.corpus.generator import load_corpus
from leandrift.eval.metrics import compute
from leandrift.lean.backend import get_backend
from leandrift.rl.prompt import build_prompt, parse_completion
from leandrift.rl.rollout import rollout
from leandrift.utils.config import apply_overrides, load_config
from leandrift.utils.io import append_jsonl, ensure_dir, write_json


def run_trl(cfg: dict, corpus) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from trl import GRPOConfig, GRPOTrainer

    arm = cfg["arm"]
    shaping = cfg["reward_shaping"]
    grpo = cfg["grpo"]
    seed = cfg.get("seed", 0)
    related_threshold = cfg["verifier"]["related_jaccard_threshold"]
    model_name = cfg["model"]["base_model"]

    backend = get_backend(cfg)
    id2item = {it.prompt.id: it for it in corpus.train}

    # Dataset: one row per training prompt.
    rows = [{"prompt": build_prompt(it, few_shot=True), "item_id": it.prompt.id}
            for it in corpus.train]
    train_ds = Dataset.from_list(rows)

    def reward_fn(prompts, completions, **kwargs) -> List[float]:
        item_ids = kwargs["item_id"]
        out: List[float] = []
        for comp_text, iid in zip(completions, item_ids):
            item = id2item[iid]
            comp = parse_completion(comp_text if isinstance(comp_text, str)
                                    else comp_text[0]["content"])
            ep = rollout(item.prompt, comp, backend, arm, shaping,
                         related_threshold=related_threshold, with_reward=True)
            out.append(ep.reward)
        return out

    # QLoRA 4-bit.
    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["quantization"]["bnb_4bit_compute_dtype"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb,
                                                 device_map="auto")
    peft_cfg = LoraConfig(
        r=cfg["lora"]["rank"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )

    run_dir = ensure_dir(os.path.join(cfg["paths"]["runs_dir"], f"arm_{arm['name']}_trl_seed{seed}"))
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    args = GRPOConfig(
        output_dir=os.path.join(run_dir, "ckpt"),
        per_device_train_batch_size=grpo["prompts_per_step"],
        num_generations=grpo["group_size"],
        learning_rate=grpo["learning_rate"],
        lr_scheduler_type=grpo["lr_schedule"],
        warmup_ratio=grpo["warmup_ratio"],
        beta=grpo["kl_coef"],
        epsilon=grpo["clip_range"],
        max_prompt_length=cfg["model"]["max_prompt_tokens"],
        max_completion_length=cfg["model"]["max_completion_tokens"],
        temperature=cfg["sampling"]["temperature"],
        top_p=cfg["sampling"]["top_p"],
        max_steps=cfg["training"]["steps"],
        logging_steps=10,
        save_steps=cfg["training"]["checkpoint_every"],
        seed=seed,
        report_to=[],
    )

    class EvalCallback(TrainerCallback):
        def on_step_end(self, args_, state, control, **kw):
            if state.global_step % cfg["training"]["eval_every"] != 0:
                return
            episodes = []
            for it in corpus.eval:
                text = _generate(model, tokenizer, build_prompt(it, few_shot=True), cfg)
                comp = parse_completion(text)
                episodes.append(rollout(it.prompt, comp, backend, arm, shaping,
                                        related_threshold=related_threshold,
                                        step=state.global_step, with_reward=False))
            append_jsonl(metrics_path, {"step": state.global_step, **compute(episodes).to_dict()})

    trainer = GRPOTrainer(
        model=model, reward_funcs=[reward_fn], args=args,
        train_dataset=train_ds, peft_config=peft_cfg,
        processing_class=tokenizer, callbacks=[EvalCallback()],
    )
    trainer.train()
    trainer.save_model(os.path.join(run_dir, "adapter"))

    from leandrift.utils.io import read_jsonl
    history = list(read_jsonl(metrics_path)) if os.path.exists(metrics_path) else []
    write_json(os.path.join(run_dir, "summary.json"),
               {"arm": arm["name"], "seed": seed, "simulated": False,
                "final": history[-1] if history else {}})
    backend.close()
    return {"run_dir": run_dir}


def _generate(model, tokenizer, prompt: str, cfg: dict) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=cfg["model"]["max_prompt_tokens"]).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=cfg["model"]["max_completion_tokens"],
            do_sample=True, temperature=cfg["sampling"]["temperature"],
            top_p=cfg["sampling"]["top_p"], pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="LEANDRIFT GRPO on TRL (single GPU)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus", default="data/corpus/corpus.json")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg = apply_overrides(cfg, {"training.steps": args.steps})
    run_trl(cfg, load_corpus(args.corpus))


if __name__ == "__main__":
    main()
