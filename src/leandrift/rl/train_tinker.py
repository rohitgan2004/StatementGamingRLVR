"""Real GRPO training on the Thinking Machines Tinker API.

This mirrors the Tinker cookbook's minimal GRPO loop
(``tinker_cookbook/recipes/rl_loop.py``) but swaps the GSM8K reward for the
LEANDRIFT per-arm reward: each sampled (FORMAL_STATEMENT, PROOF) is parsed,
verified against Lean (the warm REPL pool runs on THIS driver machine), and
scored by the arm's reward.  Tinker handles distributed LoRA training/sampling of
Qwen2.5; the statement-gaming reward loop stays entirely on your side -- exactly
the split the paper's threat model targets (the harness, not the kernel).

Run:
    export TINKER_API_KEY=...            # from the Tinker console
    python -m leandrift.rl.train_tinker --config configs/arm_E.yaml \
        --corpus data/corpus/corpus.json

Requires:  pip install -e ".[tinker]"  and a Lean backend (lean.backend: repl).
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import List

from leandrift.corpus.generator import CorpusItem, load_corpus
from leandrift.eval.metrics import compute
from leandrift.lean.backend import get_backend
from leandrift.rl.prompt import build_prompt, parse_completion
from leandrift.rl.rollout import rollout
from leandrift.utils.config import apply_overrides, load_config
from leandrift.utils.io import append_jsonl, ensure_dir, write_json


def _make_convo(item: CorpusItem, arm: dict):
    return [{"role": "user", "content": build_prompt(item, few_shot=True)}]


def run_tinker(cfg: dict, corpus, base_url: str | None = None) -> dict:
    # Imports deferred so the package works without the GPU extras installed.
    import tinker
    import torch
    from tinker import types
    from tinker.types.tensor_data import TensorData
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    arm = cfg["arm"]
    grpo = cfg["grpo"]
    shaping = cfg["reward_shaping"]
    tcfg = cfg["training"]
    seed = cfg.get("seed", 0)
    rng = random.Random(seed)
    related_threshold = cfg["verifier"]["related_jaccard_threshold"]

    model_name = cfg["model"]["base_model"]
    group_size = grpo["group_size"]
    prompts_per_step = grpo["prompts_per_step"]
    steps = tcfg["steps"]
    eval_every = tcfg["eval_every"]
    save_every = tcfg["checkpoint_every"]

    backend = get_backend(cfg)  # REPL pool (or mock) on this machine

    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = service_client.create_lora_training_client(
        base_model=model_name, rank=cfg["lora"]["rank"]
    )
    sampling_params = tinker.types.SamplingParams(
        max_tokens=cfg["model"]["max_completion_tokens"],
        temperature=cfg["sampling"]["temperature"],
        top_p=cfg["sampling"]["top_p"],
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=grpo["learning_rate"], beta1=grpo["adam_beta1"],
        beta2=grpo["adam_beta2"], eps=grpo["adam_eps"],
    )

    run_dir = ensure_dir(os.path.join(cfg["paths"]["runs_dir"], f"arm_{arm['name']}_tinker_seed{seed}"))
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    def evaluate(sampling_client, step: int) -> None:
        episodes = []
        eval_params = tinker.types.SamplingParams(
            max_tokens=cfg["model"]["max_completion_tokens"],
            temperature=cfg["sampling"]["temperature"], top_p=cfg["sampling"]["top_p"],
            stop=renderer.get_stop_sequences(),
        )
        futures = []
        for item in corpus.eval:
            mi = renderer.build_generation_prompt(_make_convo(item, arm))
            futures.append((item, mi, sampling_client.sample(
                prompt=mi, num_samples=1, sampling_params=eval_params)))
        for item, _mi, fut in futures:
            seq = fut.result().sequences[0]
            msg, _ = renderer.parse_response(seq.tokens)
            text = renderers.get_text_content(msg)
            comp = parse_completion(text)
            episodes.append(rollout(item.prompt, comp, backend, arm, shaping,
                                    related_threshold=related_threshold, step=step,
                                    with_reward=False))
        m = compute(episodes)
        append_jsonl(metrics_path, {"step": step, **m.to_dict()})

    for step in range(steps):
        if step % eval_every == 0:
            evaluate(training_client.save_weights_and_get_sampling_client(), step)
        if save_every and step > 0 and step % save_every == 0:
            training_client.save_state(name=f"{step:06d}")

        sampling_client = training_client.save_weights_and_get_sampling_client()
        batch = [corpus.train[rng.randrange(len(corpus.train))] for _ in range(prompts_per_step)]

        datums = []
        prompts, futures = [], []
        for item in batch:
            mi = renderer.build_generation_prompt(_make_convo(item, arm))
            prompts.append((item, mi))
            futures.append(sampling_client.sample(
                prompt=mi, num_samples=group_size, sampling_params=sampling_params))

        step_rewards: List[float] = []
        for (item, prompt), future in zip(prompts, futures):
            res = future.result()
            rewards_G, toks_G, lps_G = [], [], []
            for seq in res.sequences:
                msg, _ = renderer.parse_response(seq.tokens)
                text = renderers.get_text_content(msg)
                comp = parse_completion(text)
                ep = rollout(item.prompt, comp, backend, arm, shaping,
                             related_threshold=related_threshold, step=step, with_reward=True)
                rewards_G.append(ep.reward)
                toks_G.append(seq.tokens)
                lps_G.append(seq.logprobs)
            mean_r = sum(rewards_G) / len(rewards_G)
            step_rewards.append(mean_r)
            advs = [r - mean_r for r in rewards_G]
            if all(a == 0.0 for a in advs):
                continue
            for sampled_tokens, logprobs, advantage in zip(toks_G, lps_G, advs):
                ob_len = prompt.length - 1
                model_input = prompt.append(types.EncodedTextChunk(tokens=sampled_tokens[:-1]))
                target_tokens = [0] * ob_len + sampled_tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
                datums.append(types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                ))

        if datums:
            fb = training_client.forward_backward(datums, loss_fn="importance_sampling")
            opt = training_client.optim_step(adam_params)
            fb.result()
            opt.result()
        print(f"[tinker] step {step} mean_reward={sum(step_rewards)/len(step_rewards):.3f} "
              f"datums={len(datums)}")

    evaluate(training_client.save_weights_and_get_sampling_client(), steps)
    training_client.save_state(name="final")

    from leandrift.utils.io import read_jsonl
    history = list(read_jsonl(metrics_path))
    write_json(os.path.join(run_dir, "summary.json"),
               {"arm": arm["name"], "seed": seed, "simulated": False,
                "final": history[-1] if history else {}})
    backend.close()
    return {"run_dir": run_dir}


def main() -> None:
    ap = argparse.ArgumentParser(description="LEANDRIFT GRPO on Tinker")
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus", default="data/corpus/corpus.json")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg = apply_overrides(cfg, {"training.steps": args.steps})
    corpus = load_corpus(args.corpus)
    run_tinker(cfg, corpus, base_url=args.base_url)


if __name__ == "__main__":
    main()
