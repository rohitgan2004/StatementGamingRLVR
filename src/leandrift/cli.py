"""LEANDRIFT command-line interface.

Subcommands:
  gen-corpus           build the difficulty-calibrated corpus
  validate-driftclass  score DRIFTCLASS against the exact oracle (Appendix A)
  run-sim              run one arm with the simulation policy (local, no GPU)
  baselines           run pre-RL and SFT baselines (simulation)
  figures              regenerate Figure 2 and Tables 4/5 from logged runs
  reproduce           end-to-end simulation of all arms + baselines + figures
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from leandrift.utils.config import apply_overrides, load_config
from leandrift.utils.io import ensure_dir, write_json


def _cmd_gen_corpus(args) -> None:
    from leandrift.corpus.generator import generate, save_corpus

    cfg = load_config(args.config)
    corpus = generate(cfg)
    out = args.out or os.path.join(cfg["paths"]["corpus_dir"], "corpus.json")
    save_corpus(corpus, out)
    print(f"[corpus] wrote {out}")
    print(f"[corpus] train={len(corpus.train)} eval={len(corpus.eval)} meta={corpus.meta}")


def _cmd_validate_driftclass(args) -> None:
    from leandrift.microproof.validate import validate

    rep = validate(seed=args.seed)
    out = args.out or "outputs/driftclass_validation.json"
    write_json(out, {
        "agreement": rep.agreement, "precision": rep.precision, "recall": rep.recall,
        "tp": rep.tp, "fp": rep.fp, "tn": rep.tn, "fn": rep.fn,
        "n_instances": rep.n_instances, "n_candidates": rep.n_candidates,
        "disagreements": rep.disagreements[:50],
    })
    print(f"[driftclass] agreement={rep.agreement:.4f} precision={rep.precision:.4f} "
          f"recall={rep.recall:.4f} over {rep.n_candidates} candidates -> {out}")


def _load_or_build_corpus(cfg, corpus_path: Optional[str]):
    from leandrift.corpus.generator import generate, load_corpus, save_corpus

    path = corpus_path or os.path.join(cfg["paths"]["corpus_dir"], "corpus.json")
    if os.path.exists(path):
        return load_corpus(path)
    corpus = generate(cfg)
    save_corpus(corpus, path)
    return corpus


def _cmd_run_sim(args) -> None:
    from leandrift.rl.train_sim import run_sim

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg = apply_overrides(cfg, {"training.steps": args.steps})
    if args.seed is not None:
        cfg = apply_overrides(cfg, {"seed": args.seed})
    corpus = _load_or_build_corpus(cfg, args.corpus)
    res = run_sim(cfg, corpus)
    print(f"[run-sim] arm {cfg['arm']['name']} final: {res['summary']['final']}")


def _cmd_baselines(args) -> None:
    from leandrift.rl.baselines import run_baselines

    cfg = load_config(args.config)
    corpus = _load_or_build_corpus(cfg, args.corpus)
    res = run_baselines(cfg, corpus)
    print(f"[baselines] {res}")


def _cmd_figures(args) -> None:
    from leandrift.eval.figures import make_all

    cfg = load_config(args.config)
    outdir = make_all(cfg)
    print(f"[figures] wrote figures/tables to {outdir}")


def _cmd_reproduce(args) -> None:
    from leandrift.rl.train_sim import run_sim
    from leandrift.rl.baselines import run_baselines
    from leandrift.eval.figures import make_all

    base_cfg = load_config("configs/base.yaml")
    corpus = _load_or_build_corpus(base_cfg, args.corpus)

    arms = ["B", "E", "H", "Hp"]
    for arm in arms:
        cfg = load_config(f"configs/arm_{arm}.yaml")
        if args.steps is not None:
            cfg = apply_overrides(cfg, {"training.steps": args.steps})
        res = run_sim(cfg, corpus)
        print(f"[reproduce] arm {arm} final: {res['summary']['final']}")

    run_baselines(base_cfg, corpus)
    outdir = make_all(base_cfg)
    print(f"[reproduce] done. figures/tables in {outdir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="leandrift", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-corpus")
    g.add_argument("--config", default="configs/base.yaml")
    g.add_argument("--out", default=None)
    g.set_defaults(func=_cmd_gen_corpus)

    v = sub.add_parser("validate-driftclass")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--out", default=None)
    v.set_defaults(func=_cmd_validate_driftclass)

    r = sub.add_parser("run-sim")
    r.add_argument("--config", required=True)
    r.add_argument("--corpus", default=None)
    r.add_argument("--steps", type=int, default=None)
    r.add_argument("--seed", type=int, default=None)
    r.set_defaults(func=_cmd_run_sim)

    b = sub.add_parser("baselines")
    b.add_argument("--config", default="configs/base.yaml")
    b.add_argument("--corpus", default=None)
    b.set_defaults(func=_cmd_baselines)

    f = sub.add_parser("figures")
    f.add_argument("--config", default="configs/base.yaml")
    f.set_defaults(func=_cmd_figures)

    rp = sub.add_parser("reproduce")
    rp.add_argument("--corpus", default=None)
    rp.add_argument("--steps", type=int, default=None)
    rp.set_defaults(func=_cmd_reproduce)

    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
