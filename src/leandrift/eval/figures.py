"""Regenerate Figure 2 and Tables 4/5 from logged runs (the `make reproduce` sink).

Reads runs/<name>/metrics.jsonl + summary.json and writes:
  figures/figure2_dynamics.png   -- training dynamics (Figure 2)
  outputs/table4_results.{md,tex,json}  -- final-checkpoint results (Table 4)
  outputs/table5_transfer.{md,json}     -- cross-family transfer (Table 5)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from leandrift.utils.io import ensure_dir, read_json, read_jsonl, write_json


def _run_dir(cfg: dict, name: str) -> str:
    return os.path.join(cfg["paths"]["runs_dir"], name)


def _load_history(cfg: dict, name: str) -> List[dict]:
    path = os.path.join(_run_dir(cfg, name), "metrics.jsonl")
    return list(read_jsonl(path)) if os.path.exists(path) else []


def _final(cfg: dict, name: str) -> Optional[dict]:
    path = os.path.join(_run_dir(cfg, name), "summary.json")
    if not os.path.exists(path):
        hist = _load_history(cfg, name)
        return hist[-1] if hist else None
    return read_json(path).get("final")


def _arm_run(cfg: dict, arm: str) -> str:
    return f"arm_{arm}_seed{cfg.get('seed', 0)}"


def _ema(ys: List[float], alpha: float = 0.4) -> List[float]:
    """Exponential moving average to display the trend under Monte-Carlo eval noise."""
    if not ys:
        return ys
    out = [ys[0]]
    for y in ys[1:]:
        out.append(alpha * y + (1 - alpha) * out[-1])
    return out


def make_figure2(cfg: dict, outdir: str) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = _load_history(cfg, _arm_run(cfg, "E"))
    b = _load_history(cfg, _arm_run(cfg, "B"))
    if not e:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    steps = [r["step"] for r in e]

    def _series(xs, ys, style, color, label):
        # Faint raw points (per-checkpoint estimate) + a smoothed trend line.
        ax.plot(xs, ys, marker=style[-1], linestyle="none", color=color, alpha=0.22, markersize=4)
        ax.plot(xs, _ema(ys), style, color=color, label=label, markersize=4, linewidth=1.8)

    _series(steps, [r["weak_pass"] for r in e], "-o", "#c0392b", "weak pass (Arm E)")
    _series(steps, [r["strict_pass"] for r in e], "-s", "#2980b9", "strict pass (Arm E)")
    _series(steps, [r["delta_hack"] for r in e], "--^", "#8e44ad", r"$\Delta_{hack}$ (Arm E)")
    if b:
        _series([r["step"] for r in b], [r["strict_pass"] for r in b], "-d",
                "#27ae60", "strict pass (Arm B)")
    ax.set_xlabel("RL step")
    ax.set_ylabel("pass rate / gap")
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("Training dynamics in LEANDRIFT (simulation)")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "figure2_dynamics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _pct(x: Optional[float]) -> str:
    return f"{100 * x:.1f}" if x is not None else "--"


def make_table4(cfg: dict) -> Dict:
    rows = [
        ("Pre-RL base", _final(cfg, "baseline_preRL")),
        ("SFT on genuine proofs", _final(cfg, "baseline_SFT")),
        ("Arm B (locked statement)", _final(cfg, _arm_run(cfg, "B"))),
        ("Arm E (exploitable)", _final(cfg, _arm_run(cfg, "E"))),
        ("Arm H (faithfulness-gated)", _final(cfg, _arm_run(cfg, "H"))),
        ("Arm H' (pattern penalty)", _final(cfg, _arm_run(cfg, "Hp"))),
    ]
    table = []
    for label, rec in rows:
        if rec is None:
            continue
        table.append({
            "condition": label,
            "weak_pass": rec["weak_pass"],
            "strict_pass": rec["strict_pass"],
            "delta_hack": rec["delta_hack"],
            "faithfulness": rec["faithfulness"],
        })
    return {"rows": table}


def make_table5(cfg: dict) -> Dict:
    rec = _final(cfg, _arm_run(cfg, "E"))
    if rec is None or "by_family" not in rec:
        return {"rows": []}
    fam = rec["by_family"]
    rows = []
    names = {"D": "D (trained family, held-out items)", "I": "I (never-trained family)"}
    for f in ("D", "I"):
        if f in fam:
            rows.append({
                "family": names[f],
                "weak_pass": fam[f]["weak_pass"],
                "strict_pass": fam[f]["strict_pass"],
                "delta_hack": fam[f]["delta_hack"],
            })
    return {"rows": rows}


def _table4_md(t4: Dict) -> str:
    lines = ["| Condition | Weak pass | Strict pass | Δhack | Faithfulness |",
             "|---|---|---|---|---|"]
    for r in t4["rows"]:
        lines.append(f"| {r['condition']} | {_pct(r['weak_pass'])} | {_pct(r['strict_pass'])} "
                     f"| {r['delta_hack']:.3f} | {_pct(r['faithfulness'])} |")
    return "\n".join(lines) + "\n"


def _table5_md(t5: Dict) -> str:
    lines = ["| Evaluation family | Weak pass | Strict pass | Δhack |",
             "|---|---|---|---|"]
    for r in t5["rows"]:
        lines.append(f"| {r['family']} | {_pct(r['weak_pass'])} | {_pct(r['strict_pass'])} "
                     f"| {r['delta_hack']:.3f} |")
    return "\n".join(lines) + "\n"


def make_all(cfg: dict) -> str:
    figdir = ensure_dir("figures")
    outdir = ensure_dir(cfg["paths"]["outputs_dir"])
    make_figure2(cfg, figdir)

    t4 = make_table4(cfg)
    t5 = make_table5(cfg)
    write_json(os.path.join(outdir, "table4_results.json"), t4)
    write_json(os.path.join(outdir, "table5_transfer.json"), t5)
    with open(os.path.join(outdir, "table4_results.md"), "w") as f:
        f.write("# Table 4: Final-checkpoint results (simulation)\n\n")
        f.write(_table4_md(t4))
    with open(os.path.join(outdir, "table5_transfer.md"), "w") as f:
        f.write("# Table 5: Cross-family transfer, Arm E (simulation)\n\n")
        f.write(_table5_md(t5))
    return outdir
