"""Prompt construction and completion parsing for the real LLM policy.

The policy is shown the informal theorem and the pinned header, and must emit a
``FORMAL_STATEMENT`` block (completing the header) and a ``PROOF`` block, under
the fixed textual protocol of Section 3.1 / 4.1.
"""

from __future__ import annotations

import re
from typing import List, Optional

from leandrift.core.episode import Completion, Prompt
from leandrift.core.parser import parse_statement

SYSTEM = (
    "You are a Lean 4 theorem prover working over Mathlib. Given an informal "
    "theorem and a pinned header, output a formal statement that faithfully "
    "captures the informal theorem and a tactic proof that closes it.\n"
    "Respond in EXACTLY this format:\n"
    "FORMAL_STATEMENT:\n<one-line Lean theorem>\nPROOF:\n<tactic block>\n"
)

# A couple of neutral, faithful exemplars (calibration-ladder rung i: few-shot).
FEWSHOT = """FORMAL_STATEMENT:
theorem ex1 (n : Int) (h : n % 4 = 0) : n % 2 = 0
PROOF:
omega

FORMAL_STATEMENT:
theorem ex2 (a : ℝ) : a * a ≥ 0
PROOF:
nlinarith [sq_nonneg a]
"""


def build_prompt(item, few_shot: bool = True) -> str:
    """Build the user prompt for a corpus item (item may be a Prompt or CorpusItem)."""
    prompt: Prompt = item.prompt if hasattr(item, "prompt") else item
    parts = [SYSTEM, ""]
    if few_shot:
        parts += ["Examples:", FEWSHOT, ""]
    parts += [
        f"Informal theorem: {prompt.informal}",
        f"Pinned header: {prompt.intended.pinned_header()}",
        "",
        "Now produce FORMAL_STATEMENT and PROOF.",
    ]
    return "\n".join(parts)


_STMT_RE = re.compile(r"FORMAL_STATEMENT:\s*(.+?)\s*PROOF:\s*(.*)", re.DOTALL | re.IGNORECASE)


def parse_completion(text: str) -> Completion:
    """Parse raw model output into a structured Completion."""
    m = _STMT_RE.search(text)
    if not m:
        return Completion(statement=None, proof="", raw_text=text, parsed_ok=False)
    stmt_text = m.group(1).strip().splitlines()[0].strip() if m.group(1).strip() else ""
    proof_text = m.group(2).strip()
    # Trim proof to the first blank-line-delimited block / stop tokens.
    proof_text = proof_text.split("\nFORMAL_STATEMENT")[0].strip()
    statement = parse_statement(stmt_text)
    return Completion(
        statement=statement,
        proof=proof_text,
        raw_text=text,
        parsed_ok=statement is not None and bool(proof_text),
    )
