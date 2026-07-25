"""Lean backend interface + factory.

A backend answers one question: does this proof close this statement in a clean,
pinned Lean environment, and with what hygiene metadata?  Two implementations:

  * ``repl``  -- a warm pool of persistent Lean 4 REPL workers over Mathlib
                (the production substrate; used for all real runs).
  * ``mock``  -- a semantic evaluator over the divisibility / inequality
                fragments that reproduces Lean's accept/reject behavior for the
                template families *without* Lean, so the full pipeline
                (verifiers, reward, GRPO loop, metrics, figures) runs locally on
                a laptop with no GPU.  Results are clearly marked ``simulated``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from leandrift.core.statement import Statement


@dataclass
class LeanResult:
    kernel_ok: bool                       # V(s, pi): proof closes, kernel accepts
    sorries: List[str] = field(default_factory=list)
    axioms: Optional[List[str]] = None    # #print axioms output (None if unknown)
    error: str = ""
    elapsed_s: float = 0.0
    timed_out: bool = False
    simulated: bool = False


class LeanBackend:
    """Abstract backend."""

    def check(
        self,
        statement_source: str,
        proof_source: str,
        statement: Optional[Statement] = None,
    ) -> LeanResult:
        raise NotImplementedError  # pragma: no cover

    def close(self) -> None:
        pass


def get_backend(cfg: dict) -> LeanBackend:
    """Instantiate the backend named by ``cfg['lean']['backend']``."""
    lean_cfg = cfg.get("lean", {})
    kind = lean_cfg.get("backend", "mock")
    if kind == "mock":
        from leandrift.lean.mock import MockLeanBackend

        return MockLeanBackend()
    if kind == "repl":
        from leandrift.lean.pool import ReplPool

        return ReplPool(
            n_workers=cfg.get("verifier", {}).get("n_repl_workers", 8),
            timeout_s=cfg.get("verifier", {}).get("per_episode_timeout_s", 40),
            repl_path=lean_cfg.get("repl_path"),
            project_path=lean_cfg.get("project_path", "lean"),
            memoize=cfg.get("verifier", {}).get("memoize", True),
        )
    raise ValueError(f"unknown lean backend: {kind}")
