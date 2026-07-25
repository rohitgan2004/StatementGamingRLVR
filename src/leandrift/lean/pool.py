"""Warm pool of persistent Lean REPL workers with memoization (Section 4.7).

A fixed pool of workers shares the elaboration state (each has ``import Mathlib``
preloaded); results are memoized on ``hash(s_hat, pi)`` so repetitive failures
early in training and statement-variant collapse near convergence are free.  A
per-episode wall-clock budget bounds tail latency; timeouts count as failures.
"""

from __future__ import annotations

import hashlib
import queue
import threading
from typing import Dict, Optional

from leandrift.core.statement import Statement
from leandrift.lean.backend import LeanBackend, LeanResult


def _key(statement_source: str, proof_source: str) -> str:
    return hashlib.sha256(f"{statement_source}\x00{proof_source}".encode()).hexdigest()


class ReplPool(LeanBackend):
    def __init__(self, n_workers: int = 8, timeout_s: float = 40.0,
                 repl_path: Optional[str] = None, project_path: str = "lean",
                 memoize: bool = True) -> None:
        from leandrift.lean.repl import ReplWorker  # local import: needs Lean at runtime

        self.timeout_s = timeout_s
        self.memoize = memoize
        self._cache: Dict[str, LeanResult] = {}
        self._cache_lock = threading.Lock()
        self._idle: "queue.Queue[ReplWorker]" = queue.Queue()
        self._workers = []
        for _ in range(n_workers):
            w = ReplWorker(repl_path=repl_path, project_path=project_path,
                           timeout_s=timeout_s)
            self._workers.append(w)
            self._idle.put(w)

    def check(self, statement_source: str, proof_source: str,
              statement: Optional[Statement] = None) -> LeanResult:
        if self.memoize:
            k = _key(statement_source, proof_source)
            with self._cache_lock:
                if k in self._cache:
                    return self._cache[k]
        worker = self._idle.get()
        try:
            result = worker.check(statement_source, proof_source, statement)
        finally:
            self._idle.put(worker)
        if self.memoize:
            with self._cache_lock:
                self._cache[k] = result
        return result

    def close(self) -> None:
        for w in self._workers:
            w.close()
