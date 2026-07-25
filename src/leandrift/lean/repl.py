"""A single persistent Lean 4 REPL worker (leanprover-community/repl).

Communicates with the REPL over line-delimited JSON on stdin/stdout.  A base
environment with ``import Mathlib`` is loaded once at startup and reused for every
episode (state is reused rather than rebuilt), which is what makes verification
fast enough for an RL inner loop (Section 4.7).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from leandrift.core.statement import Statement
from leandrift.lean.backend import LeanResult

_PREAMBLE = "import Mathlib\n"


class ReplWorker:
    def __init__(self, repl_path: Optional[str], project_path: str = "lean",
                 timeout_s: float = 40.0) -> None:
        self.project_path = project_path
        self.timeout_s = timeout_s
        self._base_env: Optional[int] = None
        self._lock = threading.Lock()
        self._proc = self._spawn(repl_path)
        self._load_preamble()

    def _spawn(self, repl_path: Optional[str]) -> subprocess.Popen:
        # `lake env repl` runs the REPL with the project's Lean/Mathlib toolchain.
        cmd = ["lake", "env", "repl"]
        if repl_path and os.path.exists(os.path.join(repl_path, "Main.lean")):
            cmd = ["lake", "env", "lean", "--run", os.path.join(repl_path, "Main.lean")]
        return subprocess.Popen(
            cmd,
            cwd=self.project_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _send(self, obj: dict, timeout_s: Optional[float] = None) -> dict:
        """Send one JSON command and read one JSON response (blank-line delimited)."""
        assert self._proc.stdin and self._proc.stdout
        payload = json.dumps(obj)
        self._proc.stdin.write(payload + "\n\n")
        self._proc.stdin.flush()

        result_q: "queue.Queue[str]" = queue.Queue()

        def _reader() -> None:
            lines: List[str] = []
            for line in self._proc.stdout:  # type: ignore[union-attr]
                if line.strip() == "" and lines:
                    break
                lines.append(line)
            result_q.put("".join(lines))

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        try:
            raw = result_q.get(timeout=timeout_s or self.timeout_s)
        except queue.Empty:
            raise TimeoutError("Lean REPL timed out")
        return json.loads(raw) if raw.strip() else {}

    def _load_preamble(self) -> None:
        resp = self._send({"cmd": _PREAMBLE}, timeout_s=600)  # first import is slow
        self._base_env = resp.get("env", 0)

    def check(self, statement_source: str, proof_source: str,
              statement: Optional[Statement] = None) -> LeanResult:
        t0 = time.time()
        name = _extract_name(statement_source) or "leandrift_thm"
        source = f"{statement_source} := by\n{proof_source}\n"
        with self._lock:
            try:
                resp = self._send({"cmd": source, "env": self._base_env})
            except TimeoutError:
                return LeanResult(kernel_ok=False, timed_out=True,
                                  error="timeout", elapsed_s=time.time() - t0)
            except Exception as e:  # noqa: BLE001
                return LeanResult(kernel_ok=False, error=f"repl error: {e}",
                                  elapsed_s=time.time() - t0)

            messages = resp.get("messages", [])
            sorries = resp.get("sorries", [])
            errors = [m for m in messages if m.get("severity") == "error"]
            kernel_ok = not errors and not sorries
            axioms: Optional[List[str]] = None
            if kernel_ok and "env" in resp:
                axioms = self._print_axioms(name, resp["env"])
        return LeanResult(
            kernel_ok=kernel_ok,
            sorries=[str(s) for s in sorries],
            axioms=axioms,
            error="; ".join(m.get("data", "") for m in errors),
            elapsed_s=time.time() - t0,
        )

    def _print_axioms(self, name: str, env: int) -> Optional[List[str]]:
        try:
            resp = self._send({"cmd": f"#print axioms {name}", "env": env})
        except Exception:  # noqa: BLE001
            return None
        data = " ".join(m.get("data", "") for m in resp.get("messages", []))
        return _parse_axioms(data)

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self._proc.kill()


def _extract_name(statement_source: str) -> Optional[str]:
    toks = statement_source.split()
    for i, t in enumerate(toks):
        if t == "theorem" and i + 1 < len(toks):
            return toks[i + 1]
    return None


def _parse_axioms(data: str) -> List[str]:
    axioms: List[str] = []
    for line in data.splitlines():
        line = line.strip()
        if line.startswith("'") and "' depends on axioms" in line:
            continue
        for tok in line.replace(",", " ").split():
            tok = tok.strip("[]', ")
            if tok and tok[0].isalpha() and "." in tok or tok in (
                "propext", "sorryAx"):
                axioms.append(tok)
    return axioms
