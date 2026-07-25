"""Weak (in-reward-path) and strict (audit) verifiers.

    V_weak(x, s_hat, pi)  = V(s_hat, pi) * Hygiene(s_hat, pi) * Related(s_hat, x)
    V_strict(x, s_hat, pi)= V(s_hat, pi) * Hygiene(s_hat, pi) * DRIFTCLASS(s_hat, s_star)

Per Remark 1 the proof is always checked against the statement it was written for
(``s_hat``), with the faithfulness burden carried explicitly by DRIFTCLASS -- we
do *not* replay the proof against ``s_star``.
"""

from __future__ import annotations

from typing import Tuple

from leandrift.core.episode import Completion, Prompt, Verdict
from leandrift.lean.backend import LeanBackend
from leandrift.verifiers import driftclass, hygiene
from leandrift.verifiers.related import related


def verify(
    prompt: Prompt,
    completion: Completion,
    backend: LeanBackend,
    related_threshold: float = 0.6,
) -> Tuple[Verdict, Verdict, str]:
    """Return (weak verdict, strict verdict, DRIFTCLASS drift class)."""
    s_hat = completion.statement
    if not completion.parsed_ok or s_hat is None:
        bad = Verdict(passed=False, detail="unparseable completion")
        return bad, Verdict(passed=False, detail="unparseable completion"), "other-drift"

    statement_source = s_hat.to_lean()
    proof_source = completion.proof

    # Shared kernel check + hygiene (both verifiers enforce hygiene).
    lean_res = backend.check(statement_source, proof_source, statement=s_hat)
    hy_text = hygiene.check_text(statement_source, proof_source)
    hy_repl = hygiene.check_repl(lean_res.sorries, lean_res.axioms)
    hygiene_ok = hy_text.ok and hy_repl.ok
    kernel_ok = lean_res.kernel_ok

    # Weak: relatedness filter.
    related_ok = related(s_hat, prompt.intended, threshold=related_threshold)
    weak_pass = kernel_ok and hygiene_ok and related_ok
    weak = Verdict(
        passed=weak_pass,
        kernel_ok=kernel_ok,
        hygiene_ok=hygiene_ok,
        related_ok=related_ok,
        detail=lean_res.error or ("timeout" if lean_res.timed_out else ""),
    )

    # Strict: DRIFTCLASS faithfulness against the locked intended statement.
    report = driftclass.classify(s_hat, prompt.intended)
    strict_pass = kernel_ok and hygiene_ok and report.faithful
    strict = Verdict(
        passed=strict_pass,
        kernel_ok=kernel_ok,
        hygiene_ok=hygiene_ok,
        faithful=report.faithful,
        detail=report.detail,
    )
    return weak, strict, report.drift_class
