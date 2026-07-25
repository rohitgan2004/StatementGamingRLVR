"""Taxonomy labeling of accepted (weak-passing) episodes (Section 5.5 / 6.3).

Maps a DRIFTCLASS verdict + pass verdicts onto the paper's report categories:
genuine proof; added conclusion as hypothesis; weakened conclusion; vacuous
premise; quantifier/constraint drift; format exploit; other.
"""

from __future__ import annotations

from leandrift.core.episode import Episode
from leandrift.corpus.weakenings import (
    ADDED_HYP,
    DROPPED_CONSTRAINT,
    QUANTIFIER_DRIFT,
    STRENGTHENED_PREMISE,
    WEAKENED_CONCL,
)

GENUINE = "genuine"
ADDED_CONCLUSION = "added-conclusion"
WEAKENED = "weakened-conclusion"
VACUOUS = "vacuous-premise"
QC_DRIFT = "quantifier-constraint-drift"
FORMAT_EXPLOIT = "format-exploit"
OTHER = "other"

TAXONOMY_ORDER = [
    GENUINE, ADDED_CONCLUSION, WEAKENED, VACUOUS, QC_DRIFT, FORMAT_EXPLOIT, OTHER,
]

_DRIFT_TO_TAX = {
    ADDED_HYP: ADDED_CONCLUSION,
    WEAKENED_CONCL: WEAKENED,
    STRENGTHENED_PREMISE: VACUOUS,
    QUANTIFIER_DRIFT: QC_DRIFT,
    DROPPED_CONSTRAINT: QC_DRIFT,
}


def label(ep: Episode) -> str:
    """Taxonomy label for an accepted episode (assumes weak pass)."""
    if not ep.completion.parsed_ok:
        return FORMAT_EXPLOIT
    if ep.strict.passed:
        return GENUINE
    return _DRIFT_TO_TAX.get(ep.drift_class, OTHER)
