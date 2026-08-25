from __future__ import annotations

from dataclasses import dataclass

from core.outcomes import BusinessOutcome
from core.surface import SurfaceObservation

PERMISSION_DENIED_PATTERNS = ("do not have permission", "permission denied", "access denied")
NOT_FOUND_PATTERNS = ("member not found", "record not found", "not found")
VALIDATION_ERROR_PATTERNS = ("minimum opening deposit",)
_CONTEXT_RADIUS = 80


@dataclass
class Detection:
    business_outcome: BusinessOutcome | None = None
    hard_failure_reason: str | None = None


def _context(text: str, pattern: str) -> str:
    idx = text.lower().find(pattern)
    if idx == -1:
        return ""
    start = max(0, idx - _CONTEXT_RADIUS)
    end = min(len(text), idx + len(pattern) + _CONTEXT_RADIUS)
    return " ".join(text[start:end].split())


def detect(observation: SurfaceObservation) -> Detection:
    text = observation.accessibility_tree
    text_lower = text.lower()

    for pattern in PERMISSION_DENIED_PATTERNS:
        if pattern in text_lower:
            return Detection(hard_failure_reason=f"permission denied - page shows: \"{_context(text, pattern)}\"")

    for pattern in NOT_FOUND_PATTERNS:
        if pattern in text_lower:
            return Detection(
                business_outcome=BusinessOutcome(
                    code="not_found",
                    message="The requested record was not found.",
                    data={"url": observation.url, "context": _context(text, pattern)},
                )
            )

    for pattern in VALIDATION_ERROR_PATTERNS:
        if pattern in text_lower:
            return Detection(
                business_outcome=BusinessOutcome(
                    code="validation_error",
                    message="The application reported a validation error.",
                    data={"url": observation.url, "context": _context(text, pattern)},
                )
            )

    return Detection()
