from __future__ import annotations

TIMEOUT_RETRY_MULTIPLIER = 4
MAX_TIMEOUT_RETRIES = 1


def is_timeout_error(error: str | None) -> bool:
    return bool(error) and "timeout" in error.lower()
