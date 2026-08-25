from __future__ import annotations

from core.schema import Locator, LocatorStrategy


def locator_search_terms(locator: Locator) -> list[str]:
    terms = [locator.description]
    for candidate in locator.candidates:
        if candidate.strategy is LocatorStrategy.ROLE and ":" in candidate.value:
            terms.append(candidate.value.split(":", 1)[1])
        else:
            terms.append(candidate.value)
    return terms


def describe_locator_failure(action_description: str, error: str | None) -> tuple[str, str]:
    observed = error or "(no error detail returned by the surface)"
    lowered = observed.lower()
    if "ambiguous" in lowered:
        expected = f"exactly one element matching '{action_description}'"
    elif "no match" in lowered:
        expected = f"an element matching '{action_description}'"
    else:
        expected = f"'{action_description}' to be actionable"
    return expected, observed
