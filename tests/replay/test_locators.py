from core.schema import Locator, LocatorCandidate, LocatorStrategy
from replay.locators import describe_locator_failure, locator_search_terms


def test_locator_search_terms_extracts_role_accessible_name():
    locator = Locator(description="Continue button", candidates=[LocatorCandidate(strategy=LocatorStrategy.ROLE, value="button:Continue")])
    terms = locator_search_terms(locator)
    assert "Continue button" in terms
    assert "Continue" in terms


def test_locator_search_terms_uses_raw_value_for_non_role_strategies():
    locator = Locator(description="x", candidates=[LocatorCandidate(strategy=LocatorStrategy.CSS, value="main form button")])
    terms = locator_search_terms(locator)
    assert "main form button" in terms


def test_describe_ambiguous_failure():
    expected, observed = describe_locator_failure("Continue button", "no candidate resolved: role=button:Continue: ambiguous (2 matches)")
    assert "exactly one element" in expected
    assert "ambiguous" in observed


def test_describe_no_match_failure():
    expected, observed = describe_locator_failure("Continue button", "no candidate resolved: role=button:Continue: no match")
    assert "an element matching" in expected


def test_describe_unknown_failure_falls_back_to_generic_expectation():
    expected, observed = describe_locator_failure("Continue button", "some other Playwright error")
    assert "actionable" in expected


def test_describe_failure_with_no_error_text():
    expected, observed = describe_locator_failure("Continue button", None)
    assert "no error detail" in observed
