from pathlib import Path

from core.actions import ActionType
from core.policy import (
    AllowedTarget,
    AllowlistConfig,
    IntentRisk,
    PolicyChecker,
    RiskLevel,
    RiskPolicyConfig,
)
from core.schema import Action, Locator, LocatorCandidate, LocatorStrategy

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_locator() -> Locator:
    return Locator(description="x", candidates=[LocatorCandidate(strategy=LocatorStrategy.ROLE, value="x")])


def make_checker(**risk_overrides) -> PolicyChecker:
    allowlist = AllowlistConfig(
        allowed_targets=[
            AllowedTarget(
                name="mock_bank_app",
                base_url="http://localhost:8000",
                allowed_routes=["/", "/members", "/members/*", "/accounts/*"],
            )
        ],
        allowed_action_types=[ActionType.CLICK, ActionType.TYPE, ActionType.NAVIGATE],
    )
    intents = {
        "search_member": IntentRisk(risk=RiskLevel.SAFE),
        "transfer_funds": IntentRisk(risk=RiskLevel.IRREVERSIBLE, requires_human_confirmation=True),
        "delete_account": IntentRisk(risk=RiskLevel.IRREVERSIBLE, requires_human_confirmation=True, blocked=True),
    }
    intents.update(risk_overrides)
    risk_policy = RiskPolicyConfig(default=IntentRisk(risk=RiskLevel.REVIEW_REQUIRED, requires_human_confirmation=True), intents=intents)
    return PolicyChecker(allowlist, risk_policy)


def test_safe_intent_within_allowlist_is_allowed_without_confirmation():
    checker = make_checker()
    action = Action(type=ActionType.TYPE, intent="search_member", value="12345", target=make_locator())
    decision = checker.check_action(action, current_url="http://localhost:8000/members")
    assert decision.allowed
    assert not decision.requires_human_confirmation
    assert decision.risk_level is RiskLevel.SAFE


def test_irreversible_intent_is_allowed_but_requires_confirmation():
    checker = make_checker()
    action = Action(type=ActionType.CLICK, intent="transfer_funds", target=make_locator())
    decision = checker.check_action(action, current_url="http://localhost:8000/accounts/1")
    assert decision.allowed
    assert decision.requires_human_confirmation
    assert decision.risk_level is RiskLevel.IRREVERSIBLE


def test_blocked_intent_is_never_allowed_even_in_allowlisted_route():
    checker = make_checker()
    action = Action(type=ActionType.CLICK, intent="delete_account", target=make_locator())
    decision = checker.check_action(action, current_url="http://localhost:8000/accounts/1")
    assert not decision.allowed


def test_action_type_outside_allowed_action_types_is_rejected():
    checker = make_checker()
    action = Action(type=ActionType.SELECT, intent="search_member", target=make_locator())
    decision = checker.check_action(action, current_url="http://localhost:8000/members")
    assert not decision.allowed
    assert "action type" in decision.reason


def test_url_outside_every_allowlisted_target_is_rejected():
    checker = make_checker()
    action = Action(type=ActionType.NAVIGATE, intent="navigate", url="http://evil.example.com/steal")
    decision = checker.check_action(action, current_url="http://localhost:8000/")
    assert not decision.allowed


def test_route_glob_matching_respects_allowed_routes():
    checker = make_checker()
    action = Action(type=ActionType.NAVIGATE, intent="navigate", url="http://localhost:8000/settings")
    decision = checker.check_action(action, current_url="http://localhost:8000/")
    assert not decision.allowed


def test_unknown_intent_defaults_to_conservative_review_required():
    checker = make_checker()
    action = Action(type=ActionType.CLICK, intent="never_seen_before", target=make_locator())
    decision = checker.check_action(action, current_url="http://localhost:8000/members")
    assert decision.risk_level is RiskLevel.REVIEW_REQUIRED
    assert decision.requires_human_confirmation


def test_real_config_files_load_and_validate():
    checker = PolicyChecker.from_files(REPO_ROOT / "config" / "allowlist.yaml", REPO_ROOT / "config" / "risk_policy.yaml")
    action = Action(type=ActionType.TYPE, intent="search_member", value="12345", target=make_locator())
    decision = checker.check_action(action, current_url="http://localhost:8000/members")
    assert decision.allowed


def _real_checker() -> PolicyChecker:
    return PolicyChecker.from_files(REPO_ROOT / "config" / "allowlist.yaml", REPO_ROOT / "config" / "risk_policy.yaml")


def _button_locator(name: str, css: str | None = None) -> Locator:
    candidates = [LocatorCandidate(strategy=LocatorStrategy.ROLE, value=f"button:{name}")]
    if css:
        candidates.append(LocatorCandidate(strategy=LocatorStrategy.CSS, value=css))
    return Locator(description=f"{name} button on review screen", candidates=candidates)


def test_submit_intent_on_subaccounts_new_route_stays_safe():
    checker = _real_checker()
    action = Action(type=ActionType.CLICK, intent="submit_new_sub_account", target=_button_locator("Continue"))
    decision = checker.check_action(action, current_url="http://localhost:8000/members/12345/subaccounts/new")
    assert decision.allowed
    assert not decision.requires_human_confirmation
    assert decision.risk_level is RiskLevel.SAFE


def test_submit_intent_on_review_route_is_irreversible_regardless_of_intent_name():
    checker = _real_checker()
    action = Action(
        type=ActionType.CLICK,
        intent="submit_new_sub_account",
        target=_button_locator("Continue", css="main form button[value='confirm']"),
    )
    decision = checker.check_action(action, current_url="http://localhost:8000/members/12345/subaccounts/review")
    assert decision.allowed
    assert decision.requires_human_confirmation
    assert decision.risk_level is RiskLevel.IRREVERSIBLE


def test_confirm_intent_on_review_route_is_irreversible():
    checker = _real_checker()
    action = Action(
        type=ActionType.CLICK,
        intent="confirm_new_sub_account",
        target=_button_locator("Continue", css="main form button[value='confirm']"),
    )
    decision = checker.check_action(action, current_url="http://localhost:8000/members/12345/subaccounts/review")
    assert decision.allowed
    assert decision.requires_human_confirmation
    assert decision.risk_level is RiskLevel.IRREVERSIBLE


def test_cancel_on_review_route_is_not_misclassified_irreversible():
    checker = _real_checker()
    action = Action(type=ActionType.CLICK, intent="submit_new_sub_account", target=_button_locator("Cancel"))
    decision = checker.check_action(action, current_url="http://localhost:8000/members/12345/subaccounts/review")
    assert decision.allowed
    assert not decision.requires_human_confirmation
    assert decision.risk_level is RiskLevel.SAFE


def test_cancel_intent_is_also_excluded_by_name():
    checker = _real_checker()
    action = Action(type=ActionType.CLICK, intent="cancel_new_sub_account", target=_button_locator("Cancel"))
    decision = checker.check_action(action, current_url="http://localhost:8000/members/12345/subaccounts/review")
    assert decision.risk_level is not RiskLevel.IRREVERSIBLE


def test_navigation_on_review_route_is_not_misclassified_irreversible():
    checker = _real_checker()
    action = Action(
        type=ActionType.CLICK,
        intent="search_member",
        target=Locator(
            description="Member Search nav link",
            candidates=[LocatorCandidate(strategy=LocatorStrategy.ROLE, value="link:Member Search")],
        ),
    )
    decision = checker.check_action(action, current_url="http://localhost:8000/members/12345/subaccounts/review")
    assert decision.allowed
    assert not decision.requires_human_confirmation
    assert decision.risk_level is RiskLevel.SAFE
