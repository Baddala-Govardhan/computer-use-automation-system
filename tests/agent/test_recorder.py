from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent.llm_client import ActionDecision, DecisionStatus
from agent.loop import DiscoveryResult, DiscoveryStep, StopReason
from agent.recorder import record_discovery
from core.actions import ActionType
from core.policy import PolicyDecision, RiskLevel
from core.schema import Action, Locator, LocatorCandidate, LocatorStrategy
from core.surface import SurfaceActionResult, SurfaceObservation

SAFE_POLICY_DECISION = PolicyDecision(
    allowed=True, risk_level=RiskLevel.SAFE, requires_human_confirmation=False, reason="ok"
)


def make_observation(url: str) -> SurfaceObservation:
    return SurfaceObservation(url=url, title="t", accessibility_tree="- x", timestamp=datetime.now(timezone.utc))


def make_action(
    intent: str, *, type_: ActionType = ActionType.CLICK, value: str | None = None, candidates=None
) -> Action:
    target = None
    if type_ in (ActionType.CLICK, ActionType.TYPE, ActionType.SELECT, ActionType.EXTRACT):
        target = Locator(
            description=intent,
            candidates=candidates or [LocatorCandidate(strategy=LocatorStrategy.ROLE, value=f"button:{intent}")],
        )
    return Action(type=type_, intent=intent, target=target, value=value)


def make_step(
    index: int,
    action: Action,
    reasoning: str,
    *,
    success: bool = True,
    observed_url: str | None = None,
    error: str | None = None,
) -> DiscoveryStep:
    observation = make_observation(observed_url) if observed_url else None
    return DiscoveryStep(
        index=index,
        decision=ActionDecision(status=DecisionStatus.CONTINUE, reasoning=reasoning, action=action),
        policy_decision=SAFE_POLICY_DECISION,
        action_result=SurfaceActionResult(success=success, error=error, observation=observation),
    )


def make_successful_result(steps: list[DiscoveryStep], final_reasoning: str = "done") -> DiscoveryResult:
    return DiscoveryResult(stop_reason=StopReason.GOAL_COMPLETE, steps=steps, final_reasoning=final_reasoning)


def test_records_a_successful_trace():
    steps = [
        make_step(
            0, make_action("search_member", type_=ActionType.TYPE, value="12345"), "type id", observed_url="http://x/search"
        ),
        make_step(1, make_action("search_member"), "click search", observed_url="http://x/members/12345"),
    ]
    result = make_successful_result(steps)

    trace = record_discovery(result, goal="find member", discovery_run_id="discovery-1")

    assert trace.goal == "find member"
    assert trace.discovery_run_id == "discovery-1"
    assert len(trace.steps) == 2


def test_action_ordering_is_preserved():
    steps = [
        make_step(0, make_action("search_member", type_=ActionType.TYPE, value="12345"), "a", observed_url="http://x/1"),
        make_step(1, make_action("search_member"), "b", observed_url="http://x/2"),
        make_step(2, make_action("open_new_sub_account"), "c", observed_url="http://x/3"),
    ]
    result = make_successful_result(steps)

    trace = record_discovery(result, goal="g", discovery_run_id="d1")

    assert [s.action.intent for s in trace.steps] == ["search_member", "search_member", "open_new_sub_account"]
    assert [s.reasoning for s in trace.steps] == ["a", "b", "c"]


def test_locator_fallback_ordering_is_preserved():
    candidates = [
        LocatorCandidate(strategy=LocatorStrategy.ROLE, value="button:Continue", confidence=1.0),
        LocatorCandidate(strategy=LocatorStrategy.CSS, value="main form button[value='confirm']", confidence=0.5),
    ]
    action = make_action("submit_new_sub_account", candidates=candidates)
    steps = [make_step(0, action, "click continue", observed_url="http://x/review")]
    result = make_successful_result(steps)

    trace = record_discovery(result, goal="g", discovery_run_id="d1")

    recorded = trace.steps[0].action.target.candidates
    assert [c.strategy for c in recorded] == [LocatorStrategy.ROLE, LocatorStrategy.CSS]
    assert [c.value for c in recorded] == ["button:Continue", "main form button[value='confirm']"]
    assert [c.confidence for c in recorded] == [1.0, 0.5]


def test_failed_discovery_cannot_be_recorded():
    steps = [make_step(0, make_action("search_member"), "a", observed_url="http://x/1")]
    stuck_result = DiscoveryResult(stop_reason=StopReason.STUCK, steps=steps, final_reasoning="no matching element")

    with pytest.raises(ValueError, match="did not complete successfully"):
        record_discovery(stuck_result, goal="g", discovery_run_id="d1")


@pytest.mark.parametrize(
    "stop_reason", [StopReason.POLICY_BLOCKED, StopReason.CONFIRMATION_REQUIRED, StopReason.MAX_STEPS_EXCEEDED]
)
def test_every_non_success_stop_reason_is_rejected(stop_reason):
    steps = [make_step(0, make_action("search_member"), "a", observed_url="http://x/1")]
    result = DiscoveryResult(stop_reason=stop_reason, steps=steps, final_reasoning="stopped")

    with pytest.raises(ValueError):
        record_discovery(result, goal="g", discovery_run_id="d1")


def test_a_goal_complete_result_with_zero_steps_cannot_be_recorded():
    result = make_successful_result(steps=[], final_reasoning="somehow done immediately")

    with pytest.raises(ValueError, match="no successful actions"):
        record_discovery(result, goal="g", discovery_run_id="d1")


def test_failed_intermediate_actions_are_excluded_from_the_trace():
    steps = [
        make_step(0, make_action("search_member"), "tried locator A, it failed", success=False, error="no match"),
        make_step(1, make_action("search_member"), "tried locator B, this worked", observed_url="http://x/found"),
    ]
    result = make_successful_result(steps)

    trace = record_discovery(result, goal="g", discovery_run_id="d1")

    assert len(trace.steps) == 1
    assert trace.steps[0].reasoning == "tried locator B, this worked"


def test_recorder_redacts_pii_patterns_from_reasoning():
    action = make_action("view_member_detail")
    steps = [
        make_step(
            0,
            action,
            "The member's email on file is jane.doe@example.com, confirming identity.",
            observed_url="http://x/1",
        )
    ]
    result = make_successful_result(steps)

    trace = record_discovery(result, goal="g", discovery_run_id="d1")

    assert "jane.doe@example.com" not in trace.steps[0].reasoning
    assert "REDACTED" in trace.steps[0].reasoning


def test_pre_auth_credentials_are_structurally_never_part_of_a_recorded_trace():
    steps = [
        make_step(0, make_action("search_member", type_=ActionType.TYPE, value="12345"), "search", observed_url="http://x/1")
    ]
    result = make_successful_result(steps)

    trace = record_discovery(result, goal="g", discovery_run_id="d1")

    recorded_values = [s.action.value for s in trace.steps if s.action.value]
    assert "changeme123" not in recorded_values
    assert recorded_values == ["12345"]
