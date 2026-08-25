"""Exercises agent/loop.py's control flow end to end - observe -> decide ->
policy -> act -> log -> checkpoint -> repeat - with no network call and no
browser. FakeSurface and StubLLMClient are the two seams that make this
possible; if these tests pass, the loop's logic is right independent of
whether Groq or Playwright ever get involved.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from core.actions import ActionType
from core.evidence import EvidenceLogger, RunContext, RunType, new_run_id
from core.policy import (
    AllowedTarget,
    AllowlistConfig,
    IntentRisk,
    PolicyChecker,
    RiskLevel,
    RiskPolicyConfig,
    RouteRiskRule,
    RouteRuleExclusion,
)
from core.schema import Action, Locator, LocatorCandidate, LocatorStrategy
from core.surface import SurfaceActionResult, SurfaceObservation

from agent.llm_client import ActionDecision, DecisionStatus, StubLLMClient
from agent.loop import DiscoveryResult, StopReason, run_discovery


class FakeSurface:
    """A scripted, in-memory Surface double. observe() replays a fixed list
    (repeating the last entry once exhausted); act()/take_recoverable_events()
    replay their own scripted queues, defaulting to a bare success/no-events
    when the script runs out - most tests only care about a handful of calls."""

    def __init__(
        self,
        observations: list[SurfaceObservation],
        action_results: list[SurfaceActionResult] | None = None,
        recoverable_events: list[list[dict[str, Any]]] | None = None,
    ):
        self._observations = list(observations)
        self._observe_index = 0
        self._action_results = list(action_results or [])
        self._recoverable_events = list(recoverable_events or [])
        self.actions_received: list[Action] = []
        self.closed = False

    async def observe(self) -> SurfaceObservation:
        idx = min(self._observe_index, len(self._observations) - 1)
        self._observe_index += 1
        return self._observations[idx]

    async def act(self, action: Action) -> SurfaceActionResult:
        self.actions_received.append(action)
        if self._action_results:
            return self._action_results.pop(0)
        return SurfaceActionResult(success=True)

    async def snapshot(self) -> bytes:
        return b""

    def current_url(self) -> str:
        return self._observations[-1].url

    async def close(self) -> None:
        self.closed = True

    async def take_recoverable_events(self) -> list[dict[str, Any]]:
        if self._recoverable_events:
            return self._recoverable_events.pop(0)
        return []


class RaisingLLMClient:
    """Simulates a client whose decode/parse step fails outright on every
    call (e.g. the model returned malformed tool input) - distinct from
    StubLLMClient, which always hands back an already-valid ActionDecision."""

    async def decide(self, context):
        raise ValueError("model returned malformed tool input")


class FlakyLLMClient:
    """Fails with a transient error the first `fail_times` calls, then
    delegates to `then` - simulates the real, observed Groq behavior where
    an occasional tool call fails our schema validation but a retry succeeds."""

    def __init__(self, fail_times: int, then):
        self._fail_times = fail_times
        self._then = then
        self.calls = 0

    async def decide(self, context):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise ValueError("simulated transient malformed tool output")
        return await self._then.decide(context)


def make_observation(url: str = "http://x/", tree: str = '- button "Continue"') -> SurfaceObservation:
    return SurfaceObservation(url=url, title="t", accessibility_tree=tree, timestamp=datetime.now(timezone.utc))


def make_action(intent: str = "search_member") -> Action:
    return Action(
        type=ActionType.CLICK,
        intent=intent,
        target=Locator(
            description="Continue button",
            candidates=[LocatorCandidate(strategy=LocatorStrategy.ROLE, value="button:Continue")],
        ),
    )


def make_policy(safe: tuple[str, ...] = (), risky: tuple[str, ...] = (), blocked: tuple[str, ...] = ()) -> PolicyChecker:
    allowlist = AllowlistConfig(
        allowed_targets=[AllowedTarget(name="t", base_url="http://x", allowed_routes=["*"])],
        allowed_action_types=list(ActionType),
    )
    intents: dict[str, IntentRisk] = {}
    for i in safe:
        intents[i] = IntentRisk(risk=RiskLevel.SAFE)
    for i in risky:
        intents[i] = IntentRisk(risk=RiskLevel.IRREVERSIBLE, requires_human_confirmation=True)
    for i in blocked:
        intents[i] = IntentRisk(risk=RiskLevel.IRREVERSIBLE, requires_human_confirmation=True, blocked=True)
    risk_policy = RiskPolicyConfig(default=IntentRisk(risk=RiskLevel.SAFE), intents=intents)
    return PolicyChecker(allowlist, risk_policy)


def make_policy_with_review_route_rule() -> PolicyChecker:
    """Mirrors config/risk_policy.yaml's route_risk_rules entry for the
    subaccount review screen: `submit_new_sub_account` is declared SAFE by
    intent (as it genuinely is on /subaccounts/new), but a route rule
    overrides that on /subaccounts/review regardless of intent name."""
    allowlist = AllowlistConfig(
        allowed_targets=[AllowedTarget(name="t", base_url="http://x", allowed_routes=["*"])],
        allowed_action_types=list(ActionType),
    )
    risk_policy = RiskPolicyConfig(
        default=IntentRisk(risk=RiskLevel.SAFE),
        intents={"submit_new_sub_account": IntentRisk(risk=RiskLevel.SAFE)},
        route_risk_rules=[
            RouteRiskRule(
                route_pattern="/members/*/subaccounts/review",
                action_types=[ActionType.CLICK],
                exclude=RouteRuleExclusion(locator_text=["cancel"]),
                classification=IntentRisk(risk=RiskLevel.IRREVERSIBLE, requires_human_confirmation=True),
            )
        ],
    )
    return PolicyChecker(allowlist, risk_policy)


def make_evidence_logger(tmp_path) -> EvidenceLogger:
    ctx = RunContext(
        run_id=new_run_id(RunType.DISCOVERY),
        run_type=RunType.DISCOVERY,
        target="test",
        started_at=datetime.now(timezone.utc),
        goal="test goal",
    )
    return EvidenceLogger(tmp_path, ctx)


def event_types(logger: EvidenceLogger) -> list[str]:
    lines = (logger.dir / "log.jsonl").read_text().strip().splitlines()
    return [json.loads(line)["type"] for line in lines]


async def test_happy_path_reaches_done_after_one_action(tmp_path):
    obs = make_observation()
    action = make_action("search_member")
    surface = FakeSurface(observations=[obs], action_results=[SurfaceActionResult(success=True, observation=obs)])
    llm = StubLLMClient(
        [
            ActionDecision(status=DecisionStatus.CONTINUE, reasoning="search for the member", action=action),
            ActionDecision(status=DecisionStatus.DONE, reasoning="found the member"),
        ]
    )
    logger = make_evidence_logger(tmp_path)

    result = await run_discovery(
        goal="find member", surface=surface, llm=llm, policy=make_policy(safe=("search_member",)), evidence=logger
    )

    assert result.succeeded
    assert result.stop_reason is StopReason.GOAL_COMPLETE
    assert len(surface.actions_received) == 1
    assert surface.actions_received[0].intent == "search_member"
    assert len(result.steps) == 1

    types = event_types(logger)
    for expected in ("observation", "decision", "policy_decision", "action"):
        assert expected in types


async def test_stops_at_max_steps_when_model_never_signals_done(tmp_path):
    obs = make_observation()
    action = make_action("search_member")
    surface = FakeSurface(
        observations=[obs], action_results=[SurfaceActionResult(success=True, observation=obs)] * 5
    )
    llm = StubLLMClient(
        [ActionDecision(status=DecisionStatus.CONTINUE, reasoning="again", action=action) for _ in range(5)]
    )
    logger = make_evidence_logger(tmp_path)

    result = await run_discovery(
        goal="g", surface=surface, llm=llm, policy=make_policy(safe=("search_member",)), evidence=logger, max_steps=3
    )

    assert result.stop_reason is StopReason.MAX_STEPS_EXCEEDED
    assert len(result.steps) == 3


async def test_dead_end_stops_when_the_page_never_progresses(tmp_path):
    """Regression test: a real discovery run typed a literal "$100" into a
    numeric field, the app rejected it and reloaded the same page, and the
    model kept re-clicking the same control without correcting anything -
    same URL, over and over - until it exhausted the Groq rate limit and
    crashed. This is the deterministic backstop for that."""
    obs = make_observation()  # always the same URL
    action = make_action("search_member")
    surface = FakeSurface(observations=[obs], action_results=[SurfaceActionResult(success=True, observation=obs)] * 5)
    llm = StubLLMClient([ActionDecision(status=DecisionStatus.CONTINUE, reasoning="again", action=action) for _ in range(3)])
    logger = make_evidence_logger(tmp_path)

    result = await run_discovery(
        goal="g",
        surface=surface,
        llm=llm,
        policy=make_policy(safe=("search_member",)),
        evidence=logger,
        max_steps=20,
        max_dead_end_steps=3,
    )

    assert result.stop_reason is StopReason.DEAD_END
    assert "no progress" in result.final_reasoning
    assert len(result.steps) == 3
    assert "error_detected" in event_types(logger)
    # the triggering (4th) observation never reached llm.decide() - dead-end
    # is detected before spending another decision call, not after.
    assert len(llm.received_contexts) == 3


async def test_dead_end_does_not_trigger_on_normal_multi_action_form_filling(tmp_path):
    """A legitimate multi-field form (type field 1, type field 2, click
    submit) legitimately stays on the same URL for a few actions before
    progressing - the threshold must not fire on that normal case."""
    same_page = make_observation(url="http://x/form")
    next_page = make_observation(url="http://x/next")
    action = make_action("search_member")
    surface = FakeSurface(
        # One entry per iteration's observe() call: still on the form for
        # the first two (filling two fields), moved on by the third (after
        # the submit click actually took effect).
        observations=[same_page, same_page, next_page],
        action_results=[
            SurfaceActionResult(success=True, observation=same_page),
            SurfaceActionResult(success=True, observation=next_page),
        ],
    )
    llm = StubLLMClient(
        [
            ActionDecision(status=DecisionStatus.CONTINUE, reasoning="fill field 1", action=action),
            ActionDecision(status=DecisionStatus.CONTINUE, reasoning="fill field 2", action=action),
            ActionDecision(status=DecisionStatus.DONE, reasoning="done"),
        ]
    )
    logger = make_evidence_logger(tmp_path)

    result = await run_discovery(
        goal="g",
        surface=surface,
        llm=llm,
        policy=make_policy(safe=("search_member",)),
        evidence=logger,
        max_dead_end_steps=3,
    )

    assert result.stop_reason is StopReason.GOAL_COMPLETE


async def test_dead_end_does_not_trigger_when_same_url_has_changing_accessibility_state(tmp_path):
    """The URL alone is not the progress signal - a multi-field form
    legitimately stays on ONE url while the visible state changes at every
    step (account type selected, deposit typed, a validation message
    appears, the field gets corrected). This exceeds the max_dead_end_steps
    threshold in raw same-URL count, but must NOT trigger DEAD_END because
    the accessibility tree is genuinely different each time."""
    url = "http://x/subaccounts/new"
    trees = [
        '- combobox "Account Type": Savings',
        '- combobox "Account Type": Savings\n- textbox "Opening Deposit": 5',
        '- combobox "Account Type": Savings\n- textbox "Opening Deposit": 5\n- text: Minimum opening deposit is $25.00.',
        '- combobox "Account Type": Savings\n- textbox "Opening Deposit": 100',
    ]
    observations = [make_observation(url=url, tree=t) for t in trees]
    action = make_action("submit_new_sub_account")
    surface = FakeSurface(
        observations=observations,
        action_results=[SurfaceActionResult(success=True)] * (len(trees) - 1),
    )
    llm = StubLLMClient(
        [
            ActionDecision(status=DecisionStatus.CONTINUE, reasoning=f"step {i}", action=action)
            for i in range(len(trees) - 1)
        ]
        + [ActionDecision(status=DecisionStatus.DONE, reasoning="done")]
    )
    logger = make_evidence_logger(tmp_path)

    result = await run_discovery(
        goal="g",
        surface=surface,
        llm=llm,
        policy=make_policy(safe=("submit_new_sub_account",)),
        evidence=logger,
        max_dead_end_steps=3,  # same raw same-URL count that DOES trigger in the identical-observation test above
    )

    assert result.stop_reason is StopReason.GOAL_COMPLETE
    assert "error_detected" not in event_types(logger)


async def test_model_signals_stuck_directly_without_acting(tmp_path):
    surface = FakeSurface(observations=[make_observation()])
    llm = StubLLMClient([ActionDecision(status=DecisionStatus.STUCK, reasoning="no matching element")])
    logger = make_evidence_logger(tmp_path)

    result = await run_discovery(goal="g", surface=surface, llm=llm, policy=make_policy(), evidence=logger)

    assert result.stop_reason is StopReason.STUCK
    assert result.final_reasoning == "no matching element"
    assert surface.actions_received == []


async def test_consecutive_failures_trigger_stuck_before_max_steps(tmp_path):
    obs = make_observation()
    action = make_action("search_member")
    surface = FakeSurface(
        observations=[obs],
        action_results=[SurfaceActionResult(success=False, error="element not found", observation=obs)] * 5,
    )
    llm = StubLLMClient(
        [ActionDecision(status=DecisionStatus.CONTINUE, reasoning="try again", action=action) for _ in range(5)]
    )
    logger = make_evidence_logger(tmp_path)

    result = await run_discovery(
        goal="g",
        surface=surface,
        llm=llm,
        policy=make_policy(safe=("search_member",)),
        evidence=logger,
        max_steps=10,
        max_consecutive_failures=3,
    )

    assert result.stop_reason is StopReason.STUCK
    assert len(result.steps) == 3
    assert "3 consecutive" in result.final_reasoning


async def test_policy_blocked_action_stops_before_acting(tmp_path):
    surface = FakeSurface(observations=[make_observation()])
    action = make_action("delete_account")
    llm = StubLLMClient([ActionDecision(status=DecisionStatus.CONTINUE, reasoning="delete it", action=action)])
    logger = make_evidence_logger(tmp_path)

    result = await run_discovery(
        goal="g", surface=surface, llm=llm, policy=make_policy(blocked=("delete_account",)), evidence=logger
    )

    assert result.stop_reason is StopReason.POLICY_BLOCKED
    assert surface.actions_received == []
    assert "escalation_raised" in event_types(logger)


async def test_action_requiring_confirmation_stops_before_acting(tmp_path):
    surface = FakeSurface(observations=[make_observation()])
    action = make_action("transfer_funds")
    llm = StubLLMClient([ActionDecision(status=DecisionStatus.CONTINUE, reasoning="transfer", action=action)])
    logger = make_evidence_logger(tmp_path)

    result = await run_discovery(
        goal="g", surface=surface, llm=llm, policy=make_policy(risky=("transfer_funds",)), evidence=logger
    )

    assert result.stop_reason is StopReason.CONFIRMATION_REQUIRED
    assert surface.actions_received == []


async def test_route_aware_policy_stops_before_acting_even_when_intent_says_safe(tmp_path):
    """The real safety gap this was built from: a live run had the model
    reuse the intent `submit_new_sub_account` - SAFE by config, correctly so
    on /subaccounts/new - for the actual irreversible Review->Confirmation
    click, and the loop executed it because the policy check only looked at
    the intent string. With a route-aware rule in place, the loop must stop
    at CONFIRMATION_REQUIRED and never call surface.act() for that action,
    regardless of what intent the model attached to it."""
    obs = make_observation(url="http://x/members/12345/subaccounts/review")
    action = make_action("submit_new_sub_account")
    surface = FakeSurface(observations=[obs])
    llm = StubLLMClient([ActionDecision(status=DecisionStatus.CONTINUE, reasoning="confirm", action=action)])
    logger = make_evidence_logger(tmp_path)

    result = await run_discovery(
        goal="g", surface=surface, llm=llm, policy=make_policy_with_review_route_rule(), evidence=logger
    )

    assert result.stop_reason is StopReason.CONFIRMATION_REQUIRED
    assert surface.actions_received == []


async def test_invalid_model_decision_is_treated_as_stuck_not_a_crash(tmp_path):
    surface = FakeSurface(observations=[make_observation()])
    logger = make_evidence_logger(tmp_path)

    result = await run_discovery(
        goal="g", surface=surface, llm=RaisingLLMClient(), policy=make_policy(), evidence=logger, max_decision_retries=0
    )

    assert result.stop_reason is StopReason.STUCK
    assert "invalid model output" in result.final_reasoning
    assert "error_detected" in event_types(logger)


async def test_decision_retries_are_exhausted_and_logged_before_giving_up(tmp_path):
    surface = FakeSurface(observations=[make_observation()])
    logger = make_evidence_logger(tmp_path)

    result = await run_discovery(
        goal="g", surface=surface, llm=RaisingLLMClient(), policy=make_policy(), evidence=logger, max_decision_retries=2
    )

    assert result.stop_reason is StopReason.STUCK
    assert "after 3 attempts" in result.final_reasoning
    assert event_types(logger).count("error_detected") == 3


async def test_decision_retry_recovers_from_a_transient_invalid_output(tmp_path):
    obs = make_observation()
    action = make_action("search_member")
    surface = FakeSurface(observations=[obs], action_results=[SurfaceActionResult(success=True, observation=obs)])
    good_llm = StubLLMClient(
        [
            ActionDecision(status=DecisionStatus.CONTINUE, reasoning="search", action=action),
            ActionDecision(status=DecisionStatus.DONE, reasoning="done"),
        ]
    )
    flaky_llm = FlakyLLMClient(fail_times=1, then=good_llm)
    logger = make_evidence_logger(tmp_path)

    result = await run_discovery(
        goal="g",
        surface=surface,
        llm=flaky_llm,
        policy=make_policy(safe=("search_member",)),
        evidence=logger,
        max_decision_retries=2,
    )

    assert result.succeeded
    # step 0: attempt 1 fails, attempt 2 succeeds (search) -> step 1: attempt 1 succeeds (done)
    assert flaky_llm.calls == 3
    assert event_types(logger).count("error_detected") == 1


async def test_recoverable_event_is_logged_and_flagged_in_the_next_context(tmp_path):
    obs = make_observation()
    action = make_action("search_member")
    surface = FakeSurface(
        observations=[obs, obs],
        action_results=[SurfaceActionResult(success=True, observation=obs)],
        recoverable_events=[[{"type": "confirm", "message": "Continue working?", "accepted": False}]],
    )
    llm = StubLLMClient(
        [
            ActionDecision(status=DecisionStatus.CONTINUE, reasoning="go", action=action),
            ActionDecision(status=DecisionStatus.DONE, reasoning="done"),
        ]
    )
    logger = make_evidence_logger(tmp_path)

    result = await run_discovery(
        goal="g", surface=surface, llm=llm, policy=make_policy(safe=("search_member",)), evidence=logger
    )

    assert result.steps[0].recoverable_events[0]["type"] == "confirm"
    assert "recovery_attempted" in event_types(logger)
    assert llm.received_contexts[1].note is not None
    assert "Continue working?" in llm.received_contexts[1].note
