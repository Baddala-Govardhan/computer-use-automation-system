from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from core.actions import ActionType
from core.evidence import EvidenceLogger, RunContext, RunType, new_run_id
from core.outcomes import OutcomeStatus
from core.policy import AllowedTarget, AllowlistConfig, IntentRisk, PolicyChecker, RiskLevel, RiskPolicyConfig
from core.schema import (
    Action,
    Capability,
    CapabilityMetadata,
    Checkpoint,
    CheckpointType,
    InputParameter,
    Locator,
    LocatorCandidate,
    LocatorStrategy,
    OutputDefinition,
    ParamType,
    Step,
    TargetRef,
)
from core.surface import SurfaceActionResult, SurfaceObservation
from replay.engine import replay_capability

BASE_URL = "http://x"


class ScriptedSurface:
    def __init__(
        self,
        action_results: list[SurfaceActionResult] | None = None,
        recoverable_events: list[list[dict[str, Any]]] | None = None,
        starting_url: str = f"{BASE_URL}/start",
    ):
        self._action_results = list(action_results or [])
        self._recoverable_events = list(recoverable_events or [])
        self._current_url = starting_url
        self.actions_received: list[Action] = []

    async def observe(self) -> SurfaceObservation:
        return SurfaceObservation(url=self._current_url, title="t", accessibility_tree="", timestamp=datetime.now(timezone.utc))

    async def act(self, action: Action) -> SurfaceActionResult:
        self.actions_received.append(action)
        result = self._action_results.pop(0) if self._action_results else SurfaceActionResult(success=True)
        if result.observation is not None:
            self._current_url = result.observation.url
        return result

    async def snapshot(self) -> bytes:
        return b""

    def current_url(self) -> str:
        return self._current_url

    async def close(self) -> None:
        pass

    async def take_recoverable_events(self) -> list[dict[str, Any]]:
        return self._recoverable_events.pop(0) if self._recoverable_events else []


def make_observation(url: str, tree: str = "") -> SurfaceObservation:
    return SurfaceObservation(url=url, title="t", accessibility_tree=tree, timestamp=datetime.now(timezone.utc))


def make_locator(description: str, value: str = "button:X", strategy: LocatorStrategy = LocatorStrategy.ROLE, extra=None) -> Locator:
    candidates = [LocatorCandidate(strategy=strategy, value=value)]
    if extra:
        candidates.extend(extra)
    return Locator(description=description, candidates=candidates)


def make_capability(
    steps: list[Step],
    success_checkpoint: Checkpoint,
    *,
    inputs: list[InputParameter] | None = None,
    outputs: list[OutputDefinition] | None = None,
) -> Capability:
    return Capability(
        id="test_capability",
        name="Test Capability",
        version="1.0.0",
        description="A capability for testing replay/engine.py",
        target=TargetRef(app="mock_bank_app", base_url=BASE_URL),
        inputs=inputs or [],
        outputs=outputs or [],
        steps=steps,
        success_checkpoint=success_checkpoint,
        metadata=CapabilityMetadata(created_at=datetime.now(timezone.utc), discovery_run_id="discovery-x", model_used="test-model"),
    )


def two_step_capability() -> Capability:
    return make_capability(
        steps=[
            Step(
                id="step_0",
                description="Enter {member_id} into search",
                action=Action(type=ActionType.TYPE, intent="search_member", value="{member_id}", target=make_locator("Member ID box", "textbox:Member ID")),
                checkpoint=Checkpoint(description="still on search", type=CheckpointType.URL_MATCHES, url_pattern=f"{BASE_URL}/members/search"),
            ),
            Step(
                id="step_1",
                description="Click search",
                action=Action(type=ActionType.CLICK, intent="search_member", target=make_locator("Search button", "button:Search")),
                checkpoint=Checkpoint(description="on a member page", type=CheckpointType.URL_MATCHES, url_pattern=f"{BASE_URL}/members/*"),
            ),
        ],
        success_checkpoint=Checkpoint(description="reached member page", type=CheckpointType.URL_MATCHES, url_pattern=f"{BASE_URL}/members/*"),
        inputs=[InputParameter(name="member_id", type=ParamType.STRING, description="Member ID")],
    )


def make_policy(**intent_overrides: IntentRisk) -> PolicyChecker:
    allowlist = AllowlistConfig(
        allowed_targets=[AllowedTarget(name="t", base_url=BASE_URL, allowed_routes=["*"])],
        allowed_action_types=list(ActionType),
    )
    intents = {"search_member": IntentRisk(risk=RiskLevel.SAFE)}
    intents.update(intent_overrides)
    risk_policy = RiskPolicyConfig(default=IntentRisk(risk=RiskLevel.SAFE), intents=intents)
    return PolicyChecker(allowlist, risk_policy)


def make_evidence_logger(tmp_path: Path) -> EvidenceLogger:
    ctx = RunContext(
        run_id=new_run_id(RunType.REPLAY),
        run_type=RunType.REPLAY,
        target=BASE_URL,
        started_at=datetime.now(timezone.utc),
        capability_id="test_capability",
        capability_version="1.0.0",
    )
    return EvidenceLogger(tmp_path, ctx)


def event_types(logger: EvidenceLogger) -> list[str]:
    lines = (logger.dir / "log.jsonl").read_text().strip().splitlines()
    return [json.loads(line)["type"] for line in lines]


def events_by_type(logger: EvidenceLogger, type_: str) -> list[dict]:
    lines = (logger.dir / "log.jsonl").read_text().strip().splitlines()
    return [json.loads(line) for line in lines if json.loads(line)["type"] == type_]


async def test_successful_replay_with_different_inputs_than_discovery(tmp_path):
    cap = two_step_capability()
    surface = ScriptedSurface(
        action_results=[
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search")),
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/67890")),
        ]
    )
    result = await replay_capability(cap, {"member_id": "67890"}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path))

    assert result.status is OutcomeStatus.SUCCESS
    assert result.capability_id == "test_capability"
    assert surface.actions_received[0].value == "67890"
    assert "{member_id}" not in (surface.actions_received[0].value or "")


async def test_parameter_substitution_applies_to_locator_and_value(tmp_path):
    cap = two_step_capability()
    cap.steps[0].action.target.candidates[0] = cap.steps[0].action.target.candidates[0].model_copy(
        update={"value": "textbox:Member {member_id}"}
    )
    surface = ScriptedSurface(
        action_results=[
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search")),
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/999")),
        ]
    )
    await replay_capability(cap, {"member_id": "999"}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path))

    sent = surface.actions_received[0]
    assert sent.value == "999"
    assert sent.target.candidates[0].value == "textbox:Member 999"


async def test_locator_primary_candidate_success(tmp_path):
    cap = two_step_capability()
    surface = ScriptedSurface(
        action_results=[
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search")),
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/1")),
        ]
    )
    result = await replay_capability(cap, {"member_id": "1"}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path))
    assert result.status is OutcomeStatus.SUCCESS


async def test_locator_fallback_candidates_are_passed_through_unmodified(tmp_path):
    cap = two_step_capability()
    cap.steps[1].action.target = make_locator(
        "Search button",
        value="button:Search",
        extra=[LocatorCandidate(strategy=LocatorStrategy.CSS, value="main form button[value='confirm']", confidence=0.5)],
    )
    surface = ScriptedSurface(
        action_results=[
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search")),
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/1")),
        ]
    )
    result = await replay_capability(cap, {"member_id": "1"}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path))

    assert result.status is OutcomeStatus.SUCCESS
    sent_candidates = surface.actions_received[1].target.candidates
    assert [c.strategy for c in sent_candidates] == [LocatorStrategy.ROLE, LocatorStrategy.CSS]
    assert [c.value for c in sent_candidates] == ["button:Search", "main form button[value='confirm']"]


async def test_ambiguous_locator_fails_closed(tmp_path):
    cap = two_step_capability()
    surface = ScriptedSurface(
        action_results=[
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search")),
            SurfaceActionResult(success=False, error="no candidate resolved for locator 'Search button': role=button:Search: ambiguous (2 matches)"),
        ]
    )
    result = await replay_capability(cap, {"member_id": "1"}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path))

    assert result.status is OutcomeStatus.HARD_FAILURE
    assert result.failure.step_id == "step_1"
    assert "ambiguous" not in result.failure.expected
    assert "exactly one element" in result.failure.expected
    assert "ambiguous" in result.failure.observed


async def test_missing_locator_is_a_hard_failure(tmp_path):
    cap = two_step_capability()
    surface = ScriptedSurface(
        action_results=[
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search")),
            SurfaceActionResult(success=False, error="no candidate resolved for locator 'Search button': role=button:Search: no match"),
        ]
    )
    result = await replay_capability(cap, {"member_id": "1"}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path))

    assert result.status is OutcomeStatus.HARD_FAILURE
    assert "an element matching" in result.failure.expected


async def test_final_checkpoint_mismatch_is_a_hard_failure(tmp_path):
    cap = two_step_capability()
    cap.steps[1] = cap.steps[1].model_copy(update={"checkpoint": None})
    surface = ScriptedSurface(
        action_results=[
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search")),
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/somewhere/unexpected")),
        ]
    )
    result = await replay_capability(cap, {"member_id": "1"}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path))

    assert result.status is OutcomeStatus.HARD_FAILURE
    assert result.failure.step_id == "final_checkpoint"
    assert result.failure.observed == f"{BASE_URL}/somewhere/unexpected"


async def test_member_not_found_is_a_business_outcome_not_a_crash(tmp_path):
    cap = two_step_capability()
    surface = ScriptedSurface(
        action_results=[
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search")),
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/40400", tree="Member not found")),
        ]
    )
    result = await replay_capability(cap, {"member_id": "40400"}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path))

    assert result.status is OutcomeStatus.BUSINESS_OUTCOME
    assert result.business_outcome.code == "not_found"
    assert result.failure is None


async def test_permission_denied_is_a_hard_failure_not_a_business_outcome(tmp_path):
    cap = two_step_capability()
    surface = ScriptedSurface(
        action_results=[
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search")),
            SurfaceActionResult(
                success=True,
                observation=make_observation(f"{BASE_URL}/members/40300", tree="System Message: You do not have permission to perform this operation."),
            ),
        ]
    )
    result = await replay_capability(cap, {"member_id": "40300"}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path))

    assert result.status is OutcomeStatus.HARD_FAILURE
    assert result.business_outcome is None
    assert "permission" in result.failure.observed.lower()


async def test_recoverable_dialog_is_logged_and_replay_continues(tmp_path):
    cap = two_step_capability()
    surface = ScriptedSurface(
        action_results=[
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search")),
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/1")),
        ],
        recoverable_events=[[], [{"type": "confirm", "message": "Session expiring, continue?", "accepted": False}]],
    )
    result = await replay_capability(cap, {"member_id": "1"}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path))

    assert result.status is OutcomeStatus.SUCCESS
    assert len(result.recoverable_events) == 1
    assert result.recoverable_events[0].condition == "confirm"
    assert result.recoverable_events[0].step_id == "step_1"


async def test_transient_timeout_triggers_one_bounded_retry_then_succeeds(tmp_path):
    cap = two_step_capability()
    surface = ScriptedSurface(
        action_results=[
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search")),
            SurfaceActionResult(success=False, error="Page.goto: Timeout 5000ms exceeded"),
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/40800")),
        ],
    )
    result = await replay_capability(cap, {"member_id": "40800"}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path))

    assert result.status is OutcomeStatus.SUCCESS
    assert len(result.recoverable_events) == 1
    assert result.recoverable_events[0].condition == "transient_timeout"
    step_1_actions = [a for a in surface.actions_received if a.intent == "search_member" and a.type is ActionType.CLICK]
    assert len(step_1_actions) == 2
    assert step_1_actions[1].timeout_ms > step_1_actions[0].timeout_ms


async def test_timeout_retry_that_also_fails_is_a_hard_failure(tmp_path):
    cap = two_step_capability()
    surface = ScriptedSurface(
        action_results=[
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search")),
            SurfaceActionResult(success=False, error="Timeout 5000ms exceeded"),
            SurfaceActionResult(success=False, error="Timeout 20000ms exceeded"),
        ],
    )
    result = await replay_capability(cap, {"member_id": "40800"}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path))

    assert result.status is OutcomeStatus.HARD_FAILURE
    assert len(result.recoverable_events) == 1


async def test_policy_block_stops_replay_immediately(tmp_path):
    cap = two_step_capability()
    surface = ScriptedSurface(
        action_results=[SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search"))]
    )
    policy = make_policy(search_member=IntentRisk(risk=RiskLevel.IRREVERSIBLE, requires_human_confirmation=True, blocked=True))
    result = await replay_capability(cap, {"member_id": "1"}, surface=surface, policy=policy, evidence=make_evidence_logger(tmp_path))

    assert result.status is OutcomeStatus.HARD_FAILURE
    assert "policy" in result.failure.message.lower()
    assert len(surface.actions_received) == 0


async def test_action_requiring_confirmation_is_escalated_not_executed(tmp_path):
    cap = two_step_capability()
    surface = ScriptedSurface(
        action_results=[SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search"))]
    )
    policy = make_policy(search_member=IntentRisk(risk=RiskLevel.IRREVERSIBLE, requires_human_confirmation=True))
    result = await replay_capability(cap, {"member_id": "1"}, surface=surface, policy=policy, evidence=make_evidence_logger(tmp_path))

    assert result.status is OutcomeStatus.ESCALATED
    assert len(surface.actions_received) == 0


async def test_missing_required_parameter_is_a_structured_failure_not_a_crash(tmp_path):
    cap = two_step_capability()
    surface = ScriptedSurface()
    result = await replay_capability(cap, {}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path))

    assert result.status is OutcomeStatus.HARD_FAILURE
    assert result.failure.step_id == "input_validation"
    assert "member_id" in result.failure.observed
    assert len(surface.actions_received) == 0


async def test_unknown_parameter_is_a_structured_failure(tmp_path):
    cap = two_step_capability()
    surface = ScriptedSurface()
    result = await replay_capability(
        cap, {"member_id": "1", "not_a_real_param": "x"}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path)
    )

    assert result.status is OutcomeStatus.HARD_FAILURE
    assert "not_a_real_param" in result.failure.observed


async def test_invalid_typed_parameter_is_a_structured_failure(tmp_path):
    cap = make_capability(
        steps=[
            Step(
                id="step_0",
                description="type deposit",
                action=Action(type=ActionType.TYPE, intent="submit_new_sub_account", value="{opening_deposit}", target=make_locator("Deposit box", "textbox:Deposit")),
            )
        ],
        success_checkpoint=Checkpoint(description="ok", type=CheckpointType.URL_MATCHES, url_pattern=f"{BASE_URL}/*"),
        inputs=[InputParameter(name="opening_deposit", type=ParamType.NUMBER, description="deposit")],
    )
    surface = ScriptedSurface()
    result = await replay_capability(
        cap, {"opening_deposit": "not-a-number"}, surface=surface, policy=make_policy(submit_new_sub_account=IntentRisk(risk=RiskLevel.SAFE)), evidence=make_evidence_logger(tmp_path)
    )

    assert result.status is OutcomeStatus.HARD_FAILURE
    assert result.failure.step_id == "input_validation"
    assert len(surface.actions_received) == 0


def test_no_llm_or_browser_imports_anywhere_in_replay():
    replay_dir = Path(__file__).resolve().parents[2] / "replay"
    banned = ("groq", "anthropic", "playwright")
    for path in replay_dir.glob("*.py"):
        source = path.read_text().lower()
        for name in banned:
            assert f"import {name}" not in source, f"{path.name} imports {name}"
            assert f"from {name}" not in source, f"{path.name} imports from {name}"


async def test_evidence_log_contains_step_and_failure_context(tmp_path):
    cap = two_step_capability()
    surface = ScriptedSurface(
        action_results=[
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/members/search")),
            SurfaceActionResult(success=False, error="no candidate resolved: no match"),
        ]
    )
    logger = make_evidence_logger(tmp_path)
    result = await replay_capability(cap, {"member_id": "1"}, surface=surface, policy=make_policy(), evidence=logger)

    assert result.status is OutcomeStatus.HARD_FAILURE
    types = event_types(logger)
    for expected in ("run_started", "policy_decision", "action", "error_detected", "run_completed"):
        assert expected in types

    error_events = events_by_type(logger, "error_detected")
    assert error_events[0]["step_id"] == "step_1"
    assert "expected" in error_events[0]["data"]
    assert "observed" in error_events[0]["data"]

    (failure_screenshot_dir := logger.dir / "screenshots")
    assert failure_screenshot_dir.exists()
    assert any(failure_screenshot_dir.iterdir())


async def test_declared_outputs_are_extracted_and_returned(tmp_path):
    cap = make_capability(
        steps=[
            Step(
                id="step_0",
                description="click",
                action=Action(type=ActionType.CLICK, intent="search_member", target=make_locator("go", "button:Go")),
            )
        ],
        success_checkpoint=Checkpoint(description="ok", type=CheckpointType.URL_MATCHES, url_pattern=f"{BASE_URL}/*"),
        outputs=[OutputDefinition(name="savings_balance", type=ParamType.STRING, description="balance", source=make_locator("Balance cell", "cell:Balance"))],
    )
    obs_with_extract = make_observation(f"{BASE_URL}/done")
    obs_with_extract = obs_with_extract.model_copy(update={"extracted_text": "$1,204.55"})
    surface = ScriptedSurface(
        action_results=[
            SurfaceActionResult(success=True, observation=make_observation(f"{BASE_URL}/done")),
            SurfaceActionResult(success=True, observation=obs_with_extract),
        ]
    )
    result = await replay_capability(cap, {}, surface=surface, policy=make_policy(), evidence=make_evidence_logger(tmp_path))

    assert result.status is OutcomeStatus.SUCCESS
    assert result.outputs == {"savings_balance": "$1,204.55"}
