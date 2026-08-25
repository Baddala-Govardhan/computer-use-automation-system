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
    Locator,
    LocatorCandidate,
    LocatorStrategy,
    OutputDefinition,
    ParamType,
    Step,
    TargetRef,
)
from core.surface import SurfaceActionResult, SurfaceObservation
from escalation.manager import EscalationManager
from escalation.ownership import InvalidOwnershipTransition, Owner, SessionOwnership
from replay.engine import replay_capability

BASE_URL = "http://x"


class FakeSurface:
    def __init__(self, starting_url: str = f"{BASE_URL}/review", confirmation_number: str | None = None):
        self._url = starting_url
        self.actions_received: list[Action] = []
        self.confirmation_number = confirmation_number

    async def observe(self) -> SurfaceObservation:
        return SurfaceObservation(url=self._url, title="t", accessibility_tree='- button "Confirm"', timestamp=datetime.now(timezone.utc))

    async def act(self, action: Action) -> SurfaceActionResult:
        self.actions_received.append(action)
        if action.type is ActionType.CLICK:
            self._url = f"{BASE_URL}/confirmation"
        observation = await self.observe()
        if action.type is ActionType.EXTRACT:
            observation = observation.model_copy(update={"extracted_text": self.confirmation_number})
        return SurfaceActionResult(success=True, observation=observation)

    async def snapshot(self) -> bytes:
        return b"\x89PNGfake"

    def current_url(self) -> str:
        return self._url

    async def close(self) -> None:
        pass

    async def take_recoverable_events(self) -> list[dict[str, Any]]:
        return []


def make_evidence_logger(tmp_path: Path, run_type: RunType = RunType.REPLAY) -> EvidenceLogger:
    ctx = RunContext(
        run_id=new_run_id(run_type),
        run_type=run_type,
        target=BASE_URL,
        started_at=datetime.now(timezone.utc),
        capability_id="confirm_capability",
        capability_version="1.0.0",
    )
    return EvidenceLogger(tmp_path, ctx)


def make_capability() -> Capability:
    return Capability(
        id="confirm_capability",
        name="Confirm",
        version="1.0.0",
        description="test",
        target=TargetRef(app="mock_bank_app", base_url=BASE_URL),
        steps=[
            Step(
                id="step_0",
                description="Confirm",
                action=Action(
                    type=ActionType.CLICK,
                    intent="confirm_new_sub_account",
                    target=Locator(description="Confirm button", candidates=[LocatorCandidate(strategy=LocatorStrategy.ROLE, value="button:Confirm")]),
                ),
            )
        ],
        success_checkpoint=Checkpoint(description="reached confirmation", type=CheckpointType.URL_MATCHES, url_pattern=f"{BASE_URL}/confirmation"),
        metadata=CapabilityMetadata(created_at=datetime.now(timezone.utc), discovery_run_id="d1", model_used="test"),
    )


def make_capability_with_confirmation_output() -> Capability:
    base = make_capability()
    return base.model_copy(
        update={
            "outputs": [
                OutputDefinition(
                    name="confirmation_number",
                    type=ParamType.STRING,
                    description="The confirmation number generated when the new sub-account was created.",
                    source=Locator(
                        description="Confirmation Number cell",
                        candidates=[LocatorCandidate(strategy=LocatorStrategy.XPATH, value="//td[text()='Confirmation Number']/following-sibling::td[1]")],
                    ),
                )
            ]
        }
    )


def event_types(logger: EvidenceLogger) -> list[str]:
    lines = (logger.dir / "log.jsonl").read_text().strip().splitlines()
    return [json.loads(line)["type"] for line in lines]


def events_by_type(logger: EvidenceLogger, type_: str) -> list[dict]:
    lines = (logger.dir / "log.jsonl").read_text().strip().splitlines()
    return [json.loads(line) for line in lines if json.loads(line)["type"] == type_]


def confirm_action() -> Action:
    return Action(
        type=ActionType.CLICK,
        intent="confirm_new_sub_account",
        target=Locator(description="Confirm button", candidates=[LocatorCandidate(strategy=LocatorStrategy.ROLE, value="button:Confirm")]),
    )


async def test_intervention_request_contains_required_context(tmp_path):
    surface = FakeSurface()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    request = await manager.raise_intervention(
        capability=make_capability(), step_id="step_0", reason="requires human confirmation", risk_level="irreversible"
    )

    assert request.run_id == manager.evidence.run_id
    assert request.capability_id == "confirm_capability"
    assert request.capability_version == "1.0.0"
    assert request.step_id == "step_0"
    assert request.reason == "requires human confirmation"
    assert request.current_url == f"{BASE_URL}/review"
    assert request.risk_level == "irreversible"
    assert request.screenshot_ref is not None
    assert request.created_at is not None


async def test_raise_intervention_transfers_ownership_to_human(tmp_path):
    surface = FakeSurface()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    assert manager.ownership.owner is Owner.AUTOMATION

    await manager.raise_intervention(capability=make_capability(), step_id="step_0", reason="x")
    assert manager.ownership.owner is Owner.HUMAN


async def test_automation_cannot_act_while_human_owns_session(tmp_path):
    surface = FakeSurface()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=make_capability(), step_id="step_0", reason="x")

    with pytest.raises(InvalidOwnershipTransition):
        manager.ownership.require_automation()


async def test_human_action_rejected_before_handoff(tmp_path):
    surface = FakeSurface()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    with pytest.raises(InvalidOwnershipTransition):
        await manager.perform_human_action(confirm_action())
    assert surface.actions_received == []


async def test_resume_transfers_ownership_back_to_automation(tmp_path):
    surface = FakeSurface()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=make_capability(), step_id="step_0", reason="x")

    await manager.resume(note="done clicking confirm")
    assert manager.ownership.owner is Owner.AUTOMATION
    assert manager.pending_request is None


async def test_resume_without_prior_handoff_is_rejected(tmp_path):
    surface = FakeSurface()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    with pytest.raises(InvalidOwnershipTransition):
        await manager.resume()


async def test_human_action_operates_the_existing_session(tmp_path):
    surface = FakeSurface()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=make_capability(), step_id="step_0", reason="x")

    result = await manager.perform_human_action(confirm_action())
    assert result.success
    assert surface.actions_received == [confirm_action()]
    assert surface.current_url() == f"{BASE_URL}/confirmation"


async def test_human_actions_are_recorded_distinctly(tmp_path):
    surface = FakeSurface()
    logger = make_evidence_logger(tmp_path)
    manager = EscalationManager(surface, logger)
    await manager.raise_intervention(capability=make_capability(), step_id="step_0", reason="x")
    await manager.perform_human_action(confirm_action())

    assert len(manager.human_actions) == 1
    assert manager.human_actions[0].action.intent == "confirm_new_sub_account"
    assert manager.human_actions[0].success is True
    assert "human_action" in event_types(logger)


async def test_evidence_stays_associated_with_the_same_run(tmp_path):
    surface = FakeSurface()
    logger = make_evidence_logger(tmp_path)
    manager = EscalationManager(surface, logger)
    await manager.raise_intervention(capability=make_capability(), step_id="step_0", reason="x")
    await manager.perform_human_action(confirm_action())
    await manager.resume()

    lines = (logger.dir / "log.jsonl").read_text().strip().splitlines()
    run_ids = {json.loads(line)["run_id"] for line in lines}
    assert run_ids == {logger.run_id}


async def test_no_new_surface_is_created_during_handoff(tmp_path):
    surface = FakeSurface()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=make_capability(), step_id="step_0", reason="x")
    assert manager.surface is surface
    await manager.perform_human_action(confirm_action())
    assert manager.surface is surface
    await manager.resume()
    assert manager.surface is surface


async def test_control_transfer_events_recorded_both_directions(tmp_path):
    surface = FakeSurface()
    logger = make_evidence_logger(tmp_path)
    manager = EscalationManager(surface, logger)
    await manager.raise_intervention(capability=make_capability(), step_id="step_0", reason="x")
    await manager.resume(note="done")

    transfers = events_by_type(logger, "control_transfer")
    assert len(transfers) == 2
    assert transfers[0]["data"]["to"] == "human"
    assert transfers[1]["data"]["to"] == "automation"


async def test_verify_and_complete_checks_state_after_resume(tmp_path):
    surface = FakeSurface()
    capability = make_capability()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=capability, step_id="step_0", reason="x")
    await manager.perform_human_action(confirm_action())
    await manager.resume()

    result = await manager.verify_and_complete(capability)
    assert result.status is OutcomeStatus.SUCCESS
    assert result.outputs == {}


async def test_verify_and_complete_requires_automation_ownership(tmp_path):
    surface = FakeSurface()
    capability = make_capability()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=capability, step_id="step_0", reason="x")
    with pytest.raises(InvalidOwnershipTransition):
        await manager.verify_and_complete(capability)


def make_policy() -> PolicyChecker:
    allowlist = AllowlistConfig(
        allowed_targets=[AllowedTarget(name="t", base_url=BASE_URL, allowed_routes=["*"])],
        allowed_action_types=list(ActionType),
    )
    risk_policy = RiskPolicyConfig(
        default=IntentRisk(risk=RiskLevel.SAFE),
        intents={"confirm_new_sub_account": IntentRisk(risk=RiskLevel.IRREVERSIBLE, requires_human_confirmation=True)},
    )
    return PolicyChecker(allowlist, risk_policy)


async def test_replay_refuses_then_escalation_completes_on_the_same_surface(tmp_path):
    surface = FakeSurface()
    policy = make_policy()
    capability = make_capability()

    replay_logger = make_evidence_logger(tmp_path / "replay", run_type=RunType.REPLAY)
    replay_result = await replay_capability(capability, {}, surface=surface, policy=policy, evidence=replay_logger)

    assert replay_result.status is OutcomeStatus.ESCALATED
    assert surface.actions_received == []
    assert surface.current_url() == f"{BASE_URL}/review"

    escalation_logger = make_evidence_logger(tmp_path / "escalation", run_type=RunType.REPLAY)
    manager = EscalationManager(surface, escalation_logger)
    await manager.raise_intervention(
        capability=capability, step_id=replay_result.failure.step_id, reason=replay_result.failure.message
    )
    await manager.perform_human_action(confirm_action())
    await manager.resume(note="human approved and confirmed")
    final_result = await manager.verify_and_complete(capability)

    assert final_result.status is OutcomeStatus.SUCCESS
    assert surface.current_url() == f"{BASE_URL}/confirmation"
    assert len(manager.human_actions) == 1
    assert manager.human_actions[0].action.intent == "confirm_new_sub_account"


async def test_human_confirmation_reaches_the_confirmation_page(tmp_path):
    surface = FakeSurface()
    capability = make_capability()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=capability, step_id="step_0", reason="x")

    assert surface.current_url() == f"{BASE_URL}/review"
    result = await manager.perform_human_action(confirm_action())

    assert result.success
    assert surface.current_url() == f"{BASE_URL}/confirmation"


async def test_automation_verifies_checkpoint_extracts_number_and_returns_structured_success(tmp_path):
    surface = FakeSurface(confirmation_number="SA-1A2B3C4D")
    capability = make_capability_with_confirmation_output()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=capability, step_id="step_0", reason="requires human confirmation")
    await manager.perform_human_action(confirm_action())
    await manager.resume(note="human confirmed")

    result = await manager.verify_and_complete(capability)

    assert result.status is OutcomeStatus.SUCCESS
    assert result.outputs == {"confirmation_number": "SA-1A2B3C4D"}
    assert result.capability_id == capability.id
    assert result.capability_version == capability.version
    assert manager.surface is surface


async def test_different_confirmation_numbers_are_extracted_correctly_not_hardcoded(tmp_path):
    for expected_number in ("SA-7BDF2140", "SA-00000001"):
        surface = FakeSurface(confirmation_number=expected_number)
        capability = make_capability_with_confirmation_output()
        manager = EscalationManager(surface, make_evidence_logger(tmp_path / expected_number))
        await manager.raise_intervention(capability=capability, step_id="step_0", reason="x")
        await manager.perform_human_action(confirm_action())
        await manager.resume()

        result = await manager.verify_and_complete(capability)

        assert result.outputs["confirmation_number"] == expected_number


async def test_resume_without_reaching_confirmation_page_does_not_produce_success(tmp_path):
    surface = FakeSurface()
    capability = make_capability_with_confirmation_output()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=capability, step_id="step_0", reason="x")
    await manager.resume(note="human says done")

    result = await manager.verify_and_complete(capability)

    assert result.status is not OutcomeStatus.SUCCESS
    assert result.status is OutcomeStatus.HARD_FAILURE
    assert result.outputs == {}
    assert result.failure.observed == f"{BASE_URL}/review"
