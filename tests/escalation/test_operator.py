from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.actions import ActionType
from core.evidence import EvidenceLogger, RunContext, RunType, new_run_id
from core.schema import (
    Action,
    Capability,
    CapabilityMetadata,
    Checkpoint,
    CheckpointType,
    Locator,
    LocatorCandidate,
    LocatorStrategy,
    Step,
    TargetRef,
)
from core.surface import SurfaceActionResult, SurfaceObservation
from escalation.manager import EscalationManager
from escalation.operator import run_operator_console
from escalation.ownership import Owner

BASE_URL = "http://x"


class FakeSurface:
    def __init__(self, starting_url: str = f"{BASE_URL}/review"):
        self._url = starting_url
        self.actions_received: list[Action] = []

    async def observe(self) -> SurfaceObservation:
        return SurfaceObservation(url=self._url, title="t", accessibility_tree='- button "Confirm"', timestamp=datetime.now(timezone.utc))

    async def act(self, action: Action) -> SurfaceActionResult:
        self.actions_received.append(action)
        if action.type is ActionType.CLICK:
            self._url = f"{BASE_URL}/confirmation"
        return SurfaceActionResult(success=True, observation=await self.observe())

    async def snapshot(self) -> bytes:
        return b"\x89PNGfake"

    def current_url(self) -> str:
        return self._url

    async def close(self) -> None:
        pass

    async def take_recoverable_events(self) -> list[dict[str, Any]]:
        return []


def make_evidence_logger(tmp_path: Path) -> EvidenceLogger:
    ctx = RunContext(
        run_id=new_run_id(RunType.REPLAY),
        run_type=RunType.REPLAY,
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


def scripted_input(commands: list[str]):
    it = iter(commands)

    def _input(prompt: str = "") -> str:
        return next(it)

    return _input


async def test_console_click_then_resume_returns_ownership(tmp_path):
    surface = FakeSurface()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=make_capability(), step_id="step_0", reason="requires human confirmation")

    printed: list[str] = []
    await run_operator_console(
        manager,
        input_fn=scripted_input(["state", "click role button:Confirm", "resume approved"]),
        print_fn=lambda *a: printed.append(" ".join(str(x) for x in a)),
    )

    assert manager.ownership.owner is Owner.AUTOMATION
    assert len(manager.human_actions) == 1
    assert surface.current_url() == f"{BASE_URL}/confirmation"
    assert any("paused at step" in line for line in printed)
    assert any("resumed" in line for line in printed)


async def test_console_screenshot_command_saves_evidence(tmp_path):
    surface = FakeSurface()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=make_capability(), step_id="step_0", reason="x")

    printed: list[str] = []
    await run_operator_console(
        manager, input_fn=scripted_input(["screenshot", "resume"]), print_fn=lambda *a: printed.append(" ".join(str(x) for x in a))
    )

    assert any("saved:" in line for line in printed)


async def test_console_unrecognized_command_does_not_crash(tmp_path):
    surface = FakeSurface()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=make_capability(), step_id="step_0", reason="x")

    printed: list[str] = []
    await run_operator_console(
        manager, input_fn=scripted_input(["not-a-real-command", "resume"]), print_fn=lambda *a: printed.append(" ".join(str(x) for x in a))
    )

    assert manager.ownership.owner is Owner.AUTOMATION
    assert any("unrecognized command" in line for line in printed)


async def test_console_type_command_reaches_the_surface(tmp_path):
    surface = FakeSurface()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=make_capability(), step_id="step_0", reason="x")

    await run_operator_console(
        manager, input_fn=scripted_input(["type role textbox:Notes looks-good", "resume"]), print_fn=lambda *a: None
    )

    assert len(surface.actions_received) == 1
    assert surface.actions_received[0].type is ActionType.TYPE
    assert surface.actions_received[0].value == "looks-good"


async def test_console_exits_cleanly_on_eof_without_resuming(tmp_path):
    surface = FakeSurface()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=make_capability(), step_id="step_0", reason="x")

    def _raise_eof(prompt: str = "") -> str:
        raise EOFError

    await run_operator_console(manager, input_fn=_raise_eof, print_fn=lambda *a: None)

    assert manager.ownership.owner is Owner.HUMAN


async def test_console_click_supports_quoted_css_selector_with_spaces(tmp_path):
    surface = FakeSurface()
    manager = EscalationManager(surface, make_evidence_logger(tmp_path))
    await manager.raise_intervention(capability=make_capability(), step_id="step_0", reason="x")

    await run_operator_console(
        manager,
        input_fn=scripted_input(["click css \"main form button[value='confirm']\"", "resume"]),
        print_fn=lambda *a: None,
    )

    assert len(surface.actions_received) == 1
    assert surface.actions_received[0].target.candidates[0].value == "main form button[value='confirm']"
    assert surface.actions_received[0].target.candidates[0].strategy is LocatorStrategy.CSS
