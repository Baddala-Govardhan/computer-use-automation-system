from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict

from core.evidence import EvidenceEventType, EvidenceLogger, EvidenceRef
from core.outcomes import HardFailure, OutcomeStatus, RunResult
from core.schema import Action, Capability
from core.surface import Surface, SurfaceActionResult
from escalation.ownership import Owner, SessionOwnership
from replay.engine import OutputExtractionError, extract_outputs, verify_checkpoint


class InterventionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    capability_id: str
    capability_version: str
    step_id: str
    reason: str
    current_url: str
    risk_level: str | None = None
    screenshot_ref: EvidenceRef | None = None
    created_at: datetime


class HumanAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    action: Action
    success: bool
    error: str | None = None
    timestamp: datetime


class EscalationManager:
    def __init__(self, surface: Surface, evidence: EvidenceLogger, ownership: SessionOwnership | None = None):
        self.surface = surface
        self.evidence = evidence
        self.ownership = ownership or SessionOwnership()
        self.pending_request: InterventionRequest | None = None
        self.human_actions: list[HumanAction] = []

    async def raise_intervention(
        self, *, capability: Capability, step_id: str, reason: str, risk_level: str | None = None
    ) -> InterventionRequest:
        self.ownership.require_automation()

        observation = await self.surface.observe()
        screenshot_ref = self.evidence.save_screenshot(await self.surface.snapshot(), name=f"escalation_{step_id}")

        request = InterventionRequest(
            run_id=self.evidence.run_id,
            capability_id=capability.id,
            capability_version=capability.version,
            step_id=step_id,
            reason=reason,
            current_url=observation.url,
            risk_level=risk_level,
            screenshot_ref=screenshot_ref,
            created_at=datetime.now(timezone.utc),
        )
        self.pending_request = request
        self.evidence.log(
            EvidenceEventType.ESCALATION_RAISED,
            reason,
            step_id=step_id,
            data=request.model_dump(mode="json"),
            refs=[screenshot_ref],
        )

        self.ownership.transfer_to_human(reason=reason)
        self.evidence.log(
            EvidenceEventType.CONTROL_TRANSFER,
            "ownership: automation -> human",
            step_id=step_id,
            data={"reason": reason, "to": Owner.HUMAN.value},
        )
        return request

    async def perform_human_action(self, action: Action) -> SurfaceActionResult:
        self.ownership.require_human()

        result = await self.surface.act(action)
        step_id = self.pending_request.step_id if self.pending_request else "unknown"
        self.human_actions.append(
            HumanAction(
                step_id=step_id, action=action, success=result.success, error=result.error, timestamp=datetime.now(timezone.utc)
            )
        )
        self.evidence.log(
            EvidenceEventType.HUMAN_ACTION,
            f"human performed {action.type.value} ({action.intent})",
            step_id=step_id,
            data={"success": result.success, "error": result.error},
        )
        return result

    async def resume(self, *, note: str = "human signaled resume") -> None:
        self.ownership.require_human()
        step_id = self.pending_request.step_id if self.pending_request else "unknown"

        self.ownership.transfer_to_automation(reason=note)
        self.evidence.log(
            EvidenceEventType.CONTROL_TRANSFER,
            "ownership: human -> automation",
            step_id=step_id,
            data={"note": note, "to": Owner.AUTOMATION.value},
        )
        self.pending_request = None

    async def verify_and_complete(self, capability: Capability) -> RunResult:
        self.ownership.require_automation()
        observation = await self.surface.observe()
        passed, expected, observed = verify_checkpoint(capability.success_checkpoint, observation)
        self.evidence.log(
            EvidenceEventType.CHECKPOINT,
            capability.success_checkpoint.description,
            data={"passed": passed, "expected": expected, "observed": observed, "verified_after_handoff": True},
        )

        if not passed:
            failure = HardFailure(
                step_id="post_handoff_checkpoint",
                expected=expected,
                observed=observed,
                message=f"post-handoff checkpoint failed: {capability.success_checkpoint.description}",
            )
            self.evidence.log(EvidenceEventType.ERROR_DETECTED, failure.message, data={"expected": expected, "observed": observed})
            result = RunResult(
                run_id=self.evidence.run_id,
                status=OutcomeStatus.HARD_FAILURE,
                capability_id=capability.id,
                capability_version=capability.version,
                failure=failure,
            )
            self.evidence.log(EvidenceEventType.RUN_COMPLETED, "post-handoff completion failed", data={"status": result.status.value})
            return result

        try:
            outputs = await extract_outputs(capability, self.surface, self.evidence)
        except OutputExtractionError as e:
            failure = HardFailure(
                step_id=f"output_{e.output_name}",
                expected=f"output '{e.output_name}' to be extractable",
                observed=str(e),
                message=f"failed to extract declared output '{e.output_name}' after handoff",
            )
            self.evidence.log(EvidenceEventType.ERROR_DETECTED, failure.message)
            result = RunResult(
                run_id=self.evidence.run_id,
                status=OutcomeStatus.HARD_FAILURE,
                capability_id=capability.id,
                capability_version=capability.version,
                failure=failure,
            )
            self.evidence.log(EvidenceEventType.RUN_COMPLETED, "post-handoff completion failed", data={"status": result.status.value})
            return result

        result = RunResult(
            run_id=self.evidence.run_id,
            status=OutcomeStatus.SUCCESS,
            capability_id=capability.id,
            capability_version=capability.version,
            outputs=outputs,
        )
        self.evidence.log(
            EvidenceEventType.RUN_COMPLETED, "post-handoff completion succeeded", data={"outputs": list(outputs.keys())}
        )
        return result
