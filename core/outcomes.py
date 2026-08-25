from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.evidence import EvidenceRef


class OutcomeStatus(str, Enum):
    SUCCESS = "success"
    BUSINESS_OUTCOME = "business_outcome"
    HARD_FAILURE = "hard_failure"
    ESCALATED = "escalated"


class BusinessOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Stable machine-readable code, e.g. 'member_not_found'.")
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class RecoverableAction(str, Enum):
    DISMISSED_DIALOG = "dismissed_dialog"
    RETRIED_LOAD = "retried_load"
    WAITED_FOR_ELEMENT = "waited_for_element"
    REFRESHED_SESSION = "refreshed_session"


class RecoverableEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    condition: str = Field(description="Detector code that fired, e.g. 'session_timeout_dialog'.")
    action_taken: RecoverableAction
    attempt: int = Field(ge=1)


class HardFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    expected: str
    observed: str
    message: str
    evidence_ref: EvidenceRef | None = None


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: OutcomeStatus
    capability_id: str
    capability_version: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    business_outcome: BusinessOutcome | None = None
    recoverable_events: list[RecoverableEvent] = Field(default_factory=list)
    failure: HardFailure | None = None
    evidence_ref: EvidenceRef | None = Field(default=None, description="Pointer to evidence/<run_id>/.")

    @model_validator(mode="after")
    def _fields_match_status(self) -> "RunResult":
        if self.status is OutcomeStatus.SUCCESS and (self.business_outcome or self.failure):
            raise ValueError("SUCCESS must not carry a business_outcome or failure")
        if self.status is OutcomeStatus.BUSINESS_OUTCOME and not self.business_outcome:
            raise ValueError("BUSINESS_OUTCOME status requires business_outcome")
        if self.status in (OutcomeStatus.HARD_FAILURE, OutcomeStatus.ESCALATED) and not self.failure:
            raise ValueError(f"{self.status} status requires failure detail")
        return self
