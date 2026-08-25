from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.actions import ActionType


class LocatorStrategy(str, Enum):
    ROLE = "role"
    TEST_ID = "test_id"
    LABEL = "label"
    TEXT = "text"
    CSS = "css"
    XPATH = "xpath"
    COORDINATES = "coordinates"


class LocatorCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: LocatorStrategy
    value: str
    frame: str | None = Field(
        default=None,
        description="Dotted frame/iframe path to the target, if not in the top-level document.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: str | None = Field(
        default=None, description="Why this candidate was chosen / how robust it's expected to be."
    )


class Locator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(description="Human-readable name, e.g. 'Search button'.")
    candidates: list[LocatorCandidate] = Field(min_length=1)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ActionType
    intent: str = Field(
        description="Semantic label used for risk-policy lookup, e.g. 'search_member', 'submit_transfer'."
    )
    target: Locator | None = Field(default=None, description="Unused for NAVIGATE/WAIT.")
    value: str | None = Field(default=None, description="Text for TYPE, option for SELECT, key name for PRESS_KEY.")
    url: str | None = Field(default=None, description="Destination for NAVIGATE; may contain {param} placeholders.")
    timeout_ms: int = Field(default=5000, ge=0)

    @model_validator(mode="after")
    def _target_required_for_element_actions(self) -> "Action":
        needs_target = self.type in {
            ActionType.CLICK,
            ActionType.TYPE,
            ActionType.SELECT,
            ActionType.EXTRACT,
        }
        if needs_target and self.target is None:
            raise ValueError(f"action type {self.type} requires a target locator")
        if self.type is ActionType.NAVIGATE and not self.url:
            raise ValueError("NAVIGATE action requires a url")
        return self


class CheckpointType(str, Enum):
    ELEMENT_VISIBLE = "element_visible"
    ELEMENT_ABSENT = "element_absent"
    TEXT_PRESENT = "text_present"
    URL_MATCHES = "url_matches"
    EXTRACTION_NONEMPTY = "extraction_nonempty"


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    type: CheckpointType
    locator: Locator | None = None
    expected_text: str | None = None
    url_pattern: str | None = None

    @model_validator(mode="after")
    def _fields_match_type(self) -> "Checkpoint":
        if self.type in {CheckpointType.ELEMENT_VISIBLE, CheckpointType.ELEMENT_ABSENT} and not self.locator:
            raise ValueError(f"checkpoint type {self.type} requires a locator")
        if self.type is CheckpointType.TEXT_PRESENT and not self.expected_text:
            raise ValueError("TEXT_PRESENT checkpoint requires expected_text")
        if self.type is CheckpointType.URL_MATCHES and not self.url_pattern:
            raise ValueError("URL_MATCHES checkpoint requires url_pattern")
        return self


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    action: Action
    checkpoint: Checkpoint | None = Field(
        default=None, description="Verified immediately after this step's action executes."
    )
    max_retries: int = Field(default=0, ge=0, description="Bounded, deterministic retries (e.g. transient slow load).")


class ParamType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ENUM = "enum"


class InputParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: ParamType
    required: bool = True
    description: str
    example: Any | None = None
    sensitive: bool = Field(default=False, description="Caller-supplied secret/PII; never persisted verbatim.")
    enum_values: list[str] | None = None

    @model_validator(mode="after")
    def _enum_values_present(self) -> "InputParameter":
        if self.type is ParamType.ENUM and not self.enum_values:
            raise ValueError("ENUM parameter requires enum_values")
        return self


class OutputDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: ParamType
    description: str
    source: Locator = Field(description="Where this output is read from on the page at extraction time.")
    sensitive: bool = Field(default=False, description="Regulated/PII output; redacted in logs, still returned to caller.")


class TargetRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: str = Field(description="Vendor/app identifier, e.g. 'mock_bank_app' - the join key for multi-tenant reuse.")
    base_url: str
    tenant: str | None = Field(default=None, description="Set when a capability has been specialized for one tenant.")


class CapabilityMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: datetime
    discovery_run_id: str = Field(description="evidence/<run_id>/ this capability was compiled from.")
    model_used: str
    author: str = Field(default="discovery-agent")


class Capability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    version: str = Field(description="Semver-ish; bump on any change to steps/inputs/outputs.")
    description: str
    target: TargetRef
    inputs: list[InputParameter] = Field(default_factory=list)
    outputs: list[OutputDefinition] = Field(default_factory=list)
    steps: list[Step] = Field(min_length=1)
    success_checkpoint: Checkpoint
    metadata: CapabilityMetadata

    @field_validator("steps")
    @classmethod
    def _unique_step_ids(cls, steps: list[Step]) -> list[Step]:
        ids = [s.id for s in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step ids must be unique within a capability")
        return steps
