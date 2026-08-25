from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunType(str, Enum):
    DISCOVERY = "discovery"
    REPLAY = "replay"


class EvidenceEventType(str, Enum):
    RUN_STARTED = "run_started"
    OBSERVATION = "observation"
    DECISION = "decision"
    ACTION = "action"
    POLICY_DECISION = "policy_decision"
    CHECKPOINT = "checkpoint"
    ERROR_DETECTED = "error_detected"
    RECOVERY_ATTEMPTED = "recovery_attempted"
    ESCALATION_RAISED = "escalation_raised"
    CONTROL_TRANSFER = "control_transfer"
    HUMAN_ACTION = "human_action"
    RUN_COMPLETED = "run_completed"


class EvidenceRefKind(str, Enum):
    SCREENSHOT = "screenshot"
    DOM_SNAPSHOT = "dom_snapshot"
    TRACE = "trace"


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: EvidenceRefKind
    path: str


class EvidenceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    seq: int
    timestamp: datetime
    type: EvidenceEventType
    step_id: str | None = None
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    refs: list[EvidenceRef] = Field(default_factory=list)


class RunContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    run_type: RunType
    target: str
    started_at: datetime
    goal: str | None = None
    capability_id: str | None = None
    capability_version: str | None = None


def new_run_id(run_type: RunType) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:6]
    return f"{run_type.value}-{ts}-{suffix}"


_SENSITIVE_KEYS = {
    "password",
    "authorization",
    "cookie",
    "set-cookie",
    "token",
    "api_key",
    "apikey",
    "secret",
    "ssn",
    "social_security_number",
}

_REDACT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("card_number", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("bearer_token", re.compile(r"(?i)\b(bearer|token|api[_-]?key)\s*[:=]\s*\S+")),
    ("account_number", re.compile(r"\b\d{9,17}\b")),
]


def _redact_str(value: str) -> str:
    for name, pattern in _REDACT_PATTERNS:
        value = pattern.sub(f"[REDACTED:{name}]", value)
    return value


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: "[REDACTED]" if k.lower() in _SENSITIVE_KEYS else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _redact_str(value)
    return value


class EvidenceLogger:
    def __init__(self, base_dir: Path, context: RunContext):
        self.run_id = context.run_id
        self.dir = base_dir / context.run_id
        self.screenshots_dir = self.dir / "screenshots"
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.dir / "log.jsonl"
        self._seq = 0
        self._context = context
        (self.dir / "context.json").write_text(context.model_dump_json(indent=2))

    def log(
        self,
        type: EvidenceEventType,
        summary: str,
        *,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
        refs: list[EvidenceRef] | None = None,
    ) -> EvidenceEvent:
        self._seq += 1
        event = EvidenceEvent(
            run_id=self.run_id,
            seq=self._seq,
            timestamp=datetime.now(timezone.utc),
            type=type,
            step_id=step_id,
            summary=summary,
            data=redact(data or {}),
            refs=refs or [],
        )
        with self._log_path.open("a") as f:
            f.write(event.model_dump_json() + "\n")
        return event

    def save_screenshot(self, png_bytes: bytes, name: str) -> EvidenceRef:
        rel_path = f"screenshots/{name}.png"
        (self.dir / rel_path).write_bytes(png_bytes)
        return EvidenceRef(kind=EvidenceRefKind.SCREENSHOT, path=rel_path)

    def save_dom_snapshot(self, html: str, name: str) -> EvidenceRef:
        rel_path = f"screenshots/{name}.html"
        (self.dir / rel_path).write_text(html)
        return EvidenceRef(kind=EvidenceRefKind.DOM_SNAPSHOT, path=rel_path)

    def save_artifact(self, capability_json: str) -> None:
        (self.dir / "artifact.json").write_text(capability_json)

    def save_result(self, result_json: str) -> None:
        (self.dir / "result.json").write_text(result_json)
