from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent.loop import DiscoveryResult
from core.evidence import redact
from core.schema import Action


class RecordedStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_index: int = Field(description="Index into the original DiscoveryResult.steps this was recorded from.")
    action: Action
    reasoning: str = Field(
        description=(
            "The model's stated reasoning for this step, passed through core.evidence.redact() - "
            "provenance for a human reviewer, not a raw, unfiltered model transcript."
        )
    )
    observed_url: str | None = Field(
        default=None, description="URL observed immediately after this action executed, when the surface captured one."
    )


class RecordedTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    steps: list[RecordedStep]
    discovery_run_id: str | None = Field(
        default=None, description="evidence/<run_id>/ this trace was recorded from, for provenance."
    )


def record_discovery(result: DiscoveryResult, *, goal: str, discovery_run_id: str | None = None) -> RecordedTrace:
    if not result.succeeded:
        raise ValueError(
            f"cannot record a discovery run that did not complete successfully "
            f"(stop_reason={result.stop_reason.value}): {result.final_reasoning}"
        )

    steps: list[RecordedStep] = []
    for step in result.steps:
        if not step.action_result.success:
            continue

        action = step.decision.action
        assert action is not None

        observed_url = step.action_result.observation.url if step.action_result.observation else None

        steps.append(
            RecordedStep(
                source_index=step.index,
                action=action,
                reasoning=redact(step.decision.reasoning),
                observed_url=observed_url,
            )
        )

    if not steps:
        raise ValueError("discovery completed but recorded no successful actions - nothing to compile")

    return RecordedTrace(goal=goal, steps=steps, discovery_run_id=discovery_run_id)
