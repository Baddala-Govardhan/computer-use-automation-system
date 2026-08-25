from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from agent.recorder import RecordedTrace
from core.policy import PolicyChecker
from core.schema import (
    Action,
    Capability,
    CapabilityMetadata,
    Checkpoint,
    CheckpointType,
    InputParameter,
    Locator,
    OutputDefinition,
    ParamType,
    Step,
    TargetRef,
)


@dataclass
class ParameterSpec:
    name: str
    type: ParamType
    literal_value: str
    description: str
    required: bool = True
    example: Any | None = None
    sensitive: bool = False
    enum_values: list[str] | None = None


@dataclass
class OutputSpec:
    name: str
    type: ParamType
    description: str
    source_index: int
    sensitive: bool = False


class CompilationError(ValueError):
    pass


def _param_token(name: str) -> str:
    return f"{{{name}}}"


def _substitute_exact(value: str | None, parameters: list[ParameterSpec]) -> str | None:
    if value is None:
        return None
    for param in parameters:
        if value == param.literal_value:
            return _param_token(param.name)
    return value


def _substitute_within(text: str, parameters: list[ParameterSpec]) -> str:
    result = text
    for param in parameters:
        if param.literal_value and param.literal_value in result:
            result = result.replace(param.literal_value, _param_token(param.name))
    return result


def _generalize_url_pattern(url: str, parameters: list[ParameterSpec]) -> str:
    generalized = url
    for param in parameters:
        if param.literal_value and param.literal_value in generalized:
            generalized = generalized.replace(param.literal_value, "*")
    return generalized


def _substitute_locator(locator: Locator | None, parameters: list[ParameterSpec]) -> Locator | None:
    if locator is None:
        return None
    return locator.model_copy(
        update={
            "description": _substitute_within(locator.description, parameters),
            "candidates": [
                candidate.model_copy(update={"value": _substitute_within(candidate.value, parameters)})
                for candidate in locator.candidates
            ],
        }
    )


def _base_url(trace: RecordedTrace) -> str:
    for step in trace.steps:
        if step.observed_url:
            parsed = urlparse(step.observed_url)
            return f"{parsed.scheme}://{parsed.netloc}"
    raise CompilationError("no observed URL anywhere in the trace - cannot derive the target's base_url")


def compile_capability(
    trace: RecordedTrace,
    *,
    id: str,
    name: str,
    description: str,
    app: str,
    parameters: list[ParameterSpec],
    policy: PolicyChecker,
    version: str = "1.0.0",
    model_used: str = "unknown",
    author: str = "discovery-agent",
    outputs: list[OutputSpec] | None = None,
    success_checkpoint_description: str | None = None,
) -> Capability:
    outputs = outputs or []

    if not trace.steps:
        raise CompilationError("recorded trace has no steps - nothing to compile")
    if not trace.discovery_run_id:
        raise CompilationError("trace has no discovery_run_id - provenance is required to compile a capability")

    compiled_steps: list[Step] = []
    previous_url = _base_url(trace)
    for i, rstep in enumerate(trace.steps):
        policy_decision = policy.check_action(rstep.action, current_url=previous_url)
        if not policy_decision.allowed or policy_decision.requires_human_confirmation:
            raise CompilationError(
                f"step {i} (intent='{rstep.action.intent}') resolves to risk={policy_decision.risk_level.value}, "
                f"requires_human_confirmation={policy_decision.requires_human_confirmation} - a capability compiled "
                f"for unattended replay cannot include a step that needs human sign-off"
            )

        compiled_action = Action(
            type=rstep.action.type,
            intent=rstep.action.intent,
            target=_substitute_locator(rstep.action.target, parameters),
            value=_substitute_exact(rstep.action.value, parameters),
            url=_substitute_within(rstep.action.url, parameters) if rstep.action.url else None,
            timeout_ms=rstep.action.timeout_ms,
        )

        checkpoint = None
        if rstep.observed_url:
            checkpoint = Checkpoint(
                description="URL matches the expected pattern after this step.",
                type=CheckpointType.URL_MATCHES,
                url_pattern=_generalize_url_pattern(rstep.observed_url, parameters),
            )

        if rstep.reasoning:
            step_description = _substitute_within(rstep.reasoning, parameters)
        else:
            step_description = f"{compiled_action.type.value} action for intent '{compiled_action.intent}'"

        compiled_steps.append(
            Step(id=f"step_{i}", description=step_description, action=compiled_action, checkpoint=checkpoint)
        )
        if rstep.observed_url:
            previous_url = rstep.observed_url

    final_step = trace.steps[-1]
    if not final_step.observed_url:
        raise CompilationError("final recorded step has no observed URL - cannot derive a success checkpoint")

    success_checkpoint = Checkpoint(
        description=success_checkpoint_description or "Reached the expected end state for this capability.",
        type=CheckpointType.URL_MATCHES,
        url_pattern=_generalize_url_pattern(final_step.observed_url, parameters),
    )

    input_params = [
        InputParameter(
            name=p.name,
            type=p.type,
            required=p.required,
            description=p.description,
            example=p.example,
            sensitive=p.sensitive,
            enum_values=p.enum_values,
        )
        for p in parameters
    ]

    output_defs: list[OutputDefinition] = []
    for o in outputs:
        source_step = next((s for s in trace.steps if s.source_index == o.source_index), None)
        if source_step is None or source_step.action.target is None:
            raise CompilationError(
                f"output '{o.name}' references source_index={o.source_index}, which has no locator target in the trace"
            )
        output_defs.append(
            OutputDefinition(
                name=o.name,
                type=o.type,
                description=o.description,
                source=_substitute_locator(source_step.action.target, parameters),
                sensitive=o.sensitive,
            )
        )

    return Capability(
        id=id,
        name=name,
        version=version,
        description=description,
        target=TargetRef(app=app, base_url=_base_url(trace)),
        inputs=input_params,
        outputs=output_defs,
        steps=compiled_steps,
        success_checkpoint=success_checkpoint,
        metadata=CapabilityMetadata(
            created_at=datetime.now(timezone.utc),
            discovery_run_id=trace.discovery_run_id,
            model_used=model_used,
            author=author,
        ),
    )
