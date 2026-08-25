from __future__ import annotations

import fnmatch
import re
from typing import Any

from core.actions import ActionType
from core.evidence import EvidenceEventType, EvidenceLogger
from core.outcomes import (
    BusinessOutcome,
    HardFailure,
    OutcomeStatus,
    RecoverableAction,
    RecoverableEvent,
    RunResult,
)
from core.policy import PolicyChecker
from core.schema import Action, Capability, Checkpoint, CheckpointType, InputParameter, Locator, ParamType, Step
from core.surface import Surface, SurfaceObservation
from replay.detectors import detect
from replay.locators import describe_locator_failure, locator_search_terms
from replay.recovery import TIMEOUT_RETRY_MULTIPLIER, is_timeout_error

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


class InputValidationError(ValueError):
    pass


def _coerce_param(param: InputParameter, raw: Any) -> str:
    if param.type is ParamType.INTEGER:
        try:
            int(str(raw))
        except (TypeError, ValueError):
            raise InputValidationError(f"parameter '{param.name}' must be an integer, got {raw!r}") from None
    elif param.type is ParamType.NUMBER:
        try:
            float(str(raw))
        except (TypeError, ValueError):
            raise InputValidationError(f"parameter '{param.name}' must be a number, got {raw!r}") from None
    elif param.type is ParamType.BOOLEAN:
        if str(raw).lower() not in ("true", "false"):
            raise InputValidationError(f"parameter '{param.name}' must be a boolean, got {raw!r}")
    elif param.type is ParamType.ENUM:
        if str(raw) not in (param.enum_values or []):
            raise InputValidationError(f"parameter '{param.name}' must be one of {param.enum_values}, got {raw!r}")
    return str(raw)


def _validate_and_coerce_inputs(capability: Capability, inputs: dict[str, Any]) -> dict[str, str]:
    remaining = dict(inputs)
    coerced: dict[str, str] = {}
    for param in capability.inputs:
        if param.name not in remaining:
            if param.required:
                raise InputValidationError(f"missing required parameter '{param.name}'")
            continue
        coerced[param.name] = _coerce_param(param, remaining.pop(param.name))
    if remaining:
        raise InputValidationError(
            f"unknown parameter(s) for capability '{capability.id}': {', '.join(sorted(remaining))}"
        )
    return coerced


def _substitute_text(text: str | None, values: dict[str, str]) -> str | None:
    if text is None:
        return None

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise InputValidationError(f"step references undeclared parameter '{{{name}}}'")
        return values[name]

    return _PLACEHOLDER_RE.sub(repl, text)


def _substitute_locator(locator: Locator | None, values: dict[str, str]) -> Locator | None:
    if locator is None:
        return None
    return locator.model_copy(
        update={
            "description": _substitute_text(locator.description, values),
            "candidates": [
                candidate.model_copy(update={"value": _substitute_text(candidate.value, values)})
                for candidate in locator.candidates
            ],
        }
    )


def _substitute_action(action: Action, values: dict[str, str]) -> Action:
    return action.model_copy(
        update={
            "target": _substitute_locator(action.target, values),
            "value": _substitute_text(action.value, values),
            "url": _substitute_text(action.url, values),
        }
    )


def verify_checkpoint(checkpoint: Checkpoint, observation: SurfaceObservation) -> tuple[bool, str, str]:
    if checkpoint.type is CheckpointType.URL_MATCHES:
        assert checkpoint.url_pattern is not None
        passed = fnmatch.fnmatch(observation.url, checkpoint.url_pattern)
        return passed, checkpoint.url_pattern, observation.url
    if checkpoint.type is CheckpointType.TEXT_PRESENT:
        assert checkpoint.expected_text is not None
        passed = checkpoint.expected_text.lower() in observation.accessibility_tree.lower()
        return passed, f"text containing '{checkpoint.expected_text}'", "present" if passed else "not present"
    if checkpoint.type is CheckpointType.EXTRACTION_NONEMPTY:
        passed = bool(observation.extracted_text and observation.extracted_text.strip())
        return passed, "non-empty extraction", repr(observation.extracted_text)
    if checkpoint.type is CheckpointType.ELEMENT_VISIBLE:
        assert checkpoint.locator is not None
        terms = [t.lower() for t in locator_search_terms(checkpoint.locator)]
        passed = any(t in observation.accessibility_tree.lower() for t in terms)
        return passed, f"'{checkpoint.locator.description}' visible", "present" if passed else "not present"
    if checkpoint.type is CheckpointType.ELEMENT_ABSENT:
        assert checkpoint.locator is not None
        terms = [t.lower() for t in locator_search_terms(checkpoint.locator)]
        passed = not any(t in observation.accessibility_tree.lower() for t in terms)
        return passed, f"'{checkpoint.locator.description}' absent", "absent" if passed else "still present"
    raise AssertionError(f"unhandled checkpoint type: {checkpoint.type}")


class OutputExtractionError(RuntimeError):
    def __init__(self, output_name: str, message: str):
        self.output_name = output_name
        super().__init__(message)


async def extract_outputs(capability: Capability, surface: Surface, evidence: EvidenceLogger) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for output_def in capability.outputs:
        extract_action = Action(type=ActionType.EXTRACT, intent=f"extract_{output_def.name}", target=output_def.source)
        result = await surface.act(extract_action)
        evidence.log(
            EvidenceEventType.ACTION, f"extract ({output_def.name})", data={"success": result.success, "error": result.error}
        )
        if not result.success or result.observation is None or result.observation.extracted_text is None:
            raise OutputExtractionError(output_def.name, result.error or "no text extracted")
        outputs[output_def.name] = result.observation.extracted_text
    return outputs


async def _fail(
    evidence: EvidenceLogger,
    surface: Surface,
    capability: Capability,
    *,
    step_id: str,
    expected: str,
    observed: str,
    message: str,
    recoverable_log: list[RecoverableEvent] | None = None,
) -> RunResult:
    evidence_ref = None
    try:
        evidence_ref = evidence.save_screenshot(await surface.snapshot(), name=f"failure_{step_id}")
    except Exception:
        pass

    evidence.log(
        EvidenceEventType.ERROR_DETECTED,
        message,
        step_id=step_id,
        data={"expected": expected, "observed": observed},
        refs=[evidence_ref] if evidence_ref else [],
    )
    failure = HardFailure(step_id=step_id, expected=expected, observed=observed, message=message, evidence_ref=evidence_ref)
    result = RunResult(
        run_id=evidence.run_id,
        status=OutcomeStatus.HARD_FAILURE,
        capability_id=capability.id,
        capability_version=capability.version,
        failure=failure,
        recoverable_events=recoverable_log or [],
    )
    evidence.log(EvidenceEventType.RUN_COMPLETED, "replay failed", data={"status": result.status.value, "step_id": step_id})
    return result


async def replay_capability(
    capability: Capability,
    inputs: dict[str, Any],
    *,
    surface: Surface,
    policy: PolicyChecker,
    evidence: EvidenceLogger,
) -> RunResult:
    recoverable_log: list[RecoverableEvent] = []

    evidence.log(
        EvidenceEventType.RUN_STARTED,
        f"replaying capability '{capability.id}' v{capability.version}",
        data={"capability_id": capability.id, "capability_version": capability.version, "input_names": sorted(inputs)},
    )

    try:
        values = _validate_and_coerce_inputs(capability, inputs)
        substituted: list[tuple[Step, Action]] = [(step, _substitute_action(step.action, values)) for step in capability.steps]
    except InputValidationError as e:
        return await _fail(
            evidence,
            surface,
            capability,
            step_id="input_validation",
            expected="inputs matching the capability's declared parameter contract",
            observed=str(e),
            message=str(e),
        )

    last_observation: SurfaceObservation | None = None

    for step, action in substituted:
        policy_decision = policy.check_action(action, current_url=surface.current_url())
        evidence.log(
            EvidenceEventType.POLICY_DECISION,
            policy_decision.reason,
            step_id=step.id,
            data={
                "allowed": policy_decision.allowed,
                "risk_level": policy_decision.risk_level.value,
                "requires_human_confirmation": policy_decision.requires_human_confirmation,
            },
        )

        if not policy_decision.allowed:
            return await _fail(
                evidence,
                surface,
                capability,
                step_id=step.id,
                expected="action allowed by current policy",
                observed=policy_decision.reason,
                message=f"blocked by policy: {policy_decision.reason}",
                recoverable_log=recoverable_log,
            )

        if policy_decision.requires_human_confirmation:
            evidence.log(
                EvidenceEventType.ESCALATION_RAISED,
                f"action requires human confirmation: {action.intent}",
                step_id=step.id,
            )
            failure = HardFailure(
                step_id=step.id,
                expected="no confirmation required, or a human present to grant it",
                observed=f"intent '{action.intent}' requires human confirmation",
                message=f"replay stopped: '{action.intent}' requires human confirmation before proceeding",
            )
            result = RunResult(
                run_id=evidence.run_id,
                status=OutcomeStatus.ESCALATED,
                capability_id=capability.id,
                capability_version=capability.version,
                failure=failure,
                recoverable_events=recoverable_log,
            )
            evidence.log(EvidenceEventType.RUN_COMPLETED, "replay escalated", data={"status": result.status.value, "step_id": step.id})
            return result

        action_result = await surface.act(action)
        evidence.log(
            EvidenceEventType.ACTION,
            f"{action.type.value} ({action.intent})",
            step_id=step.id,
            data={"success": action_result.success, "error": action_result.error},
        )

        if not action_result.success and is_timeout_error(action_result.error):
            retry_action = action.model_copy(update={"timeout_ms": action.timeout_ms * TIMEOUT_RETRY_MULTIPLIER})
            recoverable_log.append(
                RecoverableEvent(step_id=step.id, condition="transient_timeout", action_taken=RecoverableAction.RETRIED_LOAD, attempt=1)
            )
            evidence.log(
                EvidenceEventType.RECOVERY_ATTEMPTED,
                "retrying with a longer timeout after a transient timeout",
                step_id=step.id,
                data={"new_timeout_ms": retry_action.timeout_ms},
            )
            action_result = await surface.act(retry_action)
            evidence.log(
                EvidenceEventType.ACTION,
                f"{action.type.value} ({action.intent}) [retry]",
                step_id=step.id,
                data={"success": action_result.success, "error": action_result.error},
            )

        for event in await surface.take_recoverable_events():
            recoverable_log.append(
                RecoverableEvent(
                    step_id=step.id,
                    condition=str(event.get("type", "unknown")),
                    action_taken=RecoverableAction.DISMISSED_DIALOG,
                    attempt=1,
                )
            )
            evidence.log(EvidenceEventType.RECOVERY_ATTEMPTED, "surface auto-handled a condition", step_id=step.id, data=event)

        if not action_result.success:
            action_description = action.target.description if action.target else action.intent
            expected, observed = describe_locator_failure(action_description, action_result.error)
            return await _fail(
                evidence,
                surface,
                capability,
                step_id=step.id,
                expected=expected,
                observed=observed,
                message=action_result.error or "action failed",
                recoverable_log=recoverable_log,
            )

        observation = action_result.observation or await surface.observe()
        last_observation = observation

        detection = detect(observation)
        if detection.hard_failure_reason:
            return await _fail(
                evidence,
                surface,
                capability,
                step_id=step.id,
                expected="no error/permission condition on the resulting page",
                observed=detection.hard_failure_reason,
                message=detection.hard_failure_reason,
                recoverable_log=recoverable_log,
            )
        if detection.business_outcome:
            evidence.log(
                EvidenceEventType.CHECKPOINT,
                f"business outcome detected: {detection.business_outcome.code}",
                step_id=step.id,
                data=detection.business_outcome.data,
            )
            result = RunResult(
                run_id=evidence.run_id,
                status=OutcomeStatus.BUSINESS_OUTCOME,
                capability_id=capability.id,
                capability_version=capability.version,
                business_outcome=detection.business_outcome,
                recoverable_events=recoverable_log,
            )
            evidence.log(EvidenceEventType.RUN_COMPLETED, "replay ended in a business outcome", data={"status": result.status.value})
            return result

        if step.checkpoint is not None:
            passed, expected, observed = verify_checkpoint(step.checkpoint, observation)
            evidence.log(
                EvidenceEventType.CHECKPOINT,
                step.checkpoint.description,
                step_id=step.id,
                data={"passed": passed, "expected": expected, "observed": observed},
            )
            if not passed:
                return await _fail(
                    evidence,
                    surface,
                    capability,
                    step_id=step.id,
                    expected=expected,
                    observed=observed,
                    message=f"checkpoint failed: {step.checkpoint.description}",
                    recoverable_log=recoverable_log,
                )

    if last_observation is None:
        return await _fail(
            evidence,
            surface,
            capability,
            step_id="final_checkpoint",
            expected="at least one executed step",
            observed="no steps executed",
            message="capability has no steps",
            recoverable_log=recoverable_log,
        )

    passed, expected, observed = verify_checkpoint(capability.success_checkpoint, last_observation)
    evidence.log(
        EvidenceEventType.CHECKPOINT,
        capability.success_checkpoint.description,
        data={"passed": passed, "expected": expected, "observed": observed},
    )
    if not passed:
        return await _fail(
            evidence,
            surface,
            capability,
            step_id="final_checkpoint",
            expected=expected,
            observed=observed,
            message=f"final checkpoint failed: {capability.success_checkpoint.description}",
            recoverable_log=recoverable_log,
        )

    try:
        outputs = await extract_outputs(capability, surface, evidence)
    except OutputExtractionError as e:
        return await _fail(
            evidence,
            surface,
            capability,
            step_id=f"output_{e.output_name}",
            expected=f"output '{e.output_name}' to be extractable",
            observed=str(e),
            message=f"failed to extract declared output '{e.output_name}'",
            recoverable_log=recoverable_log,
        )

    final_result = RunResult(
        run_id=evidence.run_id,
        status=OutcomeStatus.SUCCESS,
        capability_id=capability.id,
        capability_version=capability.version,
        outputs=outputs,
        recoverable_events=recoverable_log,
    )
    evidence.log(EvidenceEventType.RUN_COMPLETED, "replay succeeded", data={"outputs": list(outputs.keys())})
    return final_result
