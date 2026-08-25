from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import ValidationError

from agent.llm_client import ActionDecision, DecisionStatus, LLMClient
from agent.perception import PreviousAction, build_context, should_attach_screenshot
from core.evidence import EvidenceEventType, EvidenceLogger
from core.policy import PolicyChecker, PolicyDecision
from core.surface import Surface, SurfaceActionResult


class StopReason(str, Enum):
    GOAL_COMPLETE = "goal_complete"
    STUCK = "stuck"
    POLICY_BLOCKED = "policy_blocked"
    CONFIRMATION_REQUIRED = "confirmation_required"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    DEAD_END = "dead_end"


@dataclass
class DiscoveryStep:
    index: int
    decision: ActionDecision
    policy_decision: PolicyDecision
    action_result: SurfaceActionResult
    recoverable_events: list[dict] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    stop_reason: StopReason
    steps: list[DiscoveryStep]
    final_reasoning: str

    @property
    def succeeded(self) -> bool:
        return self.stop_reason is StopReason.GOAL_COMPLETE


async def run_discovery(
    *,
    goal: str,
    surface: Surface,
    llm: LLMClient,
    policy: PolicyChecker,
    evidence: EvidenceLogger,
    max_steps: int = 20,
    max_consecutive_failures: int = 3,
    max_decision_retries: int = 2,
    max_dead_end_steps: int = 5,
) -> DiscoveryResult:
    steps: list[DiscoveryStep] = []
    previous_actions: list[PreviousAction] = []
    consecutive_failures = 0
    last_action_failed = False
    pending_dialog: dict | None = None
    last_observation_fingerprint: tuple[str, str, str | None] | None = None
    no_progress_streak = 0

    for step_index in range(max_steps):
        observation = await surface.observe()

        fingerprint = (observation.url, observation.accessibility_tree, observation.extracted_text)
        no_progress_streak = no_progress_streak + 1 if fingerprint == last_observation_fingerprint else 1
        last_observation_fingerprint = fingerprint
        if no_progress_streak > max_dead_end_steps:
            evidence.log(
                EvidenceEventType.ERROR_DETECTED,
                f"dead end: no observable state change across {no_progress_streak} consecutive observations at {observation.url}",
                step_id=str(step_index),
            )
            return DiscoveryResult(
                stop_reason=StopReason.DEAD_END,
                steps=steps,
                final_reasoning=(
                    f"no progress: no observable state change across {no_progress_streak} "
                    f"consecutive observations at {observation.url}"
                ),
            )

        evidence.log(
            EvidenceEventType.OBSERVATION,
            f"observed {observation.url}",
            step_id=str(step_index),
            data={"url": observation.url, "title": observation.title},
        )

        attach_screenshot = should_attach_screenshot(
            last_action_failed=last_action_failed,
            unexpected_dialog=pending_dialog is not None,
            accessibility_snapshot=observation.accessibility_tree,
        )
        screenshot_png = await surface.snapshot() if attach_screenshot else None
        note = None
        if pending_dialog is not None:
            note = f"A dialog just fired and was auto-resolved: {pending_dialog}"
        context = build_context(
            goal=goal,
            observation=observation,
            previous_actions=previous_actions,
            screenshot_png=screenshot_png,
            note=note,
        )

        decision = None
        last_decision_error: Exception | None = None
        for attempt in range(max_decision_retries + 1):
            try:
                decision = await llm.decide(context)
                break
            except (ValidationError, ValueError) as e:
                last_decision_error = e
                evidence.log(
                    EvidenceEventType.ERROR_DETECTED,
                    f"model returned an invalid decision (attempt {attempt + 1}/{max_decision_retries + 1})",
                    step_id=str(step_index),
                    data={"error": str(e)},
                )
        if decision is None:
            return DiscoveryResult(
                stop_reason=StopReason.STUCK,
                steps=steps,
                final_reasoning=f"invalid model output after {max_decision_retries + 1} attempts: {last_decision_error}",
            )

        evidence.log(
            EvidenceEventType.DECISION,
            decision.reasoning,
            step_id=str(step_index),
            data={
                "status": decision.status.value,
                "action_intent": decision.action.intent if decision.action else None,
                "usage": decision.usage,
            },
        )

        if decision.status is DecisionStatus.DONE:
            return DiscoveryResult(stop_reason=StopReason.GOAL_COMPLETE, steps=steps, final_reasoning=decision.reasoning)
        if decision.status is DecisionStatus.STUCK:
            return DiscoveryResult(stop_reason=StopReason.STUCK, steps=steps, final_reasoning=decision.reasoning)

        action = decision.action
        assert action is not None

        policy_decision = policy.check_action(action, current_url=observation.url)
        evidence.log(
            EvidenceEventType.POLICY_DECISION,
            policy_decision.reason,
            step_id=str(step_index),
            data={
                "allowed": policy_decision.allowed,
                "risk_level": policy_decision.risk_level.value,
                "requires_human_confirmation": policy_decision.requires_human_confirmation,
            },
        )

        if not policy_decision.allowed:
            evidence.log(
                EvidenceEventType.ESCALATION_RAISED,
                f"action blocked by policy: {policy_decision.reason}",
                step_id=str(step_index),
            )
            return DiscoveryResult(
                stop_reason=StopReason.POLICY_BLOCKED, steps=steps, final_reasoning=policy_decision.reason
            )

        if policy_decision.requires_human_confirmation:
            evidence.log(
                EvidenceEventType.ESCALATION_RAISED,
                f"action requires human confirmation: {action.intent}",
                step_id=str(step_index),
            )
            return DiscoveryResult(
                stop_reason=StopReason.CONFIRMATION_REQUIRED,
                steps=steps,
                final_reasoning=f"'{action.intent}' requires human confirmation before proceeding",
            )

        action_result = await surface.act(action)

        evidence.log(
            EvidenceEventType.ACTION,
            f"{action.type.value} ({action.intent})",
            step_id=str(step_index),
            data={"success": action_result.success, "error": action_result.error},
        )

        recoverable_events = await surface.take_recoverable_events()
        for event in recoverable_events:
            evidence.log(
                EvidenceEventType.RECOVERY_ATTEMPTED,
                "surface auto-handled a condition",
                step_id=str(step_index),
                data=event,
            )
        pending_dialog = recoverable_events[-1] if recoverable_events else None

        steps.append(
            DiscoveryStep(
                index=step_index,
                decision=decision,
                policy_decision=policy_decision,
                action_result=action_result,
                recoverable_events=recoverable_events,
            )
        )
        previous_actions.append(
            PreviousAction(
                step_index=step_index, action=action, success=action_result.success, error=action_result.error
            )
        )

        last_action_failed = not action_result.success
        consecutive_failures = consecutive_failures + 1 if last_action_failed else 0
        if consecutive_failures >= max_consecutive_failures:
            evidence.log(
                EvidenceEventType.ESCALATION_RAISED,
                f"{consecutive_failures} consecutive action failures",
                step_id=str(step_index),
            )
            return DiscoveryResult(
                stop_reason=StopReason.STUCK,
                steps=steps,
                final_reasoning=f"{consecutive_failures} consecutive action failures",
            )

    evidence.log(EvidenceEventType.ERROR_DETECTED, f"exceeded {max_steps} steps without completing goal")
    return DiscoveryResult(
        stop_reason=StopReason.MAX_STEPS_EXCEEDED,
        steps=steps,
        final_reasoning=f"exceeded {max_steps} steps without completing goal",
    )
