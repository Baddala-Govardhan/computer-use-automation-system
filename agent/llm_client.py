from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.perception import DiscoveryContext
from core.actions import ActionType
from core.schema import Action, Locator, LocatorCandidate, LocatorStrategy


class DecisionStatus(str, Enum):
    CONTINUE = "continue"
    DONE = "done"
    STUCK = "stuck"


class ActionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DecisionStatus
    reasoning: str
    action: Action | None = None
    request_screenshot: bool = False
    usage: dict[str, int] | None = Field(
        default=None,
        description=(
            "Token usage for this decision call, when the provider reports it "
            "(prompt_tokens/completion_tokens/total_tokens). Purely informational for "
            "evidence/cost tracking - never required for a decision to be valid."
        ),
    )

    @model_validator(mode="after")
    def _action_required_when_continuing(self) -> "ActionDecision":
        if self.status is DecisionStatus.CONTINUE and self.action is None:
            raise ValueError("status=continue requires an action")
        return self


class LLMClient(Protocol):
    async def decide(self, context: DiscoveryContext) -> ActionDecision: ...


class StubLLMClient:
    def __init__(self, script: list[ActionDecision]):
        self._script = list(script)
        self.received_contexts: list[DiscoveryContext] = []

    async def decide(self, context: DiscoveryContext) -> ActionDecision:
        self.received_contexts.append(context)
        if not self._script:
            raise IndexError(f"StubLLMClient script exhausted after {len(self.received_contexts)} calls")
        return self._script.pop(0)


COMPUTER_ACTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "computer_action",
        "description": (
            "Report the next action to take against the current page, or signal that the goal is "
            "complete or that you're stuck and can't safely proceed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": [s.value for s in DecisionStatus]},
                "reasoning": {"type": "string", "description": "Why this action/status, in one or two sentences."},
                "request_screenshot": {
                    "type": "boolean",
                    "description": "Set true only if the accessibility snapshot isn't enough to decide confidently.",
                },
                "action": {
                    "type": "object",
                    "description": "Required when status is 'continue'.",
                    "properties": {
                        "type": {"type": "string", "enum": [t.value for t in ActionType]},
                        "intent": {
                            "type": "string",
                            "description": "Semantic label for this action, e.g. 'search_member', 'submit_new_sub_account'.",
                        },
                        "value": {
                            "type": "string",
                            "description": "Text for TYPE, option label for SELECT, key name for PRESS_KEY.",
                        },
                        "url": {"type": "string", "description": "Destination for NAVIGATE."},
                        "locator": {
                            "type": "object",
                            "description": "Required for CLICK/TYPE/SELECT/EXTRACT.",
                            "properties": {
                                "description": {"type": "string"},
                                "candidates": {
                                    "type": "array",
                                    "description": (
                                        "Ranked fallback chain, most confident first. If more than one control "
                                        "shares the same role+name (e.g. two identically-labeled buttons in "
                                        "different landmarks), you MUST include a second, more specific "
                                        "candidate (CSS/XPath scoped to the right landmark or section) rather "
                                        "than relying on role+name alone."
                                    ),
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "strategy": {
                                                "type": "string",
                                                "enum": [s.value for s in LocatorStrategy],
                                            },
                                            "value": {
                                                "type": "string",
                                                "description": "For ROLE: 'role:accessible name', e.g. 'button:Continue'.",
                                            },
                                            "notes": {"type": "string"},
                                        },
                                        "required": ["strategy", "value"],
                                    },
                                },
                            },
                            "required": ["description", "candidates"],
                        },
                    },
                    "required": ["type", "intent"],
                },
            },
            "required": ["status", "reasoning"],
        },
    },
}


SYSTEM_PROMPT = """You are operating a legacy internal banking web application on behalf of a \
human operator, one action at a time. Each turn you are shown the goal, the current URL, the \
history of actions taken so far, and the current page's accessibility snapshot (accessible \
roles and names, nested by landmark) - occasionally also a screenshot.

Call the `computer_action` tool exactly once per turn:
- status="continue" with an `action` describing the next step.
- status="done" once the goal has been reached exactly as stated - explain what was achieved.
- status="stuck" if you cannot safely proceed (no matching element, an unrecoverable error, or \
an impossible goal) - explain why.

When choosing a target element, prefer role + accessible name. If the accessibility snapshot \
shows more than one control with the same name in different landmarks (for example a \
"Continue" button that appears in both a navigation sidebar and the main content area), you \
MUST disambiguate: provide a second, more specific fallback candidate (CSS or XPath, scoped to \
the correct landmark or section) ranked after the role candidate, rather than guessing which \
one is correct. Set request_screenshot=true only if the accessibility snapshot genuinely isn't \
enough to decide.

Never invent data that isn't shown to you. Take only the actions necessary to accomplish the \
stated goal, and stop at exactly the point the goal describes - do not proceed past a review or \
confirmation step the goal didn't ask you to complete.

When the goal states a numeric or monetary amount (e.g. "$100", "1,000"), type only the plain \
number the field expects (e.g. "100"), not currency symbols, commas, or other human formatting - \
form fields for amounts almost always expect a plain number, and a formatted value is a common \
cause of a validation error.

After every action, check the resulting accessibility snapshot for a validation or error message \
near the field you just interacted with before deciding what to do next. If one is present, that \
field's value was rejected: correct it (e.g. reformat the amount, fix the input) and address it \
directly - do not repeatedly click the same submit/continue control without changing anything, \
since that will keep failing the same way. If a page you're waiting on genuinely hasn't changed \
after your last action, treat that as a signal something needs to be fixed before proceeding, not \
a reason to retry the identical action.

Every action's `intent` is checked against a risk policy keyed by that exact string, so naming \
matters: use the conventional snake_case name for the underlying operation, not a description of \
the mechanical step. For example, typing a search term into a search field is "search_member" or \
"search_account" (the operation), not "type_search_box" or "enter_query" (the mechanics) - name \
what the action accomplishes in domain terms, the way a person would describe the operation, not \
what UI control you happen to be interacting with.

When a single logical step requires several actions (e.g. filling multiple fields on a form and \
then clicking its submit/continue button), give every action in that sequence the SAME intent, \
named for the step as a whole (e.g. "submit_new_sub_account" for every field you fill and for the \
click that submits it) - not a separate intent per field. Risk is a property of the step being \
performed, not of which control you're touching at that instant."""


class _RawLocatorCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy: str
    value: str
    notes: str | None = None


class _RawLocator(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str
    candidates: list[_RawLocatorCandidate] = Field(min_length=1)


class _RawAction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    intent: str
    value: str | None = None
    url: str | None = None
    locator: _RawLocator | None = None


class _RawToolInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    reasoning: str
    request_screenshot: bool = False
    action: _RawAction | None = None


def _tool_input_to_action(raw: _RawAction) -> Action:
    locator: Locator | None = None
    if raw.locator is not None:
        locator = Locator(
            description=raw.locator.description,
            candidates=[
                LocatorCandidate(strategy=LocatorStrategy(c.strategy), value=c.value, notes=c.notes)
                for c in raw.locator.candidates
            ],
        )
    return Action(type=ActionType(raw.type), intent=raw.intent, target=locator, value=raw.value, url=raw.url)


class GroqLLMClient:
    def __init__(self, model: str = "openai/gpt-oss-120b", api_key: str | None = None):
        import groq

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Set it in your environment or in .env "
                "(see .env.example), or pass api_key= explicitly. GroqLLMClient "
                "cannot be constructed without it - use StubLLMClient for tests and "
                "development that don't need a live model call."
            )
        self._client = groq.AsyncGroq(api_key=resolved_key)
        self._model = model

    async def decide(self, context: DiscoveryContext) -> ActionDecision:
        import groq

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_user_content(context)},
                ],
                tools=[COMPUTER_ACTION_TOOL],
                tool_choice={"type": "function", "function": {"name": "computer_action"}},
            )
        except groq.BadRequestError as e:
            raise ValueError(f"Groq rejected the tool call (schema validation failed server-side): {e}") from e

        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        tool_call = next((tc for tc in tool_calls if tc.function.name == "computer_action"), None)
        if tool_call is None:
            raise ValueError("model response did not include a computer_action tool call")

        raw_dict = json.loads(tool_call.function.arguments)
        raw = _RawToolInput.model_validate(raw_dict)

        usage_obj = getattr(response, "usage", None)
        usage = None
        if usage_obj is not None:
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            }

        return ActionDecision(
            status=DecisionStatus(raw.status),
            reasoning=raw.reasoning,
            action=_tool_input_to_action(raw.action) if raw.action is not None else None,
            request_screenshot=raw.request_screenshot,
            usage=usage,
        )

    def _build_user_content(self, context: DiscoveryContext) -> list[dict[str, Any]]:
        history = (
            "\n".join(
                f"{pa.step_index + 1}. {pa.action.type.value} ({pa.action.intent}) -> "
                f"{'ok' if pa.success else f'FAILED: {pa.error}'}"
                for pa in context.previous_actions
            )
            or "(none yet)"
        )

        text = (
            f"Goal: {context.goal}\n"
            f"Current URL: {context.url}\n"
            f"Previous actions:\n{history}\n\n"
            f"Accessibility snapshot:\n{context.accessibility_snapshot}"
        )
        if context.note:
            text += f"\n\nNote: {context.note}"

        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if context.screenshot_png_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{context.screenshot_png_base64}"},
                }
            )
        return content
