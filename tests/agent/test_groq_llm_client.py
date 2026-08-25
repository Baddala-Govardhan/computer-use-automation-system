"""GroqLLMClient tests. Every groq.AsyncGroq call is mocked - these never
touch the network and never need a real GROQ_API_KEY."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import ValidationError

from agent.llm_client import DecisionStatus, GroqLLMClient
from agent.perception import DiscoveryContext


class _FakeFunctionCall:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name: str, arguments_dict: dict):
        self.function = _FakeFunctionCall(name, json.dumps(arguments_dict))


class _FakeToolCallRaw:
    """For arguments that are deliberately not valid JSON."""

    def __init__(self, name: str, raw_arguments: str):
        self.function = _FakeFunctionCall(name, raw_arguments)


class _FakeMessage:
    def __init__(self, tool_calls: list | None = None):
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message: _FakeMessage):
        self.message = message


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeResponse:
    def __init__(self, choices: list, usage: _FakeUsage | None = None):
        self.choices = choices
        self.usage = usage


def make_context(**overrides) -> DiscoveryContext:
    defaults = dict(
        goal="Find member 12345",
        url="http://x/members/search",
        accessibility_snapshot='- heading "Member Search"',
    )
    defaults.update(overrides)
    return DiscoveryContext(**defaults)


@pytest.fixture
def mock_groq():
    with patch("groq.AsyncGroq") as MockGroq:
        mock_instance = MockGroq.return_value
        mock_instance.chat.completions.create = AsyncMock()
        yield mock_instance


def test_missing_api_key_fails_clearly_at_construction(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqLLMClient()


def test_explicit_api_key_bypasses_env_lookup(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch("groq.AsyncGroq") as MockGroq:
        GroqLLMClient(api_key="gsk-test-explicit")
        MockGroq.assert_called_once_with(api_key="gsk-test-explicit")


async def test_decide_converts_a_well_formed_continue_response(mock_groq, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock_groq.chat.completions.create.return_value = _FakeResponse(
        [
            _FakeChoice(
                _FakeMessage(
                    tool_calls=[
                        _FakeToolCall(
                            "computer_action",
                            {
                                "status": "continue",
                                "reasoning": "Search for the member by ID.",
                                "action": {
                                    "type": "type",
                                    "intent": "search_member",
                                    "value": "12345",
                                    "locator": {
                                        "description": "Member ID search box",
                                        "candidates": [{"strategy": "label", "value": "Member ID or Name"}],
                                    },
                                },
                            },
                        )
                    ]
                )
            )
        ]
    )

    client = GroqLLMClient()
    decision = await client.decide(make_context())

    assert decision.status is DecisionStatus.CONTINUE
    assert decision.action is not None
    assert decision.action.intent == "search_member"
    assert decision.action.value == "12345"
    assert decision.action.target is not None
    assert decision.action.target.candidates[0].value == "Member ID or Name"

    call_kwargs = mock_groq.chat.completions.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "function", "function": {"name": "computer_action"}}
    assert call_kwargs["tools"][0]["function"]["name"] == "computer_action"
    assert call_kwargs["messages"][0]["role"] == "system"


async def test_decide_converts_a_done_response_without_action(mock_groq, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock_groq.chat.completions.create.return_value = _FakeResponse(
        [
            _FakeChoice(
                _FakeMessage(
                    tool_calls=[
                        _FakeToolCall(
                            "computer_action",
                            {"status": "done", "reasoning": "Reached the review screen as requested."},
                        )
                    ]
                )
            )
        ]
    )

    client = GroqLLMClient()
    decision = await client.decide(make_context())

    assert decision.status is DecisionStatus.DONE
    assert decision.action is None


async def test_decide_captures_usage_when_reported(mock_groq, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock_groq.chat.completions.create.return_value = _FakeResponse(
        [_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall("computer_action", {"status": "done", "reasoning": "ok"})]))],
        usage=_FakeUsage(prompt_tokens=1200, completion_tokens=80, total_tokens=1280),
    )

    client = GroqLLMClient()
    decision = await client.decide(make_context())

    assert decision.usage == {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280}


async def test_decide_leaves_usage_none_when_not_reported(mock_groq, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock_groq.chat.completions.create.return_value = _FakeResponse(
        [_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall("computer_action", {"status": "done", "reasoning": "ok"})]))],
        usage=None,
    )

    client = GroqLLMClient()
    decision = await client.decide(make_context())

    assert decision.usage is None


async def test_decide_raises_on_malformed_tool_input_missing_required_fields(mock_groq, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock_groq.chat.completions.create.return_value = _FakeResponse(
        [_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall("computer_action", {"status": "continue"})]))]
    )

    client = GroqLLMClient()
    with pytest.raises(ValidationError):
        await client.decide(make_context())


async def test_decide_raises_on_missing_tool_calls(mock_groq, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock_groq.chat.completions.create.return_value = _FakeResponse([_FakeChoice(_FakeMessage(tool_calls=None))])

    client = GroqLLMClient()
    with pytest.raises(ValueError, match="did not include a computer_action tool call"):
        await client.decide(make_context())


async def test_decide_wraps_server_side_schema_validation_failure_as_value_error(mock_groq, monkeypatch):
    """Groq validates the tool call against our JSON schema server-side and
    rejects a malformed generation with HTTP 400 (groq.BadRequestError)
    rather than handing it back to us to parse - observed for real when the
    model nested "reasoning" inside "action" instead of at the top level.
    This must surface as a plain ValueError so agent/loop.py's decision-retry
    path (which only knows about ValidationError/ValueError) can catch it."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    import groq

    fake_response = httpx.Response(
        status_code=400, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    )
    mock_groq.chat.completions.create.side_effect = groq.BadRequestError(
        "Tool call validation failed: missing properties: 'reasoning'", response=fake_response, body=None
    )

    client = GroqLLMClient()
    with pytest.raises(ValueError, match="Groq rejected the tool call"):
        await client.decide(make_context())


async def test_decide_raises_on_malformed_json_arguments(mock_groq, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock_groq.chat.completions.create.return_value = _FakeResponse(
        [_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCallRaw("computer_action", "{not valid json")]))]
    )

    client = GroqLLMClient()
    with pytest.raises(ValueError):  # json.JSONDecodeError is a ValueError subclass
        await client.decide(make_context())


async def test_decide_raises_on_unknown_locator_strategy(mock_groq, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock_groq.chat.completions.create.return_value = _FakeResponse(
        [
            _FakeChoice(
                _FakeMessage(
                    tool_calls=[
                        _FakeToolCall(
                            "computer_action",
                            {
                                "status": "continue",
                                "reasoning": "click it",
                                "action": {
                                    "type": "click",
                                    "intent": "submit",
                                    "locator": {
                                        "description": "x",
                                        "candidates": [{"strategy": "not_a_real_strategy", "value": "y"}],
                                    },
                                },
                            },
                        )
                    ]
                )
            )
        ]
    )

    client = GroqLLMClient()
    with pytest.raises(ValueError):
        await client.decide(make_context())


async def test_screenshot_is_attached_as_image_url_content_when_present(mock_groq, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock_groq.chat.completions.create.return_value = _FakeResponse(
        [_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall("computer_action", {"status": "done", "reasoning": "ok"})]))]
    )

    client = GroqLLMClient()
    await client.decide(make_context(screenshot_png_base64="aGVsbG8="))

    call_kwargs = mock_groq.chat.completions.create.call_args.kwargs
    content_blocks = call_kwargs["messages"][1]["content"]
    image_blocks = [b for b in content_blocks if b.get("type") == "image_url"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"]["url"] == "data:image/png;base64,aGVsbG8="


async def test_screenshot_is_omitted_when_absent(mock_groq, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock_groq.chat.completions.create.return_value = _FakeResponse(
        [_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall("computer_action", {"status": "done", "reasoning": "ok"})]))]
    )

    client = GroqLLMClient()
    await client.decide(make_context())

    call_kwargs = mock_groq.chat.completions.create.call_args.kwargs
    content_blocks = call_kwargs["messages"][1]["content"]
    assert not any(b.get("type") == "image_url" for b in content_blocks)
