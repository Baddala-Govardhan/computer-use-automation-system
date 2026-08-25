import pytest
from pydantic import ValidationError

from agent.llm_client import ActionDecision, DecisionStatus, StubLLMClient
from agent.perception import DiscoveryContext
from core.actions import ActionType
from core.schema import Action, Locator, LocatorCandidate, LocatorStrategy


def make_action() -> Action:
    return Action(
        type=ActionType.CLICK,
        intent="search_member",
        target=Locator(
            description="Search button",
            candidates=[LocatorCandidate(strategy=LocatorStrategy.ROLE, value="button:Search")],
        ),
    )


def make_context() -> DiscoveryContext:
    return DiscoveryContext(goal="g", url="http://x/", accessibility_snapshot='- button "Search"')


def test_continue_without_action_is_rejected():
    with pytest.raises(ValidationError):
        ActionDecision(status=DecisionStatus.CONTINUE, reasoning="x", action=None)


def test_done_without_action_is_valid():
    decision = ActionDecision(status=DecisionStatus.DONE, reasoning="finished")
    assert decision.action is None


def test_stuck_without_action_is_valid():
    decision = ActionDecision(status=DecisionStatus.STUCK, reasoning="no matching element")
    assert decision.action is None


async def test_stub_llm_client_plays_back_script_in_order():
    d1 = ActionDecision(status=DecisionStatus.CONTINUE, reasoning="first", action=make_action())
    d2 = ActionDecision(status=DecisionStatus.DONE, reasoning="second")
    stub = StubLLMClient([d1, d2])

    ctx = make_context()
    assert await stub.decide(ctx) is d1
    assert await stub.decide(ctx) is d2
    assert stub.received_contexts == [ctx, ctx]


async def test_stub_llm_client_raises_when_script_is_exhausted():
    stub = StubLLMClient([ActionDecision(status=DecisionStatus.DONE, reasoning="only one")])
    await stub.decide(make_context())
    with pytest.raises(IndexError):
        await stub.decide(make_context())
