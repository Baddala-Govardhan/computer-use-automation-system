from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from core.actions import ActionType
from core.schema import (
    Action,
    Capability,
    CapabilityMetadata,
    Checkpoint,
    CheckpointType,
    InputParameter,
    Locator,
    LocatorCandidate,
    LocatorStrategy,
    OutputDefinition,
    ParamType,
    Step,
    TargetRef,
)


def make_locator(value: str = "Search") -> Locator:
    return Locator(
        description=value,
        candidates=[LocatorCandidate(strategy=LocatorStrategy.ROLE, value=value)],
    )


def make_capability(**overrides) -> Capability:
    defaults = dict(
        id="lookup_member_balance",
        name="Look up member balance",
        version="1.0.0",
        description="Search for a member and read their savings balance.",
        target=TargetRef(app="mock_bank_app", base_url="http://localhost:8000"),
        inputs=[InputParameter(name="member_id", type=ParamType.STRING, description="Member ID")],
        outputs=[
            OutputDefinition(
                name="savings_balance",
                type=ParamType.STRING,
                description="Current savings balance",
                source=make_locator("Balance cell"),
            )
        ],
        steps=[
            Step(
                id="search",
                description="Search for the member",
                action=Action(type=ActionType.TYPE, intent="search_member", value="{member_id}", target=make_locator()),
            )
        ],
        success_checkpoint=Checkpoint(description="Balance visible", type=CheckpointType.EXTRACTION_NONEMPTY),
        metadata=CapabilityMetadata(
            created_at=datetime.now(timezone.utc), discovery_run_id="discovery-x", model_used="claude-sonnet-5"
        ),
    )
    defaults.update(overrides)
    return Capability(**defaults)


def test_capability_round_trips_through_json():
    cap = make_capability()
    restored = Capability.model_validate_json(cap.model_dump_json())
    assert restored == cap


def test_capability_rejects_duplicate_step_ids():
    dup_step = Step(
        id="search",
        description="duplicate id",
        action=Action(type=ActionType.CLICK, intent="search_member", target=make_locator()),
    )
    with pytest.raises(ValidationError):
        make_capability(steps=[dup_step, dup_step])


def test_capability_requires_at_least_one_step():
    with pytest.raises(ValidationError):
        make_capability(steps=[])


@pytest.mark.parametrize("action_type", [ActionType.CLICK, ActionType.TYPE, ActionType.SELECT, ActionType.EXTRACT])
def test_action_requires_target_for_element_actions(action_type):
    with pytest.raises(ValidationError):
        Action(type=action_type, intent="whatever", target=None)


def test_action_navigate_requires_url():
    with pytest.raises(ValidationError):
        Action(type=ActionType.NAVIGATE, intent="navigate", target=None, url=None)


def test_action_navigate_with_url_is_valid():
    action = Action(type=ActionType.NAVIGATE, intent="navigate", url="http://localhost:8000/members")
    assert action.url == "http://localhost:8000/members"


def test_locator_requires_at_least_one_candidate():
    with pytest.raises(ValidationError):
        Locator(description="empty", candidates=[])


def test_locator_candidate_confidence_bounds():
    with pytest.raises(ValidationError):
        LocatorCandidate(strategy=LocatorStrategy.CSS, value="#x", confidence=1.5)


@pytest.mark.parametrize(
    "checkpoint_type,kwargs",
    [
        (CheckpointType.ELEMENT_VISIBLE, {}),
        (CheckpointType.TEXT_PRESENT, {}),
        (CheckpointType.URL_MATCHES, {}),
    ],
)
def test_checkpoint_requires_matching_field(checkpoint_type, kwargs):
    with pytest.raises(ValidationError):
        Checkpoint(description="bad", type=checkpoint_type, **kwargs)


def test_checkpoint_element_visible_with_locator_is_valid():
    cp = Checkpoint(description="ok", type=CheckpointType.ELEMENT_VISIBLE, locator=make_locator())
    assert cp.locator is not None


def test_enum_input_parameter_requires_enum_values():
    with pytest.raises(ValidationError):
        InputParameter(name="status", type=ParamType.ENUM, description="status filter", enum_values=None)


def test_schema_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        Locator(description="x", candidates=[LocatorCandidate(strategy=LocatorStrategy.CSS, value="#x")], extra_field="nope")
