from __future__ import annotations

import json

import pytest

from agent.compiler import CompilationError, OutputSpec, ParameterSpec, compile_capability
from agent.recorder import RecordedStep, RecordedTrace
from core.actions import ActionType
from core.policy import AllowedTarget, AllowlistConfig, IntentRisk, PolicyChecker, RiskLevel, RiskPolicyConfig
from core.schema import (
    Action,
    Capability,
    CheckpointType,
    Locator,
    LocatorCandidate,
    LocatorStrategy,
    ParamType,
)

BASE_URL = "http://127.0.0.1:8010"


def label(value: str) -> Locator:
    return Locator(description=value, candidates=[LocatorCandidate(strategy=LocatorStrategy.LABEL, value=value)])


def role(value: str, confidence: float = 1.0, extra_css: str | None = None) -> Locator:
    candidates = [LocatorCandidate(strategy=LocatorStrategy.ROLE, value=value, confidence=confidence)]
    if extra_css:
        candidates.append(LocatorCandidate(strategy=LocatorStrategy.CSS, value=extra_css, confidence=0.5))
    return Locator(description=value, candidates=candidates)


def build_trace(member_id: str, deposit: str, *, discovery_run_id: str = "discovery-x") -> RecordedTrace:
    steps = [
        RecordedStep(
            source_index=0,
            action=Action(type=ActionType.TYPE, intent="search_member", value=member_id, target=label("Member ID or Name")),
            reasoning=f"Enter member ID {member_id} into the search field.",
            observed_url=f"{BASE_URL}/members/search",
        ),
        RecordedStep(
            source_index=1,
            action=Action(type=ActionType.CLICK, intent="search_member", target=role("button:Search")),
            reasoning="Submit the search.",
            observed_url=f"{BASE_URL}/members/{member_id}",
        ),
        RecordedStep(
            source_index=2,
            action=Action(type=ActionType.CLICK, intent="open_new_sub_account", target=role("link:Open New Sub-Account")),
            reasoning="Open the new sub-account flow.",
            observed_url=f"{BASE_URL}/members/{member_id}/subaccounts/new",
        ),
        RecordedStep(
            source_index=3,
            action=Action(
                type=ActionType.TYPE, intent="submit_new_sub_account", value=deposit, target=label("Opening Deposit")
            ),
            reasoning=f"Enter opening deposit {deposit}.",
            observed_url=f"{BASE_URL}/members/{member_id}/subaccounts/new",
        ),
        RecordedStep(
            source_index=4,
            action=Action(
                type=ActionType.CLICK,
                intent="submit_new_sub_account",
                target=role("button:Continue", extra_css="main form button[value='confirm']"),
            ),
            reasoning="Submit the form to reach the review screen.",
            observed_url=f"{BASE_URL}/members/{member_id}/subaccounts/review",
        ),
    ]
    return RecordedTrace(
        goal=f"Find member {member_id}, open a new savings sub-account, and stop at the review screen.",
        steps=steps,
        discovery_run_id=discovery_run_id,
    )


def build_parameters(member_id: str, deposit: str) -> list[ParameterSpec]:
    return [
        ParameterSpec(name="member_id", type=ParamType.STRING, literal_value=member_id, description="Member ID to search for"),
        ParameterSpec(
            name="opening_deposit", type=ParamType.NUMBER, literal_value=deposit, description="Initial deposit amount"
        ),
    ]


def make_policy(**intent_overrides: IntentRisk) -> PolicyChecker:
    allowlist = AllowlistConfig(
        allowed_targets=[AllowedTarget(name="mock_bank_app", base_url=BASE_URL, allowed_routes=["*"])],
        allowed_action_types=list(ActionType),
    )
    intents = {
        "search_member": IntentRisk(risk=RiskLevel.SAFE),
        "open_new_sub_account": IntentRisk(risk=RiskLevel.SAFE),
        "submit_new_sub_account": IntentRisk(risk=RiskLevel.SAFE),
        "confirm_new_sub_account": IntentRisk(risk=RiskLevel.IRREVERSIBLE, requires_human_confirmation=True),
    }
    intents.update(intent_overrides)
    risk_policy = RiskPolicyConfig(default=IntentRisk(risk=RiskLevel.SAFE), intents=intents)
    return PolicyChecker(allowlist, risk_policy)


def compile_bank_example(member_id: str = "12345", deposit: str = "1000") -> Capability:
    trace = build_trace(member_id, deposit)
    return compile_capability(
        trace,
        id="open_savings_subaccount_review",
        name="Open Savings Sub-Account (through review)",
        description="Searches for a member, opens a new savings sub-account, and stops at the review screen.",
        app="mock_bank_app",
        parameters=build_parameters(member_id, deposit),
        policy=make_policy(),
        model_used="openai/gpt-oss-120b",
    )


def test_generic_successful_trace_compiles():
    capability = compile_bank_example()
    assert capability.id == "open_savings_subaccount_review"
    assert len(capability.steps) == 5


def test_literal_member_id_becomes_a_parameter_placeholder():
    capability = compile_bank_example(member_id="12345", deposit="1000")

    search_step = capability.steps[0]
    assert search_step.action.value == "{member_id}"
    assert "12345" not in search_step.action.value


def test_literal_deposit_becomes_a_parameter_placeholder():
    capability = compile_bank_example(member_id="12345", deposit="1000")

    deposit_step = capability.steps[3]
    assert deposit_step.action.value == "{opening_deposit}"
    assert "1000" not in deposit_step.action.value


def test_no_literal_invocation_values_survive_anywhere_in_the_artifact():
    capability = compile_bank_example(member_id="12345", deposit="1000")
    artifact_json = capability.model_dump_json()

    assert "12345" not in artifact_json
    assert "1000" not in artifact_json
    assert "{member_id}" in artifact_json
    assert "{opening_deposit}" in artifact_json


def test_literal_embedded_in_a_locators_own_description_is_also_parameterized():
    trace = build_trace("12345", "1000")
    trace.steps[1] = RecordedStep(
        source_index=trace.steps[1].source_index,
        action=Action(
            type=ActionType.CLICK,
            intent="search_member",
            target=Locator(
                description="Click the Search button to find member 12345",
                candidates=[LocatorCandidate(strategy=LocatorStrategy.ROLE, value="button:Search")],
            ),
        ),
        reasoning=trace.steps[1].reasoning,
        observed_url=trace.steps[1].observed_url,
    )

    capability = compile_capability(
        trace,
        id="x",
        name="x",
        description="x",
        app="mock_bank_app",
        parameters=build_parameters("12345", "1000"),
        policy=make_policy(),
    )

    assert "12345" not in capability.model_dump_json()
    assert capability.steps[1].action.target.description == "Click the Search button to find member {member_id}"


def test_same_compiler_code_reproduces_a_different_member_and_deposit():
    cap_a = compile_bank_example(member_id="12345", deposit="1000")
    cap_b = compile_bank_example(member_id="67890", deposit="2500")

    assert len(cap_a.steps) == len(cap_b.steps)
    for step_a, step_b in zip(cap_a.steps, cap_b.steps):
        assert step_a.action.intent == step_b.action.intent
        assert step_a.action.type == step_b.action.type
        assert step_a.action.value == step_b.action.value

    assert "67890" not in cap_b.model_dump_json()
    assert "2500" not in cap_b.model_dump_json()
    assert "12345" not in cap_a.model_dump_json()
    assert "1000" not in cap_a.model_dump_json()


def test_input_parameters_are_typed_correctly():
    capability = compile_bank_example()
    by_name = {p.name: p for p in capability.inputs}

    assert by_name["member_id"].type == ParamType.STRING
    assert by_name["opening_deposit"].type == ParamType.NUMBER


def test_success_checkpoint_is_a_parameterized_url_pattern_not_a_literal_member_id():
    capability = compile_bank_example(member_id="12345", deposit="1000")

    assert capability.success_checkpoint.type == CheckpointType.URL_MATCHES
    assert "12345" not in capability.success_checkpoint.url_pattern
    assert capability.success_checkpoint.url_pattern.endswith("/subaccounts/review")
    assert "*" in capability.success_checkpoint.url_pattern


def test_success_checkpoint_pattern_is_identical_across_different_recordings():
    cap_a = compile_bank_example(member_id="12345", deposit="1000")
    cap_b = compile_bank_example(member_id="99999", deposit="50")

    assert cap_a.success_checkpoint.url_pattern == cap_b.success_checkpoint.url_pattern


def test_locator_fallback_strategy_is_preserved_through_compilation():
    capability = compile_bank_example()
    continue_step = capability.steps[4]

    candidates = continue_step.action.target.candidates
    assert [c.strategy for c in candidates] == [LocatorStrategy.ROLE, LocatorStrategy.CSS]
    assert candidates[0].value == "button:Continue"
    assert candidates[1].value == "main form button[value='confirm']"


def test_capability_validates_against_the_strict_core_schema():
    capability = compile_bank_example()
    restored = Capability.model_validate_json(capability.model_dump_json())
    assert restored == capability


def test_irreversible_confirm_action_is_never_part_of_the_bank_example():
    capability = compile_bank_example()
    intents = [s.action.intent for s in capability.steps]
    assert "confirm_new_sub_account" not in intents
    assert capability.success_checkpoint.url_pattern.endswith("/subaccounts/review")


def test_compilation_rejects_a_step_requiring_human_confirmation():
    trace = build_trace("12345", "1000")
    trace.steps.append(
        RecordedStep(
            source_index=5,
            action=Action(type=ActionType.CLICK, intent="confirm_new_sub_account", target=role("button:Confirm")),
            reasoning="Confirm the new sub-account.",
            observed_url=f"{BASE_URL}/members/12345/subaccounts/confirmation",
        )
    )

    with pytest.raises(CompilationError, match="confirm_new_sub_account"):
        compile_capability(
            trace,
            id="x",
            name="x",
            description="x",
            app="mock_bank_app",
            parameters=build_parameters("12345", "1000"),
            policy=make_policy(),
        )


def test_compilation_rejects_a_blocked_intent():
    trace = build_trace("12345", "1000")
    trace.steps.append(
        RecordedStep(
            source_index=5,
            action=Action(type=ActionType.CLICK, intent="delete_account", target=role("button:Delete")),
            reasoning="Delete the account.",
            observed_url=f"{BASE_URL}/members/12345",
        )
    )
    policy = make_policy(delete_account=IntentRisk(risk=RiskLevel.IRREVERSIBLE, requires_human_confirmation=True, blocked=True))

    with pytest.raises(CompilationError, match="delete_account"):
        compile_capability(
            trace,
            id="x",
            name="x",
            description="x",
            app="mock_bank_app",
            parameters=build_parameters("12345", "1000"),
            policy=policy,
        )


def test_compilation_requires_a_discovery_run_id_for_provenance():
    trace = build_trace("12345", "1000", discovery_run_id=None)  # type: ignore[arg-type]

    with pytest.raises(CompilationError, match="discovery_run_id"):
        compile_capability(
            trace,
            id="x",
            name="x",
            description="x",
            app="mock_bank_app",
            parameters=build_parameters("12345", "1000"),
            policy=make_policy(),
        )


def test_compilation_of_empty_trace_is_rejected():
    trace = RecordedTrace(goal="g", steps=[], discovery_run_id="d1")

    with pytest.raises(CompilationError, match="no steps"):
        compile_capability(
            trace, id="x", name="x", description="x", app="mock_bank_app", parameters=[], policy=make_policy()
        )


def test_sensitive_parameter_value_never_appears_in_the_serialized_artifact():
    trace = build_trace("12345", "1000")
    parameters = build_parameters("12345", "1000")
    parameters.append(
        ParameterSpec(
            name="pin", type=ParamType.STRING, literal_value="4321", description="unused in this trace", sensitive=True
        )
    )

    capability = compile_capability(
        trace,
        id="x",
        name="x",
        description="x",
        app="mock_bank_app",
        parameters=parameters,
        policy=make_policy(),
    )

    pin_param = next(p for p in capability.inputs if p.name == "pin")
    assert pin_param.sensitive is True
    assert pin_param.example is None
    assert "4321" not in capability.model_dump_json()


def test_output_definition_is_produced_for_a_declared_extract_step():
    trace = build_trace("12345", "1000")
    trace.steps.append(
        RecordedStep(
            source_index=5,
            action=Action(
                type=ActionType.EXTRACT,
                intent="read_balance",
                target=Locator(
                    description="Savings balance cell",
                    candidates=[LocatorCandidate(strategy=LocatorStrategy.XPATH, value="//td[2]")],
                ),
            ),
            reasoning="Read the savings balance.",
            observed_url=f"{BASE_URL}/members/12345/subaccounts/review",
        )
    )

    capability = compile_capability(
        trace,
        id="x",
        name="x",
        description="x",
        app="mock_bank_app",
        parameters=build_parameters("12345", "1000"),
        policy=make_policy(read_balance=IntentRisk(risk=RiskLevel.SAFE)),
        outputs=[OutputSpec(name="savings_balance", type=ParamType.STRING, description="Current savings balance", source_index=5)],
    )

    assert len(capability.outputs) == 1
    assert capability.outputs[0].name == "savings_balance"
    assert capability.outputs[0].source.candidates[0].value == "//td[2]"


def test_no_groq_or_playwright_specific_data_leaks_into_the_artifact():
    capability = compile_bank_example()
    artifact_json = capability.model_dump_json().lower()
    assert "api_key" not in artifact_json
    assert "usage" not in artifact_json
    assert "prompt_tokens" not in artifact_json
    assert "tool_call" not in artifact_json
