import pytest
from pydantic import ValidationError

from core.outcomes import BusinessOutcome, HardFailure, OutcomeStatus, RunResult


def base_kwargs(**overrides):
    defaults = dict(run_id="replay-x", capability_id="lookup_member_balance", capability_version="1.0.0")
    defaults.update(overrides)
    return defaults


def test_success_with_outputs_is_valid():
    result = RunResult(status=OutcomeStatus.SUCCESS, outputs={"savings_balance": "$1,204.55"}, **base_kwargs())
    assert result.status is OutcomeStatus.SUCCESS


def test_success_cannot_carry_business_outcome():
    with pytest.raises(ValidationError):
        RunResult(
            status=OutcomeStatus.SUCCESS,
            business_outcome=BusinessOutcome(code="member_not_found", message="no such member"),
            **base_kwargs(),
        )


def test_success_cannot_carry_failure():
    with pytest.raises(ValidationError):
        RunResult(
            status=OutcomeStatus.SUCCESS,
            failure=HardFailure(step_id="s1", expected="balance visible", observed="timeout", message="timed out"),
            **base_kwargs(),
        )


def test_business_outcome_status_requires_business_outcome():
    with pytest.raises(ValidationError):
        RunResult(status=OutcomeStatus.BUSINESS_OUTCOME, **base_kwargs())


def test_business_outcome_status_with_outcome_is_valid():
    result = RunResult(
        status=OutcomeStatus.BUSINESS_OUTCOME,
        business_outcome=BusinessOutcome(code="member_not_found", message="no such member"),
        **base_kwargs(),
    )
    assert result.business_outcome.code == "member_not_found"


@pytest.mark.parametrize("status", [OutcomeStatus.HARD_FAILURE, OutcomeStatus.ESCALATED])
def test_failure_statuses_require_failure_detail(status):
    with pytest.raises(ValidationError):
        RunResult(status=status, **base_kwargs())


def test_hard_failure_with_detail_is_valid():
    result = RunResult(
        status=OutcomeStatus.HARD_FAILURE,
        failure=HardFailure(step_id="s2", expected="results table visible", observed="404 page", message="unexpected page"),
        **base_kwargs(),
    )
    assert result.failure.step_id == "s2"
