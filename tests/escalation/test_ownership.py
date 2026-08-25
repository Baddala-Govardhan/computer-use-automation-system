import pytest

from escalation.ownership import InvalidOwnershipTransition, Owner, SessionOwnership


def test_starts_owned_by_automation():
    ownership = SessionOwnership()
    assert ownership.owner is Owner.AUTOMATION
    ownership.require_automation()  # does not raise


def test_require_human_raises_while_automation_owns():
    ownership = SessionOwnership()
    with pytest.raises(InvalidOwnershipTransition):
        ownership.require_human()


def test_transfer_to_human_succeeds_from_automation():
    ownership = SessionOwnership()
    transition = ownership.transfer_to_human(reason="needs confirmation")
    assert ownership.owner is Owner.HUMAN
    assert transition.from_owner is Owner.AUTOMATION
    assert transition.to_owner is Owner.HUMAN
    assert ownership.history == [transition]


def test_require_automation_raises_while_human_owns():
    ownership = SessionOwnership()
    ownership.transfer_to_human(reason="x")
    with pytest.raises(InvalidOwnershipTransition):
        ownership.require_automation()


def test_double_transfer_to_human_is_rejected():
    ownership = SessionOwnership()
    ownership.transfer_to_human(reason="x")
    with pytest.raises(InvalidOwnershipTransition):
        ownership.transfer_to_human(reason="y")


def test_transfer_to_automation_requires_human_to_currently_own_it():
    ownership = SessionOwnership()
    with pytest.raises(InvalidOwnershipTransition):
        ownership.transfer_to_automation(reason="resume")


def test_full_round_trip_is_recorded_in_history_order():
    ownership = SessionOwnership()
    t1 = ownership.transfer_to_human(reason="needs confirmation")
    t2 = ownership.transfer_to_automation(reason="human done")
    assert ownership.owner is Owner.AUTOMATION
    assert ownership.history == [t1, t2]
    assert t1.to_owner is Owner.HUMAN
    assert t2.to_owner is Owner.AUTOMATION


def test_double_transfer_to_automation_is_rejected():
    ownership = SessionOwnership()
    ownership.transfer_to_human(reason="x")
    ownership.transfer_to_automation(reason="y")
    with pytest.raises(InvalidOwnershipTransition):
        ownership.transfer_to_automation(reason="z")
