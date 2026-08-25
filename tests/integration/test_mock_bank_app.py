"""Acceptance checks for the target surface itself, independent of the agent/
replay code that will later drive it. If these fail, nothing downstream can
be trusted - the surface is the ground truth for what "correct" looks like.
"""

from __future__ import annotations

import pytest

import mock_bank_app.app as app_module
from mock_bank_app.data import CONFIRMED_SUBACCOUNTS


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "SLOW_RESPONSE_DELAY_SECONDS", 0.05)
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        yield c
    CONFIRMED_SUBACCOUNTS.clear()


def login(client):
    return client.post("/login", data={"username": "operator", "password": "changeme123"}, follow_redirects=True)


def test_unauthenticated_request_redirects_to_login(client):
    resp = client.get("/members/search", follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_bad_credentials_show_error_banner(client):
    resp = client.post("/login", data={"username": "operator", "password": "wrong"})
    assert resp.status_code == 401
    assert b"error-banner" in resp.data
    assert b"Invalid username" in resp.data


def test_good_credentials_reach_search(client):
    resp = login(client)
    assert resp.status_code == 200
    assert b"Member Search" in resp.data


def test_search_by_exact_id_redirects_to_detail(client):
    login(client)
    resp = client.get("/members/search", query_string={"q": "12345"}, follow_redirects=True)
    assert b"Alicia Torres" in resp.data
    assert b"Savings Balance" in resp.data


def test_search_by_name_returns_results_table(client):
    login(client)
    resp = client.get("/members/search", query_string={"q": "Marcus"})
    assert b"Marcus Webb" in resp.data
    assert b"/members/67890" in resp.data


def test_search_with_no_matches_shows_banner(client):
    login(client)
    resp = client.get("/members/search", query_string={"q": "Nobody Here"})
    assert b"error-banner" in resp.data


@pytest.mark.parametrize(
    "member_id,expected_text",
    [
        ("40400", b"Member not found"),
        ("40300", b"You do not have permission to perform this operation."),
    ],
)
def test_injected_business_outcome_states_are_reproducible(client, member_id, expected_text):
    login(client)
    resp = client.get(f"/members/{member_id}")
    assert expected_text in resp.data


def test_slow_response_state_still_renders_normal_content(client):
    login(client)
    resp = client.get("/members/40800")
    assert b"Dana Whitfield" in resp.data


def test_unexpected_dialog_state_injects_confirm_script(client):
    login(client)
    resp = client.get("/members/40900")
    assert b"window.confirm(" in resp.data
    assert b"Omar Reyes" in resp.data


def test_deposit_below_minimum_reloads_with_field_error(client):
    login(client)
    resp = client.post("/members/12345/subaccounts/new", data={"account_type": "Savings", "opening_deposit": "5"})
    assert b"field-error" in resp.data
    assert b"Minimum opening deposit" in resp.data


def test_non_numeric_deposit_reloads_with_field_error(client):
    login(client)
    resp = client.post(
        "/members/12345/subaccounts/new", data={"account_type": "Savings", "opening_deposit": "not-a-number"}
    )
    assert b"field-error" in resp.data


def test_valid_deposit_reaches_review_screen(client):
    login(client)
    resp = client.post(
        "/members/12345/subaccounts/new",
        data={"account_type": "Savings", "opening_deposit": "100.00"},
        follow_redirects=True,
    )
    assert b"Review New Sub-Account" in resp.data
    assert b"100.00" in resp.data


def test_review_screen_has_two_ambiguous_continue_buttons(client):
    login(client)
    resp = client.post(
        "/members/12345/subaccounts/new",
        data={"account_type": "Savings", "opening_deposit": "100.00"},
        follow_redirects=True,
    )
    html = resp.data.decode()
    assert html.count(">Continue<") == 2
    nav_section = html[html.index("<nav>") : html.index("</nav>")]
    assert ">Continue<" in nav_section
    main_section = html[html.index("<main>") :]
    assert ">Continue<" in main_section


def test_decorative_nav_continue_does_not_advance_the_flow(client):
    login(client)
    resp = client.post(
        "/members/12345/subaccounts/new",
        data={"account_type": "Savings", "opening_deposit": "100.00"},
        follow_redirects=True,
    )
    html = resp.data.decode()
    nav_section = html[html.index("<nav>") : html.index("</nav>")]
    assert "/members/search" in nav_section
    assert "confirmation" not in nav_section


def test_confirm_action_reaches_confirmation_page_and_persists_record(client):
    login(client)
    client.post("/members/12345/subaccounts/new", data={"account_type": "Savings", "opening_deposit": "100.00"})
    resp = client.post("/members/12345/subaccounts/review", data={"action": "confirm"}, follow_redirects=True)
    assert b"Sub-Account Confirmation" in resp.data
    assert b"Confirmation Number" in resp.data
    assert len(CONFIRMED_SUBACCOUNTS) == 1


def test_cancel_action_discards_draft_and_returns_to_detail(client):
    login(client)
    client.post("/members/12345/subaccounts/new", data={"account_type": "Savings", "opening_deposit": "100.00"})
    resp = client.post("/members/12345/subaccounts/review", data={"action": "cancel"}, follow_redirects=True)
    assert b"Member Detail" in resp.data
    assert len(CONFIRMED_SUBACCOUNTS) == 0


def test_dev_reset_clears_session_and_confirmed_subaccounts(client):
    login(client)
    client.post("/members/12345/subaccounts/new", data={"account_type": "Savings", "opening_deposit": "100.00"})
    # The record isn't created until the confirmation page is actually
    # rendered (GET /subaccounts/confirmation) - follow the redirect to reach it.
    client.post("/members/12345/subaccounts/review", data={"action": "confirm"}, follow_redirects=True)
    assert len(CONFIRMED_SUBACCOUNTS) == 1

    resp = client.post("/_dev/reset")
    assert resp.status_code == 200
    assert len(CONFIRMED_SUBACCOUNTS) == 0

    resp = client.get("/members/search", follow_redirects=False)
    assert resp.status_code == 302
