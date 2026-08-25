"""Does core.surface.Surface actually survive contact with a real, deliberately
hostile UI? Runs the real mock_bank_app under a live Flask server and drives
it with the real PlaywrightSurface + real Chromium - no mocks below core.schema.

Async throughout, matching PlaywrightSurface's async API (required so it can
run inside the same event loop as agent/loop.py's async LLMClient calls -
see agent/playwright_surface.py's module docstring for why the sync API
doesn't work here).
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import closing

import pytest
import requests

from agent.playwright_surface import PlaywrightSurface
from core.actions import ActionType
from core.schema import Action, Locator, LocatorCandidate, LocatorStrategy


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    import mock_bank_app.app as app_module

    port = _free_port()
    thread = threading.Thread(
        target=app_module.app.run,
        kwargs={"host": "127.0.0.1", "port": port, "use_reloader": False, "threaded": True},
        daemon=True,
    )
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            requests.get(f"{base_url}/login", timeout=0.5)
            break
        except requests.exceptions.ConnectionError:
            time.sleep(0.1)
    else:
        raise RuntimeError("mock_bank_app did not start in time")

    yield base_url


@pytest.fixture
async def surface(live_server):
    s = await PlaywrightSurface.create(base_url=live_server, headless=True)
    yield s
    await s.close()


def role(value: str, confidence: float = 1.0) -> Locator:
    return Locator(
        description=value,
        candidates=[LocatorCandidate(strategy=LocatorStrategy.ROLE, value=value, confidence=confidence)],
    )


def label(value: str) -> Locator:
    return Locator(description=value, candidates=[LocatorCandidate(strategy=LocatorStrategy.LABEL, value=value)])


async def _login(surface: PlaywrightSurface) -> None:
    await surface.act(Action(type=ActionType.NAVIGATE, intent="navigate", url="/login"))
    await surface.act(Action(type=ActionType.TYPE, intent="enter_username", value="operator", target=label("Username")))
    await surface.act(
        Action(type=ActionType.TYPE, intent="enter_password", value="changeme123", target=label("Password"))
    )
    r = await surface.act(Action(type=ActionType.CLICK, intent="submit_login", target=role("button:Log In")))
    assert r.success, r.error


async def test_navigate_type_click_login_flow(surface):
    await _login(surface)
    obs = await surface.observe()
    assert "/members/search" in obs.url
    assert "Member Search" in obs.accessibility_tree


async def test_search_by_id_and_extract_balance_via_xpath(surface):
    await _login(surface)

    r = await surface.act(
        Action(type=ActionType.TYPE, intent="search_member", value="12345", target=label("Member ID or Name"))
    )
    assert r.success, r.error
    r = await surface.act(Action(type=ActionType.CLICK, intent="submit_search", target=role("button:Search")))
    assert r.success, r.error
    assert "/members/12345" in r.observation.url

    balance_locator = Locator(
        description="Savings balance cell",
        candidates=[
            LocatorCandidate(
                strategy=LocatorStrategy.XPATH,
                value="//td[text()='Savings Balance']/following-sibling::td[1]",
                notes="Sibling-cell XPath: the only stable option in a legacy table layout with no test ids.",
            )
        ],
    )
    r = await surface.act(Action(type=ActionType.EXTRACT, intent="read_balance", target=balance_locator))
    assert r.success, r.error
    assert r.observation.extracted_text == "$1204.55"


async def test_business_outcome_states_are_visible_in_the_observation(surface):
    await _login(surface)

    r = await surface.act(Action(type=ActionType.NAVIGATE, intent="navigate", url="/members/40400"))
    assert r.success
    assert "Member not found" in r.observation.accessibility_tree

    r = await surface.act(Action(type=ActionType.NAVIGATE, intent="navigate", url="/members/40300"))
    assert r.success
    assert "permission" in r.observation.accessibility_tree.lower()


async def test_ambiguous_role_locator_fails_closed_instead_of_guessing(surface):
    await _login(surface)
    await surface.act(Action(type=ActionType.NAVIGATE, intent="navigate", url="/members/12345/subaccounts/new"))
    await surface.act(
        Action(type=ActionType.TYPE, intent="enter_deposit", value="100.00", target=label("Opening Deposit"))
    )
    r = await surface.act(
        Action(type=ActionType.CLICK, intent="submit_new_sub_account_form", target=role("button:Continue"))
    )
    assert r.success, r.error
    assert "/subaccounts/review" in r.observation.url

    ambiguous = role("button:Continue")
    r = await surface.act(Action(type=ActionType.CLICK, intent="submit_new_sub_account", target=ambiguous))
    assert not r.success
    assert "ambiguous" in r.error


async def test_landmark_scoped_fallback_resolves_the_real_continue_not_the_nav_trap(surface):
    await _login(surface)
    await surface.act(Action(type=ActionType.NAVIGATE, intent="navigate", url="/members/12345/subaccounts/new"))
    await surface.act(
        Action(type=ActionType.TYPE, intent="enter_deposit", value="100.00", target=label("Opening Deposit"))
    )
    await surface.act(
        Action(type=ActionType.CLICK, intent="submit_new_sub_account_form", target=role("button:Continue"))
    )

    scoped = Locator(
        description="Continue (review) - scoped to main, disambiguated from the decorative nav Continue",
        candidates=[
            LocatorCandidate(strategy=LocatorStrategy.ROLE, value="button:Continue", confidence=1.0),
            LocatorCandidate(
                strategy=LocatorStrategy.CSS,
                value="main form button[value='confirm']",
                confidence=0.5,
                notes="Falls back to landmark + form-value scoping since role+name alone is ambiguous on this page.",
            ),
        ],
    )
    r = await surface.act(Action(type=ActionType.CLICK, intent="submit_new_sub_account", target=scoped))
    assert r.success, r.error
    assert (
        "/subaccounts/confirmation" in r.observation.url
    ), "landed on the decorative nav trap instead of the real confirmation step"


async def test_unexpected_dialog_is_dismissed_by_default_and_recorded(surface):
    await _login(surface)
    assert surface.take_last_dialog_event() is None

    r = await surface.act(Action(type=ActionType.NAVIGATE, intent="navigate", url="/members/40900"))
    assert r.success, r.error  # the session isn't blocked by the native dialog

    event = surface.take_last_dialog_event()
    assert event is not None
    assert event["type"] == "confirm"
    assert "expire" in event["message"]
    assert event["accepted"] is False, "confirm/prompt dialogs must not be auto-accepted by default"

    assert surface.take_last_dialog_event() is None


async def test_dialog_policy_is_configurable_via_constructor(live_server):
    accept_everything = lambda dialog_type, message: True
    s = await PlaywrightSurface.create(base_url=live_server, headless=True, dialog_policy=accept_everything)
    try:
        await _login(s)
        r = await s.act(Action(type=ActionType.NAVIGATE, intent="navigate", url="/members/40900"))
        assert r.success, r.error
        event = s.take_last_dialog_event()
        assert event is not None
        assert event["accepted"] is True
    finally:
        await s.close()
