import base64
from datetime import datetime, timezone

from agent.perception import build_context, should_attach_screenshot
from core.surface import SurfaceObservation


def make_observation(tree: str = '- button "Search"') -> SurfaceObservation:
    return SurfaceObservation(url="http://x/", title="t", accessibility_tree=tree, timestamp=datetime.now(timezone.utc))


def test_build_context_without_screenshot_leaves_it_unset():
    ctx = build_context(goal="g", observation=make_observation(), previous_actions=[])
    assert ctx.screenshot_png_base64 is None
    assert ctx.url == "http://x/"
    assert ctx.accessibility_snapshot.startswith("- button")


def test_build_context_encodes_screenshot_as_base64():
    ctx = build_context(goal="g", observation=make_observation(), previous_actions=[], screenshot_png=b"\x89PNGdata")
    assert ctx.screenshot_png_base64 is not None
    assert base64.b64decode(ctx.screenshot_png_base64) == b"\x89PNGdata"


def test_should_attach_screenshot_when_requested_by_model():
    assert should_attach_screenshot(requested_by_model=True, accessibility_snapshot="a" * 100)


def test_should_attach_screenshot_when_last_action_failed():
    assert should_attach_screenshot(last_action_failed=True, accessibility_snapshot="a" * 100)


def test_should_attach_screenshot_when_dialog_fired():
    assert should_attach_screenshot(unexpected_dialog=True, accessibility_snapshot="a" * 100)


def test_should_attach_screenshot_when_snapshot_is_thin():
    assert should_attach_screenshot(accessibility_snapshot="")


def test_should_not_attach_screenshot_by_default_with_a_healthy_snapshot():
    assert not should_attach_screenshot(accessibility_snapshot="a" * 100)
