from __future__ import annotations

import base64

from pydantic import BaseModel, ConfigDict, Field

from core.schema import Action
from core.surface import SurfaceObservation

THIN_SNAPSHOT_CHAR_THRESHOLD = 40


class PreviousAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    action: Action
    success: bool
    error: str | None = None


class DiscoveryContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    url: str
    accessibility_snapshot: str
    previous_actions: list[PreviousAction] = Field(default_factory=list)
    screenshot_png_base64: str | None = Field(
        default=None, description="Populated only when should_attach_screenshot() says so."
    )
    note: str | None = Field(default=None, description="Extra hint surfaced to the model, e.g. why a screenshot was attached.")


def build_context(
    *,
    goal: str,
    observation: SurfaceObservation,
    previous_actions: list[PreviousAction],
    screenshot_png: bytes | None = None,
    note: str | None = None,
) -> DiscoveryContext:
    return DiscoveryContext(
        goal=goal,
        url=observation.url,
        accessibility_snapshot=observation.accessibility_tree,
        previous_actions=previous_actions,
        screenshot_png_base64=base64.b64encode(screenshot_png).decode() if screenshot_png else None,
        note=note,
    )


def should_attach_screenshot(
    *,
    requested_by_model: bool = False,
    last_action_failed: bool = False,
    unexpected_dialog: bool = False,
    accessibility_snapshot: str = "",
) -> bool:
    return (
        requested_by_model
        or last_action_failed
        or unexpected_dialog
        or len(accessibility_snapshot.strip()) < THIN_SNAPSHOT_CHAR_THRESHOLD
    )
