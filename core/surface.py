from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from core.schema import Action


class SurfaceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    accessibility_tree: str = Field(
        description=(
            "The a11y tree, in whatever textual/structured form the surface implementation "
            "produces - the primary signal shown to the LLM. For PlaywrightSurface this is "
            "Playwright's ARIA-snapshot format (role + accessible name, nested by landmark, "
            "e.g. 'navigation: - button \"Continue\"'), not a raw JSON tree: Playwright removed "
            "page.accessibility.snapshot() in favor of this, and it's a better fit for the LLM "
            "prompt anyway - it's what actually distinguishes two same-labeled controls in "
            "different landmarks. A desktop surface would produce its own OS-accessibility-API "
            "text here; replay/discovery never assume a particular tree shape, only that this "
            "is a string worth showing to a human reviewer or an LLM."
        )
    )
    extracted_text: str | None = Field(default=None, description="Optional flattened text, e.g. for EXTRACT actions.")
    timestamp: datetime


class SurfaceActionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    error: str | None = None
    observation: SurfaceObservation | None = Field(
        default=None, description="Post-action observation, when the caller requests one."
    )


@runtime_checkable
class Surface(Protocol):
    async def observe(self) -> SurfaceObservation: ...

    async def act(self, action: Action) -> SurfaceActionResult: ...

    async def snapshot(self) -> bytes: ...

    def current_url(self) -> str: ...

    async def close(self) -> None: ...

    async def take_recoverable_events(self) -> list[dict[str, Any]]: ...
