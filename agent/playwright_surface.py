from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from playwright.async_api import Locator as PlaywrightLocator
from playwright.async_api import Page, async_playwright

from core.actions import ActionType
from core.schema import Action, Locator, LocatorCandidate, LocatorStrategy
from core.surface import SurfaceActionResult, SurfaceObservation


class LocatorResolutionError(Exception):
    pass


class _CoordinateTarget:
    def __init__(self, page: Page, x: float, y: float):
        self._page = page
        self._x = x
        self._y = y

    async def click(self, timeout: float | None = None) -> None:
        await self._page.mouse.click(self._x, self._y)

    async def fill(self, value: str, timeout: float | None = None) -> None:
        await self._page.mouse.click(self._x, self._y)
        await self._page.keyboard.type(value)

    async def press(self, key: str, timeout: float | None = None) -> None:
        await self._page.mouse.click(self._x, self._y)
        await self._page.keyboard.press(key)

    async def inner_text(self, timeout: float | None = None) -> str:
        raise LocatorResolutionError("EXTRACT is not supported against a COORDINATES target")

    async def select_option(self, label: str | None = None, timeout: float | None = None) -> None:
        raise LocatorResolutionError("SELECT is not supported against a COORDINATES target")


class PlaywrightSurface:
    def __init__(self, base_url: str, dialog_policy: Callable[[str, str], bool] | None = None):
        self._base_url = base_url.rstrip("/")
        self._dialog_policy = dialog_policy or self._default_dialog_policy
        self._last_dialog: dict[str, Any] | None = None
        self._last_extracted: str | None = None
        self._playwright = None
        self._browser = None
        self._context = None
        self._page: Page | None = None

    @classmethod
    async def create(
        cls,
        base_url: str,
        headless: bool = True,
        dialog_policy: Callable[[str, str], bool] | None = None,
    ) -> "PlaywrightSurface":
        self = cls(base_url, dialog_policy=dialog_policy)
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=headless)
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()
        self._page.on("dialog", self._on_dialog)
        return self

    @staticmethod
    def _default_dialog_policy(dialog_type: str, message: str) -> bool:
        return dialog_type == "alert"

    async def observe(self) -> SurfaceObservation:
        return await self._build_observation()

    async def act(self, action: Action) -> SurfaceActionResult:
        try:
            await self._perform(action)
        except LocatorResolutionError as e:
            return SurfaceActionResult(success=False, error=str(e), observation=await self._build_observation())
        except Exception as e:
            return SurfaceActionResult(
                success=False, error=f"{type(e).__name__}: {e}", observation=await self._build_observation()
            )
        return SurfaceActionResult(success=True, observation=await self._build_observation())

    async def snapshot(self) -> bytes:
        assert self._page is not None
        return await self._page.screenshot()

    def current_url(self) -> str:
        assert self._page is not None
        return self._page.url

    async def close(self) -> None:
        assert self._context is not None and self._browser is not None and self._playwright is not None
        await self._context.close()
        await self._browser.close()
        await self._playwright.stop()

    async def _on_dialog(self, dialog) -> None:
        accept = self._dialog_policy(dialog.type, dialog.message)
        self._last_dialog = {"type": dialog.type, "message": dialog.message, "accepted": accept}
        if accept:
            await dialog.accept()
        else:
            await dialog.dismiss()

    def take_last_dialog_event(self) -> dict[str, Any] | None:
        event, self._last_dialog = self._last_dialog, None
        return event

    async def take_recoverable_events(self) -> list[dict[str, Any]]:
        event = self.take_last_dialog_event()
        return [event] if event else []

    async def _build_observation(self) -> SurfaceObservation:
        assert self._page is not None
        try:
            tree = await self._page.locator("body").aria_snapshot()
        except Exception:
            tree = ""
        return SurfaceObservation(
            url=self._page.url,
            title=await self._page.title(),
            accessibility_tree=tree,
            extracted_text=self._last_extracted,
            timestamp=datetime.now(timezone.utc),
        )

    async def _perform(self, action: Action) -> None:
        assert self._page is not None
        if action.type is ActionType.NAVIGATE:
            assert action.url is not None
            url = action.url if action.url.startswith("http") else f"{self._base_url}{action.url}"
            await self._page.goto(url, wait_until="load", timeout=action.timeout_ms)
            return
        if action.type is ActionType.WAIT:
            await self._page.wait_for_timeout(action.timeout_ms)
            return

        assert action.target is not None
        target, _candidate = await self._resolve(action.target)

        if action.type is ActionType.CLICK:
            await target.click(timeout=action.timeout_ms)
        elif action.type is ActionType.TYPE:
            await target.fill(action.value or "", timeout=action.timeout_ms)
        elif action.type is ActionType.SELECT:
            await target.select_option(label=action.value, timeout=action.timeout_ms)
        elif action.type is ActionType.PRESS_KEY:
            await target.press(action.value or "", timeout=action.timeout_ms)
        elif action.type is ActionType.EXTRACT:
            self._last_extracted = await target.inner_text(timeout=action.timeout_ms)
        else:
            raise LocatorResolutionError(f"unsupported action type: {action.type}")

    async def _resolve(self, locator: Locator) -> tuple[Any, LocatorCandidate]:
        candidates = sorted(locator.candidates, key=lambda c: -c.confidence)
        errors: list[str] = []

        for candidate in candidates:
            if candidate.strategy is LocatorStrategy.COORDINATES:
                try:
                    x_str, y_str = candidate.value.split(",")
                    return _CoordinateTarget(self._page, float(x_str), float(y_str)), candidate
                except ValueError:
                    errors.append(f"coordinates={candidate.value!r}: malformed, expected 'x,y'")
                    continue

            try:
                pw_locator = self._candidate_to_playwright(candidate)
            except ValueError as e:
                errors.append(f"{candidate.strategy.value}={candidate.value!r}: {e}")
                continue

            count = await pw_locator.count()
            if count == 0:
                errors.append(f"{candidate.strategy.value}={candidate.value!r}: no match")
                continue
            if count > 1:
                errors.append(f"{candidate.strategy.value}={candidate.value!r}: ambiguous ({count} matches)")
                continue
            return pw_locator, candidate

        raise LocatorResolutionError(
            f"no candidate resolved for locator '{locator.description}': " + "; ".join(errors)
        )

    def _candidate_to_playwright(self, candidate: LocatorCandidate) -> PlaywrightLocator:
        assert self._page is not None
        scope = self._page.frame_locator(candidate.frame) if candidate.frame else self._page
        strategy = candidate.strategy
        value = candidate.value

        if strategy is LocatorStrategy.ROLE:
            role, sep, name = value.partition(":")
            if not sep:
                raise ValueError("ROLE candidate value must be 'role:accessible name'")
            return scope.get_by_role(role, name=name, exact=True)
        if strategy is LocatorStrategy.TEST_ID:
            return scope.get_by_test_id(value)
        if strategy is LocatorStrategy.LABEL:
            return scope.get_by_label(value, exact=True)
        if strategy is LocatorStrategy.TEXT:
            return scope.get_by_text(value, exact=True)
        if strategy is LocatorStrategy.CSS:
            return scope.locator(value)
        if strategy is LocatorStrategy.XPATH:
            return scope.locator(f"xpath={value}")

        raise ValueError(f"unhandled locator strategy: {strategy}")
