from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent.llm_client import GroqLLMClient
from agent.loop import run_discovery
from agent.playwright_surface import PlaywrightSurface
from core.actions import ActionType
from core.evidence import EvidenceEventType, EvidenceLogger, RunContext, RunType, new_run_id
from core.policy import PolicyChecker
from core.schema import Action, Locator, LocatorCandidate, LocatorStrategy

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GOAL = (
    "Find member 12345, open a new savings sub-account, and stop at the review "
    "screen before final confirmation."
)
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
OPERATOR_USERNAME = "operator"
OPERATOR_PASSWORD = "changeme123"


def _label(value: str) -> Locator:
    return Locator(description=value, candidates=[LocatorCandidate(strategy=LocatorStrategy.LABEL, value=value)])


def _role(value: str) -> Locator:
    return Locator(description=value, candidates=[LocatorCandidate(strategy=LocatorStrategy.ROLE, value=value)])


async def _pre_authenticate(surface: PlaywrightSurface, policy: PolicyChecker, evidence: EvidenceLogger) -> None:
    steps = [
        Action(type=ActionType.NAVIGATE, intent="navigate", url="/login"),
        Action(type=ActionType.TYPE, intent="enter_username", value=OPERATOR_USERNAME, target=_label("Username")),
        Action(type=ActionType.TYPE, intent="enter_password", value=OPERATOR_PASSWORD, target=_label("Password")),
        Action(type=ActionType.CLICK, intent="submit_login", target=_role("button:Log In")),
    ]
    for i, action in enumerate(steps):
        policy_decision = policy.check_action(action, current_url=surface.current_url())
        if not policy_decision.allowed:
            raise RuntimeError(f"pre-authentication blocked by policy: {policy_decision.reason}")

        result = await surface.act(action)
        evidence.log(
            EvidenceEventType.ACTION,
            f"pre-auth: {action.type.value} ({action.intent})",
            step_id=f"pre-auth-{i}",
            data={"success": result.success, "error": result.error},
        )
        if not result.success:
            raise RuntimeError(f"pre-authentication failed at step '{action.intent}': {result.error}")


async def main_async(args: argparse.Namespace) -> int:
    load_dotenv(REPO_ROOT / ".env")

    try:
        llm = GroqLLMClient(model=args.model)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    policy = PolicyChecker.from_files(REPO_ROOT / "config" / "allowlist.yaml", REPO_ROOT / "config" / "risk_policy.yaml")

    run_context = RunContext(
        run_id=new_run_id(RunType.DISCOVERY),
        run_type=RunType.DISCOVERY,
        target=args.base_url,
        started_at=datetime.now(timezone.utc),
        goal=args.goal,
    )
    evidence = EvidenceLogger(REPO_ROOT / "evidence", run_context)
    evidence.log(EvidenceEventType.RUN_STARTED, f"goal: {args.goal}", data={"goal": args.goal, "base_url": args.base_url})

    print(f"run_id:   {run_context.run_id}")
    print(f"evidence: {evidence.dir}")
    print(f"goal:     {args.goal}")

    surface = await PlaywrightSurface.create(base_url=args.base_url, headless=not args.headed)
    try:
        await _pre_authenticate(surface, policy, evidence)
        result = await run_discovery(
            goal=args.goal,
            surface=surface,
            llm=llm,
            policy=policy,
            evidence=evidence,
            max_steps=args.max_steps,
        )
    finally:
        evidence.save_screenshot(await surface.snapshot(), name="final")
        await surface.close()

    evidence.log(
        EvidenceEventType.RUN_COMPLETED,
        f"stop_reason={result.stop_reason.value}",
        data={
            "stop_reason": result.stop_reason.value,
            "final_reasoning": result.final_reasoning,
            "steps_taken": len(result.steps),
        },
    )

    print(f"stop_reason: {result.stop_reason.value}")
    print(f"steps taken: {len(result.steps)}")
    print(f"reasoning:   {result.final_reasoning}")
    return 0 if result.succeeded else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the discovery agent against a live target and save evidence.")
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--headed", action="store_true", help="Show the browser instead of running headless.")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
