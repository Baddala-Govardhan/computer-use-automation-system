from __future__ import annotations

import argparse
import asyncio
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent.llm_client import GroqLLMClient
from agent.loop import StopReason, run_discovery
from agent.playwright_surface import PlaywrightSurface
from core.evidence import EvidenceLogger, RunContext, RunType, new_run_id
from core.outcomes import HardFailure, OutcomeStatus, RunResult
from core.policy import PolicyChecker
from escalation.manager import EscalationManager
from escalation.ownership import Owner
from scripts.discover import _pre_authenticate
from scripts.handoff_demo import build_confirm_capability

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GOAL = (
    "Find member 12345, open a new savings sub-account with an opening deposit of $100, "
    "and reach the confirmation screen."
)
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "openai/gpt-oss-120b"

_MEMBER_ID_RE = re.compile(r"/members/([^/]+)/")
_DEPOSIT_RE = re.compile(r"Opening Deposit[^$]*(\$[\d,]+\.\d{2})")


def _extract_member_id(url: str) -> str:
    match = _MEMBER_ID_RE.search(url)
    return match.group(1) if match else "unknown"


def _extract_opening_deposit(accessibility_tree: str) -> str | None:
    match = _DEPOSIT_RE.search(accessibility_tree)
    return match.group(1) if match else None


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
    print(f"run_id:   {run_context.run_id}")
    print(f"evidence: {evidence.dir}")
    print(f"goal:     {args.goal}")

    surface = await PlaywrightSurface.create(base_url=args.base_url, headless=not args.headed)
    exit_code = 2
    try:
        await _pre_authenticate(surface, policy, evidence)

        print("\n--- phase 1: LLM discovery (Groq) ---")
        discovery_result = await run_discovery(
            goal=args.goal, surface=surface, llm=llm, policy=policy, evidence=evidence, max_steps=args.max_steps
        )
        print(f"stop_reason: {discovery_result.stop_reason.value}")
        print(f"steps taken: {len(discovery_result.steps)}")
        print(f"reasoning:   {discovery_result.final_reasoning}")
        print(f"current url: {surface.current_url()}")

        if discovery_result.stop_reason is not StopReason.CONFIRMATION_REQUIRED:
            print(
                f"\ndiscovery stopped for a reason other than CONFIRMATION_REQUIRED "
                f"({discovery_result.stop_reason.value}) - nothing to hand off."
            )
            exit_code = 0 if discovery_result.succeeded else 2
        else:
            confirm_capability = build_confirm_capability(args.base_url)
            manager = EscalationManager(surface, evidence)
            request = await manager.raise_intervention(
                capability=confirm_capability,
                step_id=str(len(discovery_result.steps)),
                reason=discovery_result.final_reasoning,
                risk_level="irreversible",
            )

            observation = await surface.observe()
            member_id = _extract_member_id(observation.url)
            opening_deposit = _extract_opening_deposit(observation.accessibility_tree)

            print("\n--------------------------------------------------")
            print("HUMAN APPROVAL REQUIRED")
            print("--------------------------------------------------\n")
            print(f"Action: {confirm_capability.name}")
            print(f"Member ID: {member_id}")
            if opening_deposit:
                print(f"Opening Deposit: {opening_deposit}")
            print("Risk: Irreversible")
            print("\nAutomation has been paused.")
            print("The same browser session is waiting on the Review page.")
            print("\nPress Enter to approve and continue.")
            print('Type "cancel" to stop.')
            print("--------------------------------------------------")
            sys.stdout.flush()

            try:
                response = input().strip().lower()
            except EOFError:
                print(
                    "\n(no interactive stdin available - an approval prompt gating an "
                    "irreversible action can't default to 'approved'. Run this in a real "
                    "terminal to approve. Treating as cancelled.)"
                )
                response = "cancel"

            if response == "cancel":
                await manager.resume(note="human declined the irreversible confirmation")
                failure = HardFailure(
                    step_id=request.step_id,
                    expected="human approval to perform the irreversible confirmation",
                    observed="human typed 'cancel'",
                    message="human declined the irreversible confirmation - action not performed",
                )
                final_result = RunResult(
                    run_id=evidence.run_id,
                    status=OutcomeStatus.ESCALATED,
                    capability_id=confirm_capability.id,
                    capability_version=confirm_capability.version,
                    failure=failure,
                )
                print("\nCancelled. The irreversible action was not performed.")
                print(f"Status: {final_result.status.value.upper()}")
                exit_code = 2
            else:
                approve_result = await manager.perform_human_action(confirm_capability.steps[0].action)
                if not approve_result.success:
                    await manager.resume(note="approved action failed to execute")
                    failure = HardFailure(
                        step_id=request.step_id,
                        expected="the approved confirmation action to succeed",
                        observed=approve_result.error or "unknown error",
                        message="approved confirmation action failed - automation cannot proceed",
                    )
                    final_result = RunResult(
                        run_id=evidence.run_id,
                        status=OutcomeStatus.HARD_FAILURE,
                        capability_id=confirm_capability.id,
                        capability_version=confirm_capability.version,
                        failure=failure,
                    )
                    print(f"\nApproved action failed: {approve_result.error}")
                    print(f"Status: {final_result.status.value.upper()}")
                    exit_code = 2
                else:
                    await manager.resume(note="human reviewed and approved the new sub-account")
                    assert manager.ownership.owner is Owner.AUTOMATION

                    final_result = await manager.verify_and_complete(confirm_capability)
                    exit_code = 0 if final_result.status is OutcomeStatus.SUCCESS else 2

                    if final_result.status is OutcomeStatus.SUCCESS:
                        print("\nHuman approval received.")
                        print("Confirmation submitted.")
                        print("\nAutomation resumed.")
                        print("Confirmation page verified.")
                        print(f"\nStatus: {final_result.status.value.upper()}")
                        print(f"Confirmation Number: {final_result.outputs.get('confirmation_number')}")
                    else:
                        print(f"\nStatus: {final_result.status.value.upper()}")
                        if final_result.failure is not None:
                            print(f"Reason: {final_result.failure.message}")

            if args.headed and final_result.status is OutcomeStatus.SUCCESS:
                print("\nBrowser left open for inspection.")
                sys.stdout.flush()
                try:
                    input("Press Enter to close...")
                except EOFError:
                    print(
                        "\n(no interactive stdin available, so nothing was actually waited on - "
                        "run this command in a real terminal, not an IDE launcher, to pause here.)"
                    )
    finally:
        evidence.save_screenshot(await surface.snapshot(), name="final")
        await surface.close()

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM discovery through Review, then human handoff to Confirmation.")
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--headed", action="store_true", help="Show the browser instead of running headless.")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
