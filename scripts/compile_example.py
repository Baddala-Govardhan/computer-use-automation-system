from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent.compiler import ParameterSpec, compile_capability
from agent.llm_client import GroqLLMClient
from agent.loop import run_discovery
from agent.playwright_surface import PlaywrightSurface
from agent.recorder import RecordedTrace, record_discovery
from core.actions import ActionType
from core.evidence import EvidenceEventType, EvidenceLogger, RunContext, RunType, new_run_id
from core.policy import PolicyChecker
from core.schema import ParamType
from scripts.discover import _pre_authenticate

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_URL = "http://127.0.0.1:8000"
MODEL_NAME = "openai/gpt-oss-120b"
MEMBER_ID_FOR_GOAL = "12345"
GOAL = f"Find member {MEMBER_ID_FOR_GOAL}, open a new savings sub-account, and stop at the review screen."


def _find_typed_value(trace: RecordedTrace, *, locator_hint: str) -> str:
    hint = locator_hint.lower()
    for step in trace.steps:
        if step.action.type is not ActionType.TYPE or not step.action.value or step.action.target is None:
            continue
        haystack = " ".join([step.action.target.description] + [c.value for c in step.action.target.candidates]).lower()
        if hint in haystack:
            return step.action.value
    raise RuntimeError(f"no recorded TYPE action targeting a field matching '{locator_hint}'")


async def main_async() -> int:
    load_dotenv(REPO_ROOT / ".env")

    try:
        llm = GroqLLMClient(model=MODEL_NAME)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    policy = PolicyChecker.from_files(REPO_ROOT / "config" / "allowlist.yaml", REPO_ROOT / "config" / "risk_policy.yaml")

    run_context = RunContext(
        run_id=new_run_id(RunType.DISCOVERY),
        run_type=RunType.DISCOVERY,
        target=BASE_URL,
        started_at=datetime.now(timezone.utc),
        goal=GOAL,
    )
    evidence = EvidenceLogger(REPO_ROOT / "evidence", run_context)
    evidence.log(EvidenceEventType.RUN_STARTED, f"goal: {GOAL}", data={"goal": GOAL, "base_url": BASE_URL})
    print(f"run_id:   {run_context.run_id}")
    print(f"evidence: {evidence.dir}")

    surface = await PlaywrightSurface.create(base_url=BASE_URL, headless=True)
    try:
        await _pre_authenticate(surface, policy, evidence)
        result = await run_discovery(goal=GOAL, surface=surface, llm=llm, policy=policy, evidence=evidence, max_steps=20)
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

    if not result.succeeded:
        print("discovery did not complete successfully - not compiling an artifact.", file=sys.stderr)
        return 2

    trace = record_discovery(result, goal=GOAL, discovery_run_id=run_context.run_id)

    member_id_value = _find_typed_value(trace, locator_hint="member id")
    opening_deposit_value = _find_typed_value(trace, locator_hint="deposit")
    print(f"discovered parameter values: member_id={member_id_value!r} opening_deposit={opening_deposit_value!r}")

    parameters = [
        ParameterSpec(
            name="member_id",
            type=ParamType.STRING,
            literal_value=member_id_value,
            description="Member ID to search for and open a sub-account under.",
            example=member_id_value,
        ),
        ParameterSpec(
            name="opening_deposit",
            type=ParamType.NUMBER,
            literal_value=opening_deposit_value,
            description="Opening deposit amount for the new savings sub-account.",
            example=opening_deposit_value,
        ),
    ]

    capability = compile_capability(
        trace,
        id="open_savings_subaccount_review",
        name="Open Savings Sub-Account (through review)",
        description=(
            "Searches for a member by ID, opens a new savings sub-account, enters the opening "
            "deposit, and stops at the review screen before final confirmation."
        ),
        app="mock_bank_app",
        parameters=parameters,
        policy=policy,
        model_used=MODEL_NAME,
        version="1.0.0",
    )

    artifact_json = capability.model_dump_json(indent=2)
    evidence.save_artifact(artifact_json)

    examples_dir = REPO_ROOT / "artifacts" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = examples_dir / "open_savings_subaccount_review.json"
    artifact_path.write_text(artifact_json)

    print(f"artifact saved: {artifact_path}")
    print(f"artifact also saved to: {evidence.dir / 'artifact.json'}")
    print()
    print(artifact_json)
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
