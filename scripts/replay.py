from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent.playwright_surface import PlaywrightSurface
from core.evidence import EvidenceLogger, RunContext, RunType, new_run_id
from core.outcomes import OutcomeStatus
from core.policy import PolicyChecker
from core.schema import Capability
from replay.engine import replay_capability
from scripts.discover import _pre_authenticate

REPO_ROOT = Path(__file__).resolve().parents[1]


def _override_timeouts(capability: Capability, timeout_ms: int) -> Capability:
    new_steps = [
        step.model_copy(update={"action": step.action.model_copy(update={"timeout_ms": timeout_ms})})
        for step in capability.steps
    ]
    return capability.model_copy(update={"steps": new_steps})


def _make_pre_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("artifact_path", help="Path to a saved Capability artifact JSON file.")
    parser.add_argument("--base-url", default=None, help="Override the artifact's declared target.base_url.")
    parser.add_argument(
        "--action-timeout-ms", type=int, default=None, help="Override every step's action timeout_ms."
    )
    parser.add_argument("--headed", action="store_true", help="Show the browser instead of running headless.")
    return parser


async def main_async(capability: Capability, inputs: dict[str, str], args: argparse.Namespace) -> int:
    base_url = args.base_url or capability.target.base_url

    policy = PolicyChecker.from_files(REPO_ROOT / "config" / "allowlist.yaml", REPO_ROOT / "config" / "risk_policy.yaml")

    run_context = RunContext(
        run_id=new_run_id(RunType.REPLAY),
        run_type=RunType.REPLAY,
        target=base_url,
        started_at=datetime.now(timezone.utc),
        capability_id=capability.id,
        capability_version=capability.version,
    )
    evidence = EvidenceLogger(REPO_ROOT / "evidence", run_context)
    print(f"run_id:     {run_context.run_id}")
    print(f"evidence:   {evidence.dir}")
    print(f"capability: {capability.id} v{capability.version}")
    print(f"inputs:     {inputs}")

    surface = await PlaywrightSurface.create(base_url=base_url, headless=not args.headed)
    try:
        await _pre_authenticate(surface, policy, evidence)
        result = await replay_capability(capability, inputs, surface=surface, policy=policy, evidence=evidence)
    finally:
        evidence.save_screenshot(await surface.snapshot(), name="final")
        await surface.close()

    evidence.save_result(result.model_dump_json(indent=2))

    print(f"status:     {result.status.value}")
    if result.outputs:
        print(f"outputs:    {result.outputs}")
    if result.business_outcome:
        print(f"outcome:    {result.business_outcome.code} - {result.business_outcome.message}")
    if result.failure:
        print(f"failure:    step={result.failure.step_id!r} expected={result.failure.expected!r} observed={result.failure.observed!r}")
    if result.recoverable_events:
        print(f"recovered:  {[e.condition for e in result.recoverable_events]}")

    return 0 if result.status is OutcomeStatus.SUCCESS else 2


def main() -> int:
    pre_parser = _make_pre_parser()
    known, _ = pre_parser.parse_known_args()

    artifact_path = Path(known.artifact_path)
    if not artifact_path.exists():
        print(f"error: artifact not found: {artifact_path}", file=sys.stderr)
        return 1
    capability = Capability.model_validate_json(artifact_path.read_text())

    parser = argparse.ArgumentParser(parents=[pre_parser], description=f"Replay capability '{capability.id}' (no LLM).")
    for param in capability.inputs:
        flag = f"--{param.name.replace('_', '-')}"
        parser.add_argument(flag, dest=param.name, required=param.required, default=None, help=param.description)
    args = parser.parse_args()

    if args.action_timeout_ms:
        capability = _override_timeouts(capability, args.action_timeout_ms)

    inputs = {p.name: getattr(args, p.name) for p in capability.inputs if getattr(args, p.name) is not None}

    return asyncio.run(main_async(capability, inputs, args))


if __name__ == "__main__":
    raise SystemExit(main())
