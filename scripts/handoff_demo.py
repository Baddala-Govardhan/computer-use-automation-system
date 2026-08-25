from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from agent.playwright_surface import PlaywrightSurface
from core.evidence import EvidenceLogger, RunContext, RunType, new_run_id
from core.outcomes import OutcomeStatus
from core.policy import PolicyChecker
from core.schema import (
    Action,
    ActionType,
    Capability,
    CapabilityMetadata,
    Checkpoint,
    CheckpointType,
    Locator,
    LocatorCandidate,
    LocatorStrategy,
    OutputDefinition,
    ParamType,
    Step,
    TargetRef,
)
from escalation.manager import EscalationManager
from escalation.operator import run_operator_console
from escalation.ownership import Owner
from replay.engine import replay_capability
from scripts.discover import _pre_authenticate

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8000"
ARTIFACT_PATH = REPO_ROOT / "artifacts" / "examples" / "open_savings_subaccount_review.json"

MEMBER_ID = "11111"
OPENING_DEPOSIT = "300"


def build_confirm_capability(base_url: str) -> Capability:
    locator = Locator(
        description="Confirm button on review screen",
        candidates=[
            LocatorCandidate(strategy=LocatorStrategy.ROLE, value="button:Continue", confidence=1.0),
            LocatorCandidate(
                strategy=LocatorStrategy.CSS,
                value="main form button[value='confirm']",
                confidence=0.5,
                notes="role+name alone is ambiguous with the review page's decorative nav Continue trap",
            ),
        ],
    )
    return Capability(
        id="confirm_new_savings_subaccount",
        name="Confirm New Savings Sub-Account",
        version="1.0.0",
        description=(
            "Clicks Continue on an already-open sub-account review screen, permanently creating the "
            "sub-account. Hand-authored for the escalation demo, not discovered/compiled - deliberately "
            "kept separate from open_savings_subaccount_review, which intentionally ends at review."
        ),
        target=TargetRef(app="mock_bank_app", base_url=base_url),
        steps=[
            Step(
                id="step_0",
                description="Click Continue to permanently confirm the new sub-account.",
                action=Action(type=ActionType.CLICK, intent="confirm_new_sub_account", target=locator),
            )
        ],
        success_checkpoint=Checkpoint(
            description="Reached the confirmation page",
            type=CheckpointType.URL_MATCHES,
            url_pattern=f"{base_url}/members/*/subaccounts/confirmation",
        ),
        outputs=[
            OutputDefinition(
                name="confirmation_number",
                type=ParamType.STRING,
                description="The confirmation number generated when the new sub-account was created.",
                source=Locator(
                    description="Confirmation Number cell",
                    candidates=[
                        LocatorCandidate(
                            strategy=LocatorStrategy.XPATH,
                            value="//td[text()='Confirmation Number']/following-sibling::td[1]",
                            notes="Sibling-cell XPath - the only stable option in a legacy table layout with no test ids.",
                        )
                    ],
                ),
            )
        ],
        metadata=CapabilityMetadata(
            created_at=datetime.now(timezone.utc),
            discovery_run_id="hand-authored-not-discovered",
            model_used="none (hand-authored for the escalation demo)",
            author="escalation-demo-script",
        ),
    )


async def main() -> int:
    policy = PolicyChecker.from_files(REPO_ROOT / "config" / "allowlist.yaml", REPO_ROOT / "config" / "risk_policy.yaml")
    review_capability = Capability.model_validate_json(ARTIFACT_PATH.read_text())
    confirm_capability = build_confirm_capability(BASE_URL)

    run_context = RunContext(
        run_id=new_run_id(RunType.REPLAY),
        run_type=RunType.REPLAY,
        target=BASE_URL,
        started_at=datetime.now(timezone.utc),
        capability_id=confirm_capability.id,
        capability_version=confirm_capability.version,
    )
    evidence = EvidenceLogger(REPO_ROOT / "evidence", run_context)
    print(f"run_id:   {run_context.run_id}")
    print(f"evidence: {evidence.dir}")

    surface = await PlaywrightSurface.create(base_url=BASE_URL, headless=True)
    passed = False
    manager: EscalationManager | None = None
    try:
        await _pre_authenticate(surface, policy, evidence)

        print(f"\n--- phase 1: replay '{review_capability.id}' to reach review (member={MEMBER_ID}) ---")
        review_result = await replay_capability(
            review_capability,
            {"member_id": MEMBER_ID, "opening_deposit": OPENING_DEPOSIT},
            surface=surface,
            policy=policy,
            evidence=evidence,
        )
        print(f"review replay status: {review_result.status.value}")
        assert review_result.status is OutcomeStatus.SUCCESS, "demo requires reaching review first"
        print(f"current url: {surface.current_url()}")

        print(f"\n--- phase 2: attempt '{confirm_capability.id}' (expected to be refused, needs human) ---")
        confirm_attempt = await replay_capability(confirm_capability, {}, surface=surface, policy=policy, evidence=evidence)
        print(f"confirm attempt status: {confirm_attempt.status.value}")
        assert confirm_attempt.status is OutcomeStatus.ESCALATED
        assert confirm_attempt.failure is not None
        print(f"reason: {confirm_attempt.failure.message}")
        print(f"url unchanged (automation never clicked): {surface.current_url()}")

        print("\n--- phase 3: human takes over the SAME live session ---")
        manager = EscalationManager(surface, evidence)
        request = await manager.raise_intervention(
            capability=confirm_capability,
            step_id=confirm_attempt.failure.step_id,
            reason=confirm_attempt.failure.message,
            risk_level="irreversible",
        )
        print(f"InterventionRequest: step={request.step_id!r} reason={request.reason!r}")
        print(f"ownership: {manager.ownership.owner.value}")

        scripted_commands = iter(
            [
                "state",
                "click role button:Continue",
                "click css \"main form button[value='confirm']\"",
                "resume human reviewed and approved the new sub-account",
            ]
        )

        def scripted_input(prompt: str = "") -> str:
            return next(scripted_commands)

        await run_operator_console(manager, input_fn=scripted_input, print_fn=print)

        print(f"\nownership after resume: {manager.ownership.owner.value}")
        assert manager.ownership.owner is Owner.AUTOMATION

        print("\n--- phase 4: automation verifies what the human left behind and extracts outputs ---")
        final_result = await manager.verify_and_complete(confirm_capability)
        print(f"final status: {final_result.status.value}")
        if final_result.failure is not None:
            print(f"failure: expected={final_result.failure.expected!r} observed={final_result.failure.observed!r}")
        print(f"outputs: {final_result.outputs}")
        passed = final_result.status is OutcomeStatus.SUCCESS

        print(f"\nhuman actions recorded: {len(manager.human_actions)}")
        for ha in manager.human_actions:
            print(f"  - {ha.action.type.value} ({ha.action.intent}) success={ha.success}")
        print(
            "ownership transition history: "
            f"{[(t.from_owner.value, t.to_owner.value, t.reason) for t in manager.ownership.history]}"
        )
    finally:
        evidence.save_screenshot(await surface.snapshot(), name="final")
        await surface.close()

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
