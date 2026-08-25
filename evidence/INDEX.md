# Evidence Index

This directory contains real, unedited run evidence (`log.jsonl` +
screenshots) from live discovery and replay runs against the mock bank app.
Most subdirectories here are development/iteration runs, preserved but not
required reading. The six runs below are the canonical evidence for review.

## Genuine LLM discovery

`discovery-20260821T080757Z-e5c9cd/`

A real Groq-driven discovery run (goal: "Find member 12345, open a new
savings sub-account, and stop at the review screen"). Shows a full
observe -> decide -> act loop against the live mock app, with a
`policy_decision` logged before every action and a clean
`stop_reason=goal_complete`. Proves real LLM-driven UI interaction, not a
scripted trace.

## Saved capability artifact

[`../artifacts/examples/open_savings_subaccount_review.json`](../artifacts/examples/open_savings_subaccount_review.json)

The compiled `Capability` produced from a real discovery recording: 5
ordered steps, 2 typed inputs (`member_id`, `opening_deposit`), per-step and
final URL-match checkpoints, ranked locator candidates. Not hand-written -
see `agent/compiler.py` / `scripts/compile_example.py`.

## Deterministic replay success

`replay-20260821T081832Z-1a38ae/`

Replays the saved capability above with **no LLM involved** (`replay/engine.py`
imports no LLM client at all). Uses invocation inputs different from the
discovery run that produced the artifact, proving the same compiled steps
generalize to new inputs, not just the ones they were recorded with.

## Business outcome

`replay-20260820T222436Z-dc1574/`

Same capability, replayed against member `40400` (mock app's injectable
"not found" condition). Ends in `OutcomeStatus.BUSINESS_OUTCOME`
(`member_not_found`) - a legitimate, named result, not a crash or a
misclassified failure.

## Hard failure

`replay-20260820T222457Z-dd737d/`

Same capability, replayed against member `40300` (injectable "permission
denied" condition). Ends in `OutcomeStatus.HARD_FAILURE` with structured
step/expected/observed detail and a failure screenshot
(`screenshots/failure_step_1.png`).

## Human handoff / confirmation

`discovery-20260825T042330Z-929b8a/`

The full end-to-end flow, live-verified event by event:

```
Review reached
  -> route-aware policy classifies the confirmation action irreversible
     (the model actually proposed intent 'submit_new_sub_account' - the
     same intent already safe on the prior page; the route rule, not the
     intent string, is what caught it)
  -> control_transfer: automation -> human
  -> human_action (click) - recorded distinctly from a normal automated
     action; zero automated `action` events occur before the next transfer
  -> control_transfer: human -> automation
  -> checkpoint independently re-verified (verified_after_handoff: true)
  -> confirmation_number extracted at runtime (SA-EF6560A0, confirmed
     against the actual rendered confirmation screenshot, not assumed from
     terminal output)
  -> SUCCESS
```

One consistent `run_id` runs through every event in this sequence -
discovery, escalation, human action, resume, verification, extraction, and
completion.
