# Design Report

## 1. Architecture

```text
Natural-language goal
        |
        v
Discovery Agent + Groq
        |
        v
   PolicyChecker
        |
        v
      Surface
        |
        v
   Mock Bank UI
        |
        v
 Recorder -> Compiler
        |
        v
    Capability
        |
        v
 Replay Engine
    (No LLM)
```

Key decisions:

- Groq is used during discovery only.
- Accessibility-tree/ARIA is the primary perception signal.
- `Surface` separates UI-specific control from discovery/replay logic.
- Discovery and replay use the same `PolicyChecker`.
- `PlaywrightSurface` is the implemented web surface.

The review screen has two buttons that both say "Continue": one is a decorative
nav link, the other submits the form. Role+name alone can't tell them apart -
the accessibility tree can, because it shows which landmark each one sits in.
Screenshots would face the same disambiguation problem with less structure to
work from.

`PolicyChecker.check_action()` is called from exactly two places -
`agent/loop.py` and `replay/engine.py` - one gate, not two that could drift
apart.

## 2. Artifact schema

| Component  | Purpose                                 |
| ---------- | ---------------------------------------- |
| Version    | Identifies artifact version             |
| Inputs     | Typed invocation parameters             |
| Outputs    | Typed values returned to the caller     |
| Steps      | Ordered deterministic actions           |
| Locator    | Ranked candidates for finding a control |
| Checkpoint | Verifies expected state                 |

The real example, `artifacts/examples/open_savings_subaccount_review.json`,
was compiled from a real discovery run. The literal values the model typed
became named parameters:

```text
12345 -> {member_id}
100   -> {opening_deposit}
```

The same compiled steps later replay correctly with different values
(`67890` / `250`), no code changes.

The artifact carries none of the raw LLM transcript - no prompt, no
reasoning, no Groq-specific structure. It only keeps a pointer
(`discovery_run_id`) back to the evidence directory it came from.

## 3. Determinism & error handling

```text
Capability + Inputs
        |
        v
Parameter substitution
        |
        v
Policy check
        |
        v
Resolve locator
        |
        v
Execute action
        |
        v
Detect business/error state
        |
        v
Verify checkpoint
```

| Outcome            | Meaning                        | Example             |
| ------------------ | ------------------------------- | -------------------- |
| `SUCCESS`          | Capability completed            | Review page reached |
| `BUSINESS_OUTCOME` | Valid negative business result  | Member not found    |
| Recoverable        | Temporary/known condition       | Slow page or dialog |
| `HARD_FAILURE`     | Cannot safely continue          | Permission denied   |
| `ESCALATED`        | Human intervention required     | Irreversible action |

Real findings from building this:

- Replay does not call the LLM.
- Locator resolution fails closed on ambiguity - a candidate matching more
  than one element is treated as failed, never resolved with `.first()`.
- A URL-only checkpoint was not enough: a "member not found" page and a real
  member page can share the same URL pattern. Replay checks page content for
  a known error condition before accepting a checkpoint as passed.
- Dead-end detection fingerprints URL + accessibility tree + extracted text,
  not URL alone - a normal multi-field form stays on one URL for several
  steps and must not be mistaken for a stuck run.

## 4. Heterogeneity & multi-tenant

```text
                 Surface
                    |
          ---------------------
          |                   |
 PlaywrightSurface       DesktopSurface
   implemented              future
```

### Implemented

- `PlaywrightSurface`

### Designed, not implemented

- Desktop/OS-accessibility surface
- Production multi-tenant infrastructure

- `Surface` is the extension seam - discovery, replay, and escalation depend
  only on the interface, not on Playwright.
- Capability semantics do not depend directly on Playwright.
- `TargetRef.app` identifies the application/vendor, so a compiled capability
  is meant to be reusable across deployments of the same vendor UI.
- `TargetRef.tenant` allows specialization for one customer's customization.
- Locator/checkpoint failures expose UI drift as an ordinary `HARD_FAILURE`,
  not a special case.
- Artifact versioning allows a corrected capability to be shipped without
  ambiguity about which one is current.

## 5. Escalation & handoff

```text
AUTOMATION
    |
    v
Review page
    |
    v
Irreversible action detected
    |
    v
PAUSE
    |
    v
AUTOMATION -> HUMAN
    |
    v
Human confirms on SAME session
    |
    v
HUMAN -> AUTOMATION
    |
    v
Verify Confirmation page
    |
    v
Extract confirmation_number
    |
    v
SUCCESS
```

- `InterventionRequest` carries context: reason, current URL, a screenshot.
- Ownership is explicit - AUTOMATION/HUMAN, invalid transitions raise instead
  of silently succeeding.
- The same `PlaywrightSurface` is preserved - never a new session.
- Human actions are recorded as `human_action`, not a normal `action`.
- Automation does not act while HUMAN owns the session.
- After resume, automation independently re-observes state, re-checks the
  checkpoint, and re-extracts outputs - it does not trust "resume" as proof.

This was checked against a real run: the evidence log shows the handoff, one
`human_action` event with zero automated `action` events in between, an
independent checkpoint check (`verified_after_handoff: true`), and a real
confirmation number pulled off the page.

## 6. Safety

| Guardrail             | What it does                                                               |
| ---------------------- | --------------------------------------------------------------------------- |
| Allowlist              | Restricts permitted targets/routes/actions                                |
| Intent policy          | Classifies normal action intent                                           |
| Route-aware risk rule  | Protects risky controls even when the model uses an unexpected intent name |
| Human approval         | Required before irreversible action                                       |
| Redaction              | Prevents sensitive values from being persisted                            |

The most useful finding from this project came from a real run, not a design
review. The model reused the intent `submit_new_sub_account` for two
different things: the safe Form -> Review step, and the irreversible
Review -> Confirmation step, since both buttons say "Continue" in the UI.
Because the first use was correctly marked safe, the second one - the one
that actually creates the account - was let through without stopping for a
human.

This showed that an LLM-generated intent string cannot be the security
boundary, because the model isn't guaranteed to name two different actions
differently. The fix: risk policy can also key off route and action type. On
the review page, the control that submits the confirmation is classified
irreversible regardless of what intent name the model attaches to it, with
Cancel and navigation links carved out explicitly. Checked live: a later run
had the model reuse the same intent on the review page, and the evidence log
shows the route rule overriding it, correctly requiring human confirmation.

## 7. Cuts

Left out on purpose:

- No full operator web console.
- No desktop `Surface` implementation.
- No production multi-tenant infrastructure.
- No capability catalog/API.
- No LLM fallback during replay.
- No voice/chat UI.
- No scaling infrastructure.

These were left out to focus on the required end-to-end vertical slice. The
current interfaces leave room to add them later without changing the
discovery, artifact, or replay contracts.
