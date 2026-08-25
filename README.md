# Computer-Use Automation System

This project automates workflows in an application with no API. An LLM
finds the steps needed to complete a task through the UI during discovery.
A successful run compiles into a typed Capability that can be replayed
later without the LLM. Actions marked risky or irreversible pause
automation and hand the same browser session to a human.

```
Goal -> LLM discovery -> Capability -> deterministic replay
Risky action -> human handoff on same session -> automation resumes
```

Design decisions and trade-offs are in [`REPORT.md`](./REPORT.md). This
README covers setup and the main demo paths.

## Architecture

| Package | Purpose |
|---|---|
| `core/` | Capability schema, policy rules, `Surface` interface, evidence logging |
| `agent/` | Discovery loop, Groq client, recorder, compiler |
| `replay/` | Runs a compiled Capability, no LLM import |
| `escalation/` | Human handoff and session ownership |
| `mock_bank_app/` | The target app - a small legacy-style Flask app |
| `config/` | Allowlist and risk policy |
| `artifacts/` | Saved Capability JSON files |
| `evidence/` | Per-run logs and screenshots (see [`INDEX.md`](./evidence/INDEX.md)) |
| `scripts/` | CLI entry points used below |

## Prerequisites

- Python >= 3.11 (see `pyproject.toml`)
- A [Groq](https://console.groq.com/) API key - only needed for real discovery. Replay, the tests, and the mock app don't need one.
- Playwright's Chromium browser

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          

pip install -e ".[dev]"
playwright install chromium
```

## Start the mock bank

```bash
python -m mock_bank_app.app
```

Runs on `http://127.0.0.1:8000` by default, matching every script's default
`--base-url` below. If 8000 is already taken on your machine, run
`PORT=8010 python -m mock_bank_app.app` instead and add
`--base-url http://127.0.0.1:8010` to whatever you run next
(`config/allowlist.yaml` permits both ports).

One exception: the saved example artifact under `artifacts/` was recorded
against port 8010, and its checkpoints are pinned to that exact URL - replay
(below) always needs port 8010, regardless of which port you use for
discovery.

## Mock login

The scripts log in automatically with the local demo credentials
`operator` / `changeme123`. These are synthetic credentials for the mock
application only.

If you open the mock bank manually in a browser, use the same credentials
on the login page.

## Run discovery

```bash
python -m scripts.discover --headed
```

Uses Groq. You give it a natural-language goal, Groq decides the next UI
action, and Playwright executes it, with a policy check before every action.
Default goal: find member 12345, open a new savings sub-account, stop at the
review screen. Drop `--headed` for a headless run, or pass `--goal "..."`
for a different task. Needs `GROQ_API_KEY`; evidence goes to
`evidence/discovery-<run_id>/`.

## Run the full confirmation / handoff demo

```bash
python -m scripts.discover_to_confirmation --base-url http://127.0.0.1:8000 --headed
```

(Or `--base-url http://127.0.0.1:8010` on the override port.)

The agent reaches the review page on its own, then stops: the final
account-creation action is classified as irreversible, and it prints
`HUMAN APPROVAL REQUIRED`. Press Enter to approve, or type `cancel` to stop.
Approval runs the confirmation on the same browser session, then automation
takes back control, verifies the confirmation page loaded, and pulls the
confirmation number off it.

Terminal output is short by design. Full detail - accessibility tree,
locator attempts, every policy and ownership event - is still written to
`evidence/<run_id>/log.jsonl`.

## Compile a capability

```bash
python -m scripts.compile_example
```

Uses Groq. Runs a real discovery, records it, and compiles the recording
into a versioned Capability at
`artifacts/examples/open_savings_subaccount_review.json`. That file already
exists in the repo, so you don't need to run this to try replay below -
running it overwrites the existing artifact with a new one.

## Replay a capability

Replay reads the saved Capability, takes the parameters you supply, and runs
the recorded steps. No LLM call happens - `replay/engine.py` doesn't import
an LLM client at all.

```bash
PORT=8010 python -m mock_bank_app.app        # separate terminal
```

```bash
python -m scripts.replay artifacts/examples/open_savings_subaccount_review.json \
  --member-id 67890 --opening-deposit 250 \
  --base-url http://127.0.0.1:8010
```

The inputs here (`67890` / `250`) differ from the discovery run that
produced this artifact (`12345` / `100`) - same recorded steps, different
member and deposit amount.

## Error/outcome examples

The mock app has member IDs that trigger specific conditions on purpose (see
`mock_bank_app/data.py`):

| Member ID | Condition | Replay result |
|---|---|---|
| `40400` | not found | `BUSINESS_OUTCOME` |
| `40300` | permission denied | `HARD_FAILURE` |
| `40800` | slow response | recovered (retried) |
| `40900` | unexpected dialog | recovered (dismissed) |

```bash
# member not found
python -m scripts.replay artifacts/examples/open_savings_subaccount_review.json \
  --member-id 40400 --opening-deposit 100 --base-url http://127.0.0.1:8010

# permission denied
python -m scripts.replay artifacts/examples/open_savings_subaccount_review.json \
  --member-id 40300 --opening-deposit 100 --base-url http://127.0.0.1:8010
```

Both print a structured result and write full evidence. The
permission-denied run also saves a failure screenshot.

## Tests

```bash
python -m pytest tests/
```

204 tests pass, no live server or Groq key needed. Discovery, replay, and
escalation logic are tested against fakes/stubs; `tests/integration/` is the
exception, using a real headless browser and mock app instance.

## Evidence

[`evidence/INDEX.md`](./evidence/INDEX.md) points at the runs worth
reading: one clean discovery run, the saved capability, a replay success, a
business outcome, a hard failure, and a full human-handoff run. Everything
else in `evidence/` is kept from building this project but isn't required
reading. Logs and screenshots are committed on purpose - Section 6.3 of the
assignment asks for them as a deliverable, not build output.

## Safety

- **Allowlist** (`config/allowlist.yaml`) - which targets, routes, and action types automation may touch, checked before every action.
- **Route-aware risk rules** (`config/risk_policy.yaml`) - risk isn't decided from the model's intent string alone; a rule can override the intent-based classification for a given route and action type. See REPORT.md, Section 6, for the real bug this was built to fix.
- **Human confirmation** - actions classified as irreversible are blocked from automated execution and require human approval.
- **Redaction** - typed values like passwords and deposit amounts are never written to evidence logs.
- **Synthetic data only** - the mock app uses fake member records and a throwaway local login.

## Running without Groq

Only real discovery (`scripts.discover`, `scripts.discover_to_confirmation`,
`scripts.compile_example`) needs `GROQ_API_KEY`. Tests use a stub LLM client
and never touch the network; replay doesn't import an LLM client at all.
