from __future__ import annotations

import shlex

from core.actions import ActionType
from core.schema import Action, Locator, LocatorCandidate, LocatorStrategy
from escalation.manager import EscalationManager

_HELP = "commands: state | screenshot | click <strategy> <value> | type <strategy> <value> <text> | resume [note]"


async def run_operator_console(manager: EscalationManager, *, input_fn=input, print_fn=print) -> None:
    request = manager.pending_request
    print_fn(f"=== OPERATOR CONSOLE === run={manager.evidence.run_id}")
    if request is not None:
        print_fn(f"paused at step '{request.step_id}': {request.reason}")
        print_fn(f"current url: {request.current_url}")
    print_fn(_HELP)

    while True:
        try:
            line = input_fn("operator> ").strip()
        except (EOFError, StopIteration):
            break
        if not line:
            continue

        try:
            parts = shlex.split(line)
        except ValueError as e:
            print_fn(f"could not parse command (unbalanced quotes?): {e}")
            continue
        cmd = parts[0].lower()

        if cmd in ("resume", "done"):
            note = " ".join(parts[1:]) if len(parts) > 1 else "human signaled resume"
            await manager.resume(note=note)
            print_fn(f"resumed - ownership returned to automation ({note})")
            break

        elif cmd == "state":
            observation = await manager.surface.observe()
            print_fn(f"url: {observation.url}")
            print_fn(observation.accessibility_tree[:1500])

        elif cmd == "screenshot":
            ref = manager.evidence.save_screenshot(await manager.surface.snapshot(), name="operator_manual")
            print_fn(f"saved: {ref.path}")

        elif cmd == "click" and len(parts) >= 3:
            strategy, value = parts[1], parts[2]
            action = Action(
                type=ActionType.CLICK,
                intent="human_click",
                target=Locator(
                    description=f"human-directed click: {value}",
                    candidates=[LocatorCandidate(strategy=LocatorStrategy(strategy), value=value)],
                ),
            )
            result = await manager.perform_human_action(action)
            print_fn(f"click -> success={result.success} error={result.error}")

        elif cmd == "type" and len(parts) >= 4:
            strategy, value, text = parts[1], parts[2], " ".join(parts[3:])
            action = Action(
                type=ActionType.TYPE,
                intent="human_type",
                value=text,
                target=Locator(
                    description=f"human-directed type: {value}",
                    candidates=[LocatorCandidate(strategy=LocatorStrategy(strategy), value=value)],
                ),
            )
            result = await manager.perform_human_action(action)
            print_fn(f"type -> success={result.success} error={result.error}")

        else:
            print_fn(f"unrecognized command: {line!r}")
            print_fn(_HELP)
