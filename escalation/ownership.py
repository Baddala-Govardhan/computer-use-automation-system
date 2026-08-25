from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Owner(str, Enum):
    AUTOMATION = "automation"
    HUMAN = "human"


class InvalidOwnershipTransition(RuntimeError):
    pass


@dataclass
class OwnershipTransition:
    from_owner: Owner
    to_owner: Owner
    reason: str
    timestamp: datetime


@dataclass
class SessionOwnership:
    owner: Owner = Owner.AUTOMATION
    history: list[OwnershipTransition] = field(default_factory=list)

    def require_automation(self) -> None:
        if self.owner is not Owner.AUTOMATION:
            raise InvalidOwnershipTransition(f"automation cannot act while '{self.owner.value}' owns the session")

    def require_human(self) -> None:
        if self.owner is not Owner.HUMAN:
            raise InvalidOwnershipTransition(f"human action rejected - '{self.owner.value}' currently owns the session")

    def transfer_to_human(self, reason: str) -> OwnershipTransition:
        if self.owner is not Owner.AUTOMATION:
            raise InvalidOwnershipTransition(f"cannot transfer to human: current owner is already '{self.owner.value}'")
        transition = OwnershipTransition(
            from_owner=Owner.AUTOMATION, to_owner=Owner.HUMAN, reason=reason, timestamp=datetime.now(timezone.utc)
        )
        self.owner = Owner.HUMAN
        self.history.append(transition)
        return transition

    def transfer_to_automation(self, reason: str) -> OwnershipTransition:
        if self.owner is not Owner.HUMAN:
            raise InvalidOwnershipTransition(f"cannot transfer to automation: current owner is already '{self.owner.value}'")
        transition = OwnershipTransition(
            from_owner=Owner.HUMAN, to_owner=Owner.AUTOMATION, reason=reason, timestamp=datetime.now(timezone.utc)
        )
        self.owner = Owner.AUTOMATION
        self.history.append(transition)
        return transition
