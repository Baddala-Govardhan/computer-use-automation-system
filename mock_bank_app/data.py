from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class Member:
    id: str
    name: str
    email: str
    ssn: str
    savings_balance: Decimal
    checking_balance: Decimal


@dataclass
class SubAccount:
    id: str
    member_id: str
    account_type: str
    opening_deposit: Decimal
    confirmation_number: str


MEMBERS: dict[str, Member] = {
    "12345": Member(
        id="12345", name="Alicia Torres", email="alicia.torres@example.com", ssn="123-45-6789",
        savings_balance=Decimal("1204.55"), checking_balance=Decimal("389.10"),
    ),
    "67890": Member(
        id="67890", name="Marcus Webb", email="marcus.webb@example.com", ssn="234-56-7890",
        savings_balance=Decimal("58210.00"), checking_balance=Decimal("1200.00"),
    ),
    "11111": Member(
        id="11111", name="Priya Natarajan", email="priya.n@example.com", ssn="345-67-8901",
        savings_balance=Decimal("430.20"), checking_balance=Decimal("75.00"),
    ),
    "40800": Member(
        id="40800", name="Dana Whitfield", email="dana.whitfield@example.com", ssn="456-78-9012",
        savings_balance=Decimal("2310.40"), checking_balance=Decimal("512.00"),
    ),
    "40900": Member(
        id="40900", name="Omar Reyes", email="omar.reyes@example.com", ssn="567-89-0123",
        savings_balance=Decimal("875.00"), checking_balance=Decimal("120.15"),
    ),
}

SPECIAL_MEMBER_IDS: dict[str, str] = {
    "40400": "not_found",
    "40300": "permission_denied",
    "40800": "slow_response",
    "40900": "unexpected_dialog",
}

SLOW_RESPONSE_DELAY_SECONDS = 4

MIN_OPENING_DEPOSIT = Decimal("25.00")
ACCOUNT_TYPES = ["Savings", "Money Market"]

CONFIRMED_SUBACCOUNTS: list[SubAccount] = []
