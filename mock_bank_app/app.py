from __future__ import annotations

import time
import uuid
from decimal import Decimal, InvalidOperation
from functools import wraps

from flask import Flask, redirect, render_template, request, session, url_for

from mock_bank_app.data import (
    ACCOUNT_TYPES,
    CONFIRMED_SUBACCOUNTS,
    MEMBERS,
    MIN_OPENING_DEPOSIT,
    SLOW_RESPONSE_DELAY_SECONDS,
    SPECIAL_MEMBER_IDS,
    SubAccount,
)

OPERATOR_USERNAME = "operator"
OPERATOR_PASSWORD = "changeme123"

app = Flask(__name__)
app.secret_key = "dev-only-secret-not-for-production"


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def get_special_condition(member_id: str) -> str | None:
    return SPECIAL_MEMBER_IDS.get(member_id)


def get_member(member_id: str):
    return MEMBERS.get(member_id)


@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("members_search") if session.get("logged_in") else url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == OPERATOR_USERNAME and password == OPERATOR_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("members_search"))
        return render_template("login.html", error="Invalid username or password."), 401
    return render_template("login.html", error=None)


@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/members/search", methods=["GET"])
@login_required
def members_search():
    query = request.args.get("q", "").strip()
    if not query:
        return render_template("search.html", query=None, results=None)
    if query.isdigit():
        return redirect(url_for("member_detail", member_id=query))
    results = [m for m in MEMBERS.values() if query.lower() in m.name.lower()]
    return render_template("search.html", query=query, results=results)


@app.route("/members/<member_id>", methods=["GET"])
@login_required
def member_detail(member_id):
    condition = get_special_condition(member_id)
    if condition == "not_found":
        return render_template("not_found.html", member_id=member_id)
    if condition == "permission_denied":
        return render_template("permission_denied.html", member_id=member_id)

    member = get_member(member_id)
    if member is None:
        return render_template("not_found.html", member_id=member_id)

    if condition == "slow_response":
        time.sleep(SLOW_RESPONSE_DELAY_SECONDS)

    session["current_member_id"] = member.id
    return render_template(
        "member_detail.html", member=member, trigger_dialog=(condition == "unexpected_dialog")
    )


@app.route("/members/<member_id>/subaccounts/new", methods=["GET", "POST"])
@login_required
def subaccount_new(member_id):
    member = get_member(member_id)
    if member is None:
        return render_template("not_found.html", member_id=member_id)

    if request.method == "POST":
        account_type = request.form.get("account_type", ACCOUNT_TYPES[0])
        raw_deposit = request.form.get("opening_deposit", "")
        try:
            deposit = Decimal(raw_deposit)
        except (InvalidOperation, ValueError):
            deposit = None

        if deposit is None or deposit < MIN_OPENING_DEPOSIT:
            return render_template(
                "subaccount_new.html",
                member=member,
                account_types=ACCOUNT_TYPES,
                form_account_type=account_type,
                form_opening_deposit=raw_deposit,
                field_error=f"Minimum opening deposit is ${MIN_OPENING_DEPOSIT:.2f}.",
            )

        session["draft"] = {
            "member_id": member.id,
            "account_type": account_type,
            "opening_deposit": str(deposit),
        }
        return redirect(url_for("subaccount_review", member_id=member.id))

    return render_template(
        "subaccount_new.html",
        member=member,
        account_types=ACCOUNT_TYPES,
        form_account_type=None,
        form_opening_deposit=None,
        field_error=None,
    )


def _current_draft(member_id: str) -> dict | None:
    draft = session.get("draft")
    if not draft or draft.get("member_id") != member_id:
        return None
    return draft


@app.route("/members/<member_id>/subaccounts/review", methods=["GET", "POST"])
@login_required
def subaccount_review(member_id):
    member = get_member(member_id)
    if member is None:
        return render_template("not_found.html", member_id=member_id)

    draft = _current_draft(member_id)
    if draft is None:
        return redirect(url_for("subaccount_new", member_id=member_id))

    if request.method == "POST":
        action = request.form.get("action")
        if action == "cancel":
            session.pop("draft", None)
            return redirect(url_for("member_detail", member_id=member_id))
        if action == "confirm":
            session["confirmed_draft"] = draft
            session.pop("draft", None)
            return redirect(url_for("subaccount_confirmation", member_id=member_id))

    return render_template(
        "subaccount_review.html",
        member=member,
        draft={"account_type": draft["account_type"], "opening_deposit": Decimal(draft["opening_deposit"])},
    )


@app.route("/members/<member_id>/subaccounts/confirmation", methods=["GET"])
@login_required
def subaccount_confirmation(member_id):
    member = get_member(member_id)
    if member is None:
        return render_template("not_found.html", member_id=member_id)

    confirmed = session.get("confirmed_draft")
    if not confirmed or confirmed.get("member_id") != member_id:
        return redirect(url_for("member_detail", member_id=member_id))

    confirmation_number = f"SA-{uuid.uuid4().hex[:8].upper()}"
    record = SubAccount(
        id=uuid.uuid4().hex,
        member_id=member.id,
        account_type=confirmed["account_type"],
        opening_deposit=Decimal(confirmed["opening_deposit"]),
        confirmation_number=confirmation_number,
    )
    CONFIRMED_SUBACCOUNTS.append(record)
    session.pop("confirmed_draft", None)

    return render_template(
        "subaccount_confirmation.html",
        member=member,
        draft={"account_type": record.account_type, "opening_deposit": record.opening_deposit},
        confirmation_number=confirmation_number,
    )


@app.route("/_dev/reset", methods=["POST"])
def dev_reset():
    session.clear()
    CONFIRMED_SUBACCOUNTS.clear()
    return {"status": "reset"}, 200


if __name__ == "__main__":
    import os

    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), threaded=True)
