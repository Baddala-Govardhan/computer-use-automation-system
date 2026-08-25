from datetime import datetime, timezone
from pathlib import Path

import json

from core.evidence import (
    EvidenceEventType,
    EvidenceLogger,
    RunContext,
    RunType,
    new_run_id,
    redact,
)


def make_context(tmp_run_id: str | None = None) -> RunContext:
    return RunContext(
        run_id=tmp_run_id or new_run_id(RunType.DISCOVERY),
        run_type=RunType.DISCOVERY,
        target="mock_bank_app",
        started_at=datetime.now(timezone.utc),
        goal="look up member 12345",
    )


def test_new_run_id_is_prefixed_and_unique():
    a, b = new_run_id(RunType.DISCOVERY), new_run_id(RunType.DISCOVERY)
    assert a.startswith("discovery-")
    assert a != b


def test_redact_masks_sensitive_keys_case_insensitively():
    out = redact({"Password": "hunter2", "member_id": "12345"})
    assert out["Password"] == "[REDACTED]"
    assert out["member_id"] == "12345"


def test_redact_masks_pii_patterns_in_free_text():
    out = redact("SSN 123-45-6789, contact a@b.com")
    assert "123-45-6789" not in out
    assert "a@b.com" not in out
    assert "REDACTED" in out


def test_redact_recurses_into_nested_lists_and_dicts():
    out = redact({"steps": [{"note": "email me at a@b.com"}, {"token": "abc123"}]})
    assert "a@b.com" not in out["steps"][0]["note"]
    assert out["steps"][1]["token"] == "[REDACTED]"


def test_evidence_logger_creates_run_directory_layout(tmp_path: Path):
    ctx = make_context()
    logger = EvidenceLogger(tmp_path, ctx)
    assert (logger.dir / "context.json").exists()
    assert logger.screenshots_dir.exists()


def test_evidence_logger_never_writes_secret_to_log(tmp_path: Path):
    ctx = make_context()
    logger = EvidenceLogger(tmp_path, ctx)
    logger.log(EvidenceEventType.ACTION, "typed password", data={"password": "hunter2"})
    contents = (logger.dir / "log.jsonl").read_text()
    assert "hunter2" not in contents


def test_evidence_logger_events_have_monotonic_seq(tmp_path: Path):
    ctx = make_context()
    logger = EvidenceLogger(tmp_path, ctx)
    logger.log(EvidenceEventType.RUN_STARTED, "started")
    logger.log(EvidenceEventType.ACTION, "did something")
    lines = (logger.dir / "log.jsonl").read_text().strip().splitlines()
    seqs = [json.loads(line)["seq"] for line in lines]
    assert seqs == [1, 2]


def test_evidence_logger_saves_screenshot_and_returns_ref(tmp_path: Path):
    ctx = make_context()
    logger = EvidenceLogger(tmp_path, ctx)
    ref = logger.save_screenshot(b"\x89PNG-fake-bytes", name="step_1")
    assert (logger.dir / ref.path).exists()
    assert ref.path == "screenshots/step_1.png"
