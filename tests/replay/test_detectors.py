from datetime import datetime, timezone

from core.surface import SurfaceObservation
from replay.detectors import detect


def make_observation(tree: str, url: str = "http://x/y") -> SurfaceObservation:
    return SurfaceObservation(url=url, title="t", accessibility_tree=tree, timestamp=datetime.now(timezone.utc))


def test_detects_member_not_found_as_business_outcome():
    detection = detect(make_observation("- text: Member not found"))
    assert detection.business_outcome is not None
    assert detection.business_outcome.code == "not_found"
    assert detection.hard_failure_reason is None


def test_detects_permission_denied_as_hard_failure_reason():
    detection = detect(make_observation("System Message: You do not have permission to perform this operation."))
    assert detection.hard_failure_reason is not None
    assert detection.business_outcome is None


def test_detects_validation_error_as_business_outcome():
    detection = detect(make_observation("Minimum opening deposit is $25.00."))
    assert detection.business_outcome is not None
    assert detection.business_outcome.code == "validation_error"


def test_clean_page_detects_nothing():
    detection = detect(make_observation("- heading \"Member Search\" [level=2]"))
    assert detection.business_outcome is None
    assert detection.hard_failure_reason is None


def test_permission_denied_takes_priority_over_not_found_wording_collision():
    # Defensive: if a page somehow mentioned both, permission (hard failure)
    # should win - it's the more severe condition.
    detection = detect(make_observation("You do not have permission to view this record; it may not be found."))
    assert detection.hard_failure_reason is not None
