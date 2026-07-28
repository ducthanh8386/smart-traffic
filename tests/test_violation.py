from core.storage import ViolationStorage
from core.violation import ViolationDetector


def make_detector(tmp_path, direction: str) -> ViolationDetector:
    return ViolationDetector(
        storage=ViolationStorage(tmp_path / "violations.sqlite3"),
        evidence_dir=tmp_path / "evidence",
        save_evidence=False,
        crossing_direction=direction,
    )


def test_crossed_line_respects_down_direction(tmp_path) -> None:
    detector = make_detector(tmp_path, "down")
    line = ((0, 10), (100, 10))

    assert detector._crossed_line(1, (50, 5), line) is False
    assert detector._crossed_line(1, (50, 12), line) is True


def test_crossed_line_rejects_wrong_direction(tmp_path) -> None:
    detector = make_detector(tmp_path, "down")
    line = ((0, 10), (100, 10))

    assert detector._crossed_line(1, (50, 15), line) is False
    assert detector._crossed_line(1, (50, 8), line) is False


def test_crossed_line_respects_up_direction(tmp_path) -> None:
    detector = make_detector(tmp_path, "up")
    line = ((0, 10), (100, 10))

    assert detector._crossed_line(1, (50, 15), line) is False
    assert detector._crossed_line(1, (50, 8), line) is True


def test_crossed_line_allows_both_directions(tmp_path) -> None:
    detector = make_detector(tmp_path, "both")
    line = ((0, 10), (100, 10))

    assert detector._crossed_line(1, (50, 5), line) is False
    assert detector._crossed_line(1, (50, 12), line) is True
    assert detector._crossed_line(2, (50, 15), line) is False
    assert detector._crossed_line(2, (50, 8), line) is True


def test_red_light_violation_only_logs_on_red_and_once(tmp_path) -> None:
    storage = ViolationStorage(tmp_path / "violations.sqlite3")
    detector = ViolationDetector(
        storage=storage,
        evidence_dir=tmp_path / "evidence",
        save_evidence=False,
        crossing_direction="down",
    )
    line = ((0, 10), (100, 10))
    obj = {
        "track_id": 1,
        "bbox": (0, 0, 20, 20),
        "class_name": "car",
        "confidence": 0.9,
        "center_point": (50, 5),
    }

    assert detector.check_red_light_violation(None, [obj], line, "GREEN", "s", 1) == []
    obj["center_point"] = (50, 12)
    assert detector.check_red_light_violation(None, [obj], line, "GREEN", "s", 2) == []

    obj["track_id"] = 2
    obj["center_point"] = (50, 5)
    assert detector.check_red_light_violation(None, [obj], line, "RED", "s", 3) == []
    obj["center_point"] = (50, 12)
    violations = detector.check_red_light_violation(None, [obj], line, "RED", "s", 4)
    assert len(violations) == 1
    obj["center_point"] = (50, 5)
    assert detector.check_red_light_violation(None, [obj], line, "RED", "s", 5) == []
    obj["center_point"] = (50, 12)
    assert detector.check_red_light_violation(None, [obj], line, "RED", "s", 6) == []
    assert len(storage.list_recent()) == 1

def test_wrong_lane_violation(tmp_path) -> None:
    storage = ViolationStorage(tmp_path / "violations.sqlite3")
    detector = ViolationDetector(
        storage=storage,
        evidence_dir=tmp_path / "evidence",
        save_evidence=False,
    )
    lanes_config = [
        {
            "name": "Lane Oto",
            "allowed_classes": ["car"],
            "roi_ratio": {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 1.0}
        },
        {
            "name": "Lane Xe May",
            "allowed_classes": ["motorcycle"],
            "roi_ratio": {"x1": 0.5, "y1": 0.0, "x2": 1.0, "y2": 1.0}
        }
    ]

    obj1 = {
        "track_id": 1,
        "bbox": (20, 45, 30, 55),
        "class_name": "motorcycle",
        "confidence": 0.85,
        "center_point": (25, 50),
    }

    obj2 = {
        "track_id": 2,
        "bbox": (20, 45, 30, 55),
        "class_name": "car",
        "confidence": 0.90,
        "center_point": (25, 50),
    }

    violations = detector.check_wrong_lane_violation(None, [obj1], lanes_config, 100, 100, "session1", 1)
    assert len(violations) == 1
    assert violations[0]["violation_type"] == "wrong_lane_violation"
    assert violations[0]["class_name"] == "motorcycle"

    violations2 = detector.check_wrong_lane_violation(None, [obj2], lanes_config, 100, 100, "session1", 1)
    assert len(violations2) == 0

    violations_dup = detector.check_wrong_lane_violation(None, [obj1], lanes_config, 100, 100, "session1", 2)
    assert len(violations_dup) == 0

