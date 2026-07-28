from core.storage import ViolationStorage, get_violation_storage


def test_storage_appends_and_reads_recent_rows(tmp_path) -> None:
    storage = ViolationStorage(tmp_path / "violations.sqlite3")

    storage.append(
        {
            "timestamp": "2026-06-01 10:00:00",
            "session_id": "abc",
            "frame_index": 42,
            "track_id": 7,
            "class_name": "car",
            "violation_type": "red_light_violation",
            "confidence": 0.91,
            "evidence_path": "/api/evidence/red_light/example.jpg",
        }
    )

    rows = storage.list_recent()

    assert rows == [
        {
            "timestamp": "2026-06-01 10:00:00",
            "session_id": "abc",
            "frame_index": 42,
            "track_id": 7,
            "class_name": "car",
            "violation_type": "red_light_violation",
            "confidence": 0.91,
            "evidence_path": "/api/evidence/red_light/example.jpg",
        }
    ]


def test_storage_cache_reuses_instance(tmp_path) -> None:
    db_path = tmp_path / "violations.sqlite3"

    assert get_violation_storage(db_path) is get_violation_storage(db_path)
