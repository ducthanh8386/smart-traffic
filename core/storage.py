from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any


VIOLATION_COLUMNS = [
    "timestamp",
    "session_id",
    "frame_index",
    "track_id",
    "class_name",
    "violation_type",
    "confidence",
    "evidence_path",
]


class ViolationStorage:
    """Thread-safe SQLite storage for violation events."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.lock = Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self.lock, self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_path TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp)"
            )

    def append(self, violation: dict[str, Any]) -> None:
        row = {column: violation.get(column, "") for column in VIOLATION_COLUMNS}
        with self.lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO violations (
                    timestamp, session_id, frame_index, track_id, class_name,
                    violation_type, confidence, evidence_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(row[column] for column in VIOLATION_COLUMNS),
            )

    def list_recent(self, limit: int = 500) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 2000))
        with self.lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT timestamp, session_id, frame_index, track_id, class_name,
                       violation_type, confidence, evidence_path
                FROM violations
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]


def get_violation_storage(db_path: str | Path) -> ViolationStorage:
    return _get_violation_storage(str(Path(db_path).resolve()))


@lru_cache(maxsize=8)
def _get_violation_storage(resolved_db_path: str) -> ViolationStorage:
    return ViolationStorage(resolved_db_path)
