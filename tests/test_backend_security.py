import pytest
from fastapi import HTTPException

from backend.main import resolve_evidence_path


def test_evidence_path_rejects_traversal() -> None:
    with pytest.raises(HTTPException):
        resolve_evidence_path("../logs/violations.sqlite3")
