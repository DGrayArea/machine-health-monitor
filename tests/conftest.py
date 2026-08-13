"""
Shared pytest fixtures.

The important thing here: tests must NEVER write to the real audit database.
We repoint config.DB_PATH and config.AUDIT_LOG_PATH at a temp directory before
anything imports them, so a test run cannot pollute the evidence trail that the
dashboard shows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Every test gets its own empty database and audit log."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "AUDIT_LOG_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr(config, "EXPORTS_DIR", tmp_path / "exports")

    from backend import database
    monkeypatch.setattr(database, "_conn", None)
    database.init_db(tmp_path / "test.db")
    yield
    if database._conn is not None:
        database._conn.close()
        database._conn = None


@pytest.fixture
def model_available() -> bool:
    from backend import predictor
    return predictor.is_ready()


@pytest.fixture
def client(monkeypatch):
    """FastAPI test client with the background simulator disabled."""
    monkeypatch.setattr(config, "SIM_AUTOSTART", False)
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers(client) -> dict:
    res = client.post("/api/auth/login", json={
        "username": config.DEMO_USERNAME,
        "password": config.DEMO_PASSWORD,
    })
    assert res.status_code == 200, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


@pytest.fixture
def normal_reading() -> dict:
    """A healthy operating point: ~6.9 kW, 10 K delta-T, fresh tool."""
    return {
        "air_temp": 298.1,
        "process_temp": 308.6,
        "rot_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0,
        "product_type": "M",
    }
