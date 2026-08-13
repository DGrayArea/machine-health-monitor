"""
The audit trail. Every prediction and alert is written to durable storage.

Two sinks, on purpose:
  SQLite (outputs/logs/monitoring.db) is queryable. The dashboard's alert
  history and the downloadable reports are both queries against it.

  JSONL (outputs/logs/audit_log.jsonl) is append-only, one JSON object per line,
  never updated or deleted. If the database is tampered with, the flat log still
  shows what the system saw and what it said.

For maintenance work this is not optional. If the machine breaks and the system
said "Normal" five minutes earlier, you need to be able to show exactly what
readings it was given.

This uses the stdlib sqlite3 rather than an ORM. The schema is two tables, and
keeping the SQL visible makes the data model easier to explain.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import config

# SQLite allows one writer at a time. The simulator thread and the HTTP handlers
# both write, so writes are guarded by a lock and the connection is opened with
# check_same_thread=False.
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    source        TEXT    NOT NULL,   -- 'api' | 'simulator'
    username      TEXT,
    air_temp      REAL    NOT NULL,
    process_temp  REAL    NOT NULL,
    rot_speed     REAL    NOT NULL,
    torque        REAL    NOT NULL,
    tool_wear     REAL    NOT NULL,
    product_type  TEXT    NOT NULL,
    temp_diff     REAL    NOT NULL,
    power         REAL    NOT NULL,
    strain        REAL    NOT NULL,
    status        TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    probabilities TEXT    NOT NULL,   -- JSON blob
    rul_minutes   REAL,               -- remaining useful life, cutting minutes
    rul_binding   TEXT                -- 'tool_wear' | 'overstrain'
);

CREATE TABLE IF NOT EXISTS alerts (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id      INTEGER REFERENCES predictions(id),
    timestamp          TEXT NOT NULL,
    severity           TEXT NOT NULL,   -- 'Warning' | 'Critical'
    status             TEXT NOT NULL,   -- model status that caused it
    title              TEXT NOT NULL,
    message            TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    confidence         REAL NOT NULL,
    source             TEXT NOT NULL,
    triggered_rules    TEXT NOT NULL    -- JSON blob
);

-- The dashboard always asks for the most recent N, so index the sort column.
CREATE INDEX IF NOT EXISTS idx_pred_ts  ON predictions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alert_ts ON alerts(timestamp DESC);
"""


def utc_now() -> str:
    """
    ISO-8601 UTC. Always store UTC and convert to local time only for display.

    Millisecond precision rather than seconds, because MHM_SIM_INTERVAL can be
    set below 1 s and the RUL wear-rate estimator fits a slope against these
    timestamps. At second resolution a fast tick rate would give several readings
    the same timestamp, the slope would divide by zero, and the clock-time
    projection would quietly disappear.
    """
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# Columns added after the first release. CREATE TABLE IF NOT EXISTS leaves an
# existing table alone, so a database created before RUL existed would keep the
# old shape and every insert would fail. SQLite has no ADD COLUMN IF NOT EXISTS,
# so check PRAGMA table_info first.
MIGRATIONS: dict[str, list[tuple[str, str]]] = {
    "predictions": [
        ("rul_minutes", "REAL"),
        ("rul_binding", "TEXT"),
    ],
}


def _apply_migrations(conn: sqlite3.Connection) -> None:
    for table, columns in MIGRATIONS.items():
        existing = {row["name"] for row in
                    conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, sql_type in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
    conn.commit()


def get_connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
        _conn.commit()
        _apply_migrations(_conn)
    return _conn


def init_db(db_path: Path | None = None) -> None:
    """Create the schema. Passing db_path lets tests point at a temporary file."""
    global _conn
    if db_path is not None:
        config.DB_PATH = db_path
        if _conn is not None:
            _conn.close()
        _conn = None
    get_connection()


def _append_audit(event_type: str, payload: dict[str, Any]) -> None:
    """Append-only flat log. A failure here must never break a request."""
    try:
        config.AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"event": event_type, "logged_at": utc_now(), **payload},
                          default=str)
        with config.AUDIT_LOG_PATH.open("a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass  # the flat log is best-effort, the database is the record


def log_prediction(
    *,
    reading: dict[str, Any],
    derived: dict[str, float],
    status: str,
    confidence: float,
    probabilities: dict[str, float],
    source: str,
    username: str | None = None,
    timestamp: str | None = None,
    remaining_life: dict[str, Any] | None = None,
) -> int:
    """Save one prediction. Returns the row id so an alert can point at it."""
    ts = timestamp or utc_now()
    life = remaining_life or {}
    conn = get_connection()
    with _lock:
        cur = conn.execute(
            """INSERT INTO predictions
               (timestamp, source, username, air_temp, process_temp, rot_speed,
                torque, tool_wear, product_type, temp_diff, power, strain,
                status, confidence, probabilities, rul_minutes, rul_binding)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ts, source, username,
                reading["air_temp"], reading["process_temp"], reading["rot_speed"],
                reading["torque"], reading["tool_wear"], reading.get("product_type", "M"),
                derived["temp_diff"], derived["power"], derived["strain"],
                status, confidence, json.dumps(probabilities),
                life.get("remaining_min"), life.get("binding_constraint"),
            ),
        )
        conn.commit()
        prediction_id = int(cur.lastrowid)

    _append_audit("prediction", {
        "prediction_id": prediction_id, "timestamp": ts, "source": source,
        "username": username, "reading": reading, "derived": derived,
        "status": status, "confidence": confidence, "probabilities": probabilities,
        "remaining_life": life or None,
    })
    return prediction_id


def log_alert(
    *,
    prediction_id: int | None,
    severity: str,
    status: str,
    title: str,
    message: str,
    recommended_action: str,
    confidence: float,
    source: str,
    triggered_rules: list[dict[str, Any]],
    timestamp: str | None = None,
) -> int:
    ts = timestamp or utc_now()
    conn = get_connection()
    with _lock:
        cur = conn.execute(
            """INSERT INTO alerts
               (prediction_id, timestamp, severity, status, title, message,
                recommended_action, confidence, source, triggered_rules)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (prediction_id, ts, severity, status, title, message,
             recommended_action, confidence, source, json.dumps(triggered_rules)),
        )
        conn.commit()
        alert_id = int(cur.lastrowid)

    _append_audit("alert", {
        "alert_id": alert_id, "prediction_id": prediction_id, "timestamp": ts,
        "severity": severity, "status": status, "title": title, "message": message,
        "recommended_action": recommended_action, "confidence": confidence,
        "source": source, "triggered_rules": triggered_rules,
    })
    return alert_id


def log_auth_event(username: str, success: bool, reason: str = "") -> None:
    """Login attempts are logged too: who looked at the machine, and when."""
    _append_audit("auth", {"username": username, "success": success, "reason": reason})


def recent_alerts(limit: int = 50) -> list[dict[str, Any]]:
    rows = get_connection().execute(
        "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def recent_predictions(limit: int = 200) -> list[dict[str, Any]]:
    rows = get_connection().execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def summary_counts() -> dict[str, int]:
    """Status totals, used in the report header."""
    rows = get_connection().execute(
        "SELECT status, COUNT(*) AS n FROM predictions GROUP BY status"
    ).fetchall()
    return {r["status"]: r["n"] for r in rows}
