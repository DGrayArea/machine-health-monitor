"""
Tests for the backend API.

These cover the three things that would quietly break the whole system:
  1. Login actually protects the endpoints, not just the login page.
  2. /api/predict returns a valid status and confidence for real readings.
  3. Every prediction and alert reaches the audit trail.
"""

from __future__ import annotations

import pytest

from backend import config, database

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ------------------------------------------------------------------ health

def test_health_is_public_and_reports_model_state(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in ("ok", "degraded")
    assert isinstance(body["model_loaded"], bool)


# -------------------------------------------------------------------- auth

def test_login_succeeds_with_demo_credentials(client):
    res = client.post("/api/auth/login", json={
        "username": config.DEMO_USERNAME, "password": config.DEMO_PASSWORD,
    })
    assert res.status_code == 200
    assert res.json()["access_token"]
    assert res.json()["username"] == config.DEMO_USERNAME


@pytest.mark.parametrize("payload", [
    {"username": "engineer", "password": "wrong"},
    {"username": "nobody", "password": "maintenance123"},
])
def test_login_rejects_bad_credentials(client, payload):
    res = client.post("/api/auth/login", json=payload)
    assert res.status_code == 401
    # The message must not reveal which half was wrong.
    assert "username or password" in res.json()["detail"].lower()


@pytest.mark.parametrize("method,path", [
    ("get", "/api/live"),
    ("get", "/api/alerts"),
    ("get", "/api/predictions"),
    ("get", "/api/report/csv"),
    ("get", "/api/report/pdf"),
    ("post", "/api/simulator/start"),
])
def test_protected_endpoints_reject_anonymous_access(client, method, path):
    assert getattr(client, method)(path).status_code == 401


def test_predict_rejects_anonymous_access(client, normal_reading):
    assert client.post("/api/predict", json=normal_reading).status_code == 401


def test_garbage_token_is_rejected(client):
    res = client.get("/api/live", headers={"Authorization": "Bearer not.a.jwt"})
    assert res.status_code == 401


def test_token_signed_with_the_wrong_key_is_rejected(client):
    """The point of signing: a self-made token must not be accepted."""
    import jwt
    forged = jwt.encode({"sub": config.DEMO_USERNAME, "exp": 9999999999},
                        "attacker-key", algorithm="HS256")
    res = client.get("/api/live", headers={"Authorization": f"Bearer {forged}"})
    assert res.status_code == 401


# ----------------------------------------------------------------- predict

def test_predict_returns_a_valid_status_and_confidence(
    client, auth_headers, normal_reading, model_available
):
    if not model_available:
        pytest.skip("no trained model — run scripts/train_model.py")

    res = client.post("/api/predict", json=normal_reading, headers=auth_headers)
    assert res.status_code == 200
    body = res.json()

    assert body["status"] in ("Normal", "Warning", "Fault")
    assert 0.0 <= body["confidence"] <= 1.0
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert set(body["probabilities"]) == {"Normal", "Warning", "Fault"}
    assert body["prediction_id"] is not None


def test_predict_computes_derived_features(
    client, auth_headers, normal_reading, model_available
):
    if not model_available:
        pytest.skip("no trained model")

    body = client.post("/api/predict", json=normal_reading,
                       headers=auth_headers).json()
    derived = body["derived"]
    assert derived["temp_diff"] == pytest.approx(10.5, abs=0.01)
    assert derived["power"] == pytest.approx(6951, rel=0.02)
    assert derived["strain"] == pytest.approx(0.0)


def test_healthy_reading_produces_no_alert(
    client, auth_headers, normal_reading, model_available
):
    if not model_available:
        pytest.skip("no trained model")
    body = client.post("/api/predict", json=normal_reading,
                       headers=auth_headers).json()
    assert body["status"] == "Normal"
    assert body["alert"] is None


def test_overloaded_reading_produces_a_critical_alert(
    client, auth_headers, model_available
):
    if not model_available:
        pytest.skip("no trained model")

    # About 10.5 kW, well past the 9 kW power limit.
    reading = {"air_temp": 300.0, "process_temp": 311.0, "rot_speed": 1400,
               "torque": 72.0, "tool_wear": 30, "product_type": "M"}
    body = client.post("/api/predict", json=reading, headers=auth_headers).json()

    assert body["status"] == "Fault"
    assert body["alert"]["severity"] == "Critical"
    assert body["alert"]["recommended_action"]
    assert any(r["rule_id"] == "power_high" for r in body["alert"]["triggered_rules"])


@pytest.mark.parametrize("field,value", [
    ("rot_speed", 0),          # a stopped spindle means a sensor fault
    ("rot_speed", 99999),
    ("air_temp", 100),         # below any plausible ambient
    ("torque", -5),
    ("tool_wear", -1),
    ("product_type", "X"),     # not a real quality tier
])
def test_implausible_readings_are_rejected_with_422(
    client, auth_headers, normal_reading, field, value
):
    payload = dict(normal_reading, **{field: value})
    res = client.post("/api/predict", json=payload, headers=auth_headers)
    assert res.status_code == 422


def test_missing_field_is_rejected(client, auth_headers, normal_reading):
    payload = dict(normal_reading)
    del payload["torque"]
    assert client.post("/api/predict", json=payload,
                       headers=auth_headers).status_code == 422


# ------------------------------------------------------------- audit trail

def test_prediction_is_written_to_the_audit_trail(
    client, auth_headers, normal_reading, model_available
):
    if not model_available:
        pytest.skip("no trained model")

    assert database.recent_predictions() == []
    client.post("/api/predict", json=normal_reading, headers=auth_headers)

    rows = database.recent_predictions()
    assert len(rows) == 1
    assert rows[0]["source"] == "api"
    assert rows[0]["username"] == config.DEMO_USERNAME
    assert rows[0]["status"] == "Normal"

    # And to the append-only flat log as well.
    assert config.AUDIT_LOG_PATH.exists()
    assert '"event": "prediction"' in config.AUDIT_LOG_PATH.read_text()


def test_alert_is_written_to_the_audit_trail(client, auth_headers, model_available):
    if not model_available:
        pytest.skip("no trained model")

    reading = {"air_temp": 300.0, "process_temp": 311.0, "rot_speed": 1400,
               "torque": 72.0, "tool_wear": 30, "product_type": "M"}
    client.post("/api/predict", json=reading, headers=auth_headers)

    alerts = database.recent_alerts()
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "Critical"
    assert alerts[0]["recommended_action"]

    listed = client.get("/api/alerts", headers=auth_headers).json()
    assert len(listed) == 1
    assert listed[0]["severity"] == "Critical"

    # The dashboard's Detail column reads triggered_rules[0].detail, so if the
    # API drops that field the column quietly shows a dash on every row.
    rules = listed[0]["triggered_rules"]
    assert rules, "alert history must carry the rules that tripped"
    assert rules[0]["detail"]
    # Operator-facing text uses display units, kW, not the raw SI watts.
    assert "9.0 kW limit" in rules[0]["detail"]


def test_login_attempts_are_audited(client):
    client.post("/api/auth/login", json={"username": "engineer",
                                         "password": "wrong"})
    log = config.AUDIT_LOG_PATH.read_text()
    assert '"event": "auth"' in log
    assert '"success": false' in log


# --------------------------------------------------------------- simulator

def test_simulator_tick_produces_and_logs_a_reading(model_available):
    if not model_available:
        pytest.skip("no trained model")

    from backend.simulator import SimulationRunner
    runner = SimulationRunner(interval=0.01, buffer_size=10)

    record = runner.tick()
    assert record is not None
    assert record["status"] in ("Normal", "Warning", "Fault")
    assert record["reading"]["rot_speed"] > 0
    assert len(runner.buffer) == 1
    assert len(database.recent_predictions()) == 1


def test_simulator_buffer_never_grows_without_bound(model_available):
    if not model_available:
        pytest.skip("no trained model")

    from backend.simulator import SimulationRunner
    runner = SimulationRunner(interval=0.01, buffer_size=5)
    for _ in range(12):
        runner.tick()
    assert len(runner.buffer) == 5


def test_injected_overload_drives_the_machine_out_of_limits(model_available):
    if not model_available:
        pytest.skip("no trained model")

    from backend.simulator import SimulationRunner
    runner = SimulationRunner(interval=0.01, buffer_size=60)
    runner.machine.inject("overload")

    statuses = [runner.tick()["status"] for _ in range(25)]
    assert "Fault" in statuses, f"overload never tripped a fault: {set(statuses)}"


def test_tool_wear_cycle_resets_after_maintenance(model_available):
    if not model_available:
        pytest.skip("no trained model")

    from backend.simulator import SimulationRunner
    runner = SimulationRunner(interval=0.01, buffer_size=300)
    runner.machine.inject("tool_wear")          # jump to 205 min

    wears = [runner.tick()["reading"]["tool_wear"] for _ in range(40)]
    assert max(wears) > 200
    assert wears[-1] < max(wears), "the tool was never changed"


def test_unknown_injection_scenario_is_rejected(client, auth_headers):
    res = client.post("/api/simulator/inject/explode", headers=auth_headers)
    assert res.status_code == 400
    assert "unknown scenario" in res.json()["detail"].lower()


# ----------------------------------------------------------------- reports

def test_csv_report_downloads_with_a_header_row(
    client, auth_headers, normal_reading, model_available
):
    if not model_available:
        pytest.skip("no trained model")
    client.post("/api/predict", json=normal_reading, headers=auth_headers)

    res = client.get("/api/report/csv", headers=auth_headers)
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert "attachment; filename=" in res.headers["content-disposition"]

    lines = res.text.strip().splitlines()
    assert lines[0].startswith("timestamp,source,status,confidence")
    # Column names carry their unit, so the file says what it contains.
    assert "air_temp_c" in lines[0] and "power_kw" in lines[0]
    assert len(lines) == 2

    # The values must actually be converted, not just relabelled.
    row = dict(zip(lines[0].split(","), lines[1].split(",")))
    assert float(row["air_temp_c"]) == pytest.approx(298.1 - 273.15, abs=0.01)
    assert float(row["power_kw"]) == pytest.approx(6.95, abs=0.01)
    # A temperature difference converts one to one, with no offset subtracted.
    assert float(row["temp_diff_c"]) == pytest.approx(10.5, abs=0.01)


def test_pdf_report_downloads_as_a_valid_pdf(
    client, auth_headers, normal_reading, model_available
):
    if not model_available:
        pytest.skip("no trained model")
    client.post("/api/predict", json=normal_reading, headers=auth_headers)

    res = client.get("/api/report/pdf", headers=auth_headers)
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert res.content.startswith(b"%PDF-")     # the PDF magic number
    assert len(res.content) > 1000
