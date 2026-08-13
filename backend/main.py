"""
The API.

    GET  /                          the dashboard (static HTML/JS)
    GET  /api/health                liveness + model status        [public]
    POST /api/auth/login            username/password -> JWT       [public]
    GET  /api/auth/me               who am I                       [auth]
    GET  /api/model/info            what model is loaded           [auth]
    POST /api/predict               sensor reading -> prediction    [auth]
    GET  /api/live                  recent simulated readings      [auth]
    GET  /api/alerts                alert history                  [auth]
    GET  /api/predictions           prediction history             [auth]
    POST /api/simulator/start|stop  control the simulator          [auth]
    POST /api/simulator/inject/{s}  force a fault, for demos       [auth]
    GET  /api/report/csv|pdf        download a report              [auth]

    Interactive docs are at /docs. FastAPI generates them from the Pydantic
    models, so they stay in step with the code.

Run it:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend import auth, config, database, predictor, reporting
from backend.alerts import build_alert
from backend.schemas import (
    AlertRecord,
    LiveSnapshot,
    LoginRequest,
    PredictionResponse,
    SensorReading,
    TokenResponse,
)
from backend.simulator import runner


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown. Anything expensive happens once, here."""
    for directory in (config.LOGS_DIR, config.EXPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    database.init_db()

    if predictor.is_ready():
        info = predictor.metadata()
        print(f"[model] loaded {info['model_type']} "
              f"({info['n_estimators']} trees), trained {info['trained_at']}")
        if config.SIM_AUTOSTART:
            runner.start()
            print(f"[sim  ] started, tick = {runner.interval}s")
    else:
        print("[model] NOT FOUND — /api/predict will return 503.\n"
              "        Run: python scripts/clean_data.py && python scripts/train_model.py")

    if config.SECRET_KEY_IS_EPHEMERAL:
        print("[auth ] MHM_SECRET_KEY not set — using a random per-process key. "
              "Tokens will be invalidated on restart.")

    yield
    runner.stop()


app = FastAPI(
    title="Machine Health Monitoring API",
    description="Predictive maintenance for a milling machine — "
                "Random Forest classifier + physical threshold rules.",
    version="1.0.0",
    lifespan=lifespan,
)

# The dashboard is served from the same origin, so CORS is not strictly needed.
# It is enabled for localhost only, so a separate frontend dev server (Vite on
# :5173, say) can talk to this API without any changes.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Public
# --------------------------------------------------------------------------

@app.get("/api/health", tags=["system"])
def health() -> dict:
    ready = predictor.is_ready()
    return {
        "status": "ok" if ready else "degraded",
        "model_loaded": ready,
        # RUL always works, since it is physics. This is the optional model.
        "rul_model_loaded": predictor.rul_model_available(),
        "simulator_running": runner.running,
        "readings_buffered": len(runner.buffer),
        "last_simulator_error": runner.last_error,
    }


@app.post("/api/auth/login", response_model=TokenResponse, tags=["auth"])
def login(payload: LoginRequest) -> TokenResponse:
    user = auth.authenticate(payload.username, payload.password)
    if user is None:
        # Deliberately vague, so it never reveals whether the username exists.
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Incorrect username or password."
        )
    token, ttl = auth.create_token(user["username"])
    return TokenResponse(access_token=token, expires_in_seconds=ttl,
                         username=user["username"])


# --------------------------------------------------------------------------
# Authenticated
# --------------------------------------------------------------------------

@app.get("/api/auth/me", tags=["auth"])
def me(user: dict = Depends(auth.current_user)) -> dict:
    return {"username": user["username"], "role": user["role"]}


@app.get("/api/model/info", tags=["model"])
def model_info(user: dict = Depends(auth.current_user)) -> dict:
    try:
        return predictor.metadata()
    except predictor.ModelNotAvailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))


@app.post("/api/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_endpoint(
    reading: SensorReading,
    user: dict = Depends(auth.current_user),
) -> PredictionResponse:
    """
    Classify one sensor reading.

    The path through: validate, derive features, run the model, apply the
    threshold rules, combine them, write to the audit trail, respond.
    """
    try:
        result = predictor.predict(reading.model_dump())
    except predictor.ModelNotAvailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))

    effective_status, alert = build_alert(
        model_status=result["status"],
        confidence=result["confidence"],
        features=result["features"],
        product_type=reading.product_type,
    )

    timestamp = database.utc_now()
    prediction_id = database.log_prediction(
        reading=reading.model_dump(), derived=result["derived"],
        status=effective_status, confidence=result["confidence"],
        probabilities=result["probabilities"], source="api",
        username=user["username"], timestamp=timestamp,
        remaining_life=result["rul"],
    )
    if alert is not None:
        database.log_alert(
            prediction_id=prediction_id, severity=alert["severity"],
            status=effective_status, title=alert["title"], message=alert["message"],
            recommended_action=alert["recommended_action"],
            confidence=result["confidence"], source="api",
            triggered_rules=alert["triggered_rules"], timestamp=timestamp,
        )

    return PredictionResponse(
        timestamp=timestamp, status=effective_status,
        confidence=result["confidence"], probabilities=result["probabilities"],
        reading=reading, derived=result["derived"], alert=alert,
        remaining_life=result["rul"], prediction_id=prediction_id,
    )


@app.get("/api/live", response_model=LiveSnapshot, tags=["simulator"])
def live(
    limit: int = Query(60, ge=1, le=500),
    user: dict = Depends(auth.current_user),
) -> LiveSnapshot:
    """The most recent simulated readings. This is what the dashboard polls."""
    return LiveSnapshot(
        running=runner.running,
        interval_seconds=runner.interval,
        machine_state=runner.machine.state,
        readings=runner.snapshot(limit),
    )


@app.get("/api/alerts", response_model=list[AlertRecord], tags=["alerts"])
def alerts(
    limit: int = Query(50, ge=1, le=500),
    severity: str | None = Query(None, pattern="^(Warning|Critical)$"),
    user: dict = Depends(auth.current_user),
) -> list[AlertRecord]:
    rows = database.recent_alerts(limit=limit if severity is None else 500)
    if severity:
        rows = [r for r in rows if r["severity"] == severity][:limit]
    # triggered_rules is stored as a JSON string, so decode before validating.
    rows = reporting.normalise_alert_rows(rows)
    return [AlertRecord(**{k: r[k] for k in AlertRecord.model_fields}) for r in rows]


@app.get("/api/predictions", tags=["prediction"])
def predictions(
    limit: int = Query(100, ge=1, le=1000),
    user: dict = Depends(auth.current_user),
) -> list[dict]:
    rows = database.recent_predictions(limit=limit)
    for r in rows:
        if isinstance(r.get("probabilities"), str):
            r["probabilities"] = json.loads(r["probabilities"])
    return rows


@app.post("/api/simulator/start", tags=["simulator"])
def sim_start(user: dict = Depends(auth.current_user)) -> dict:
    if not predictor.is_ready():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "No trained model — cannot run the simulator.")
    return {"started": runner.start(), "running": runner.running}


@app.post("/api/simulator/stop", tags=["simulator"])
def sim_stop(user: dict = Depends(auth.current_user)) -> dict:
    return {"stopped": runner.stop(), "running": runner.running}


@app.post("/api/simulator/inject/{scenario}", tags=["simulator"])
def sim_inject(scenario: str, user: dict = Depends(auth.current_user)) -> dict:
    """Force a fault so the dashboard can be shown going red on cue."""
    try:
        message = runner.machine.inject(scenario)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    return {"scenario": scenario, "message": message}


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

@app.get("/api/report/csv", tags=["reports"])
def report_csv(
    limit: int = Query(config.REPORT_MAX_ROWS, ge=1, le=5000),
    user: dict = Depends(auth.current_user),
) -> Response:
    name, payload = reporting.build_csv(database.recent_predictions(limit=limit))
    return Response(
        content=payload, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.get("/api/report/pdf", tags=["reports"])
def report_pdf(
    limit: int = Query(config.REPORT_MAX_ROWS, ge=1, le=2000),
    user: dict = Depends(auth.current_user),
) -> Response:
    name, payload = reporting.build_pdf(
        database.recent_predictions(limit=limit),
        reporting.normalise_alert_rows(database.recent_alerts(limit=50)),
        username=user["username"],
    )
    return Response(
        content=payload, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


# --------------------------------------------------------------------------
# Frontend, declared last so it cannot shadow an /api route
# --------------------------------------------------------------------------

# `no-cache` means revalidate before reusing, not never cache: the browser still
# gets a cheap 304 when the file has not changed. Without it, an edit to
# styles.css or app.js can sit hidden behind a cached copy, which looks exactly
# like a bug in the code.
STATIC_HEADERS = {"Cache-Control": "no-cache"}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(config.FRONTEND_DIR / "index.html", headers=STATIC_HEADERS)


@app.get("/app.js", include_in_schema=False)
def app_js() -> FileResponse:
    return FileResponse(config.FRONTEND_DIR / "app.js",
                        media_type="application/javascript", headers=STATIC_HEADERS)


@app.get("/styles.css", include_in_schema=False)
def styles_css() -> FileResponse:
    return FileResponse(config.FRONTEND_DIR / "styles.css",
                        media_type="text/css", headers=STATIC_HEADERS)
