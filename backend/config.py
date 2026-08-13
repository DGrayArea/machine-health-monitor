"""
Backend configuration. Every setting can be overridden with an environment
variable, so the same code runs on a laptop and on a real deployment unchanged.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.getenv("MHM_MODEL_PATH", ROOT / "model" / "health_model.pkl"))
# Optional. RUL comes from the physics formula in backend/rul.py and this
# regressor is only a cross-check, so the API works without it.
RUL_MODEL_PATH = Path(os.getenv("MHM_RUL_MODEL_PATH", ROOT / "model" / "rul_model.pkl"))
FRONTEND_DIR = ROOT / "frontend"

# Everything the system generates lives under outputs/, split by kind, so that
# nothing generated is ever mixed in with source:
#   outputs/figures/  PNG plots from the evaluation script
#   outputs/metrics/  JSON metrics (cleaning report, evaluation scores)
#   outputs/logs/     runtime audit trail (SQLite DB + append-only JSONL)
#   outputs/exports/  reports downloaded from the dashboard (CSV / PDF)
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"
EXPORTS_DIR = OUTPUTS_DIR / "exports"

DB_PATH = Path(os.getenv("MHM_DB_PATH", LOGS_DIR / "monitoring.db"))
AUDIT_LOG_PATH = Path(os.getenv("MHM_AUDIT_LOG", LOGS_DIR / "audit_log.jsonl"))

# --- Auth ---------------------------------------------------------------
# In a real deployment MHM_SECRET_KEY must be set. If it is not, a random one is
# generated per process, so tokens stop working when the server restarts. That is
# inconvenient but much safer than shipping a signing key that anyone reading
# this repo could forge tokens with.
SECRET_KEY = os.getenv("MHM_SECRET_KEY") or secrets.token_urlsafe(32)
SECRET_KEY_IS_EPHEMERAL = "MHM_SECRET_KEY" not in os.environ
JWT_ALGORITHM = "HS256"
TOKEN_TTL_MINUTES = int(os.getenv("MHM_TOKEN_TTL_MIN", "480"))  # one shift

# Demo credentials for the local dashboard. Demo only: the whole user store is
# a dict in backend/auth.py. See the README for the caveats.
DEMO_USERNAME = os.getenv("MHM_DEMO_USER", "engineer")
DEMO_PASSWORD = os.getenv("MHM_DEMO_PASSWORD", "maintenance123")

# Login rate limiting. Without it, the 200k-iteration hash is the only thing
# slowing a password guesser down, and that is a CPU cost we pay, not them.
LOGIN_MAX_ATTEMPTS = int(os.getenv("MHM_LOGIN_MAX_ATTEMPTS", "8"))
LOGIN_WINDOW_SECONDS = float(os.getenv("MHM_LOGIN_WINDOW_SEC", "300"))

# --- Alerts -------------------------------------------------------------
# An unchanged condition is re-logged at most this often. The simulator ticks
# every 1.5 s, so without this a single cooling fault writes 40 identical rows a
# minute and the alert history becomes unreadable. See backend/alerts.py.
ALERT_REPEAT_SECONDS = float(os.getenv("MHM_ALERT_REPEAT_SEC", "60"))

# --- Simulator ----------------------------------------------------------
SIM_INTERVAL_SECONDS = float(os.getenv("MHM_SIM_INTERVAL", "1.5"))
SIM_BUFFER_SIZE = int(os.getenv("MHM_SIM_BUFFER", "240"))  # ~6 min of history
SIM_AUTOSTART = os.getenv("MHM_SIM_AUTOSTART", "1") == "1"

# How many recent readings a downloaded report covers.
REPORT_MAX_ROWS = int(os.getenv("MHM_REPORT_ROWS", "200"))
