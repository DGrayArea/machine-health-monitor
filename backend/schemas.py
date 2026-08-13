"""
Request/response shapes.

Pydantic validates every incoming payload before it reaches our code, so a
malformed or out-of-range sensor reading is rejected with a clear 422 instead of
producing a confident nonsense prediction. The `ge`/`le` bounds mirror
backend/thresholds.PLAUSIBLE — physically impossible values are a sensor fault.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HealthStatus = Literal["Normal", "Warning", "Fault"]
Severity = Literal["Info", "Warning", "Critical"]


class SensorReading(BaseModel):
    """
    One snapshot from the machine's sensors.

    UNITS: this API speaks SI — temperatures in **kelvin**, power in watts —
    because that is what the model was trained on and what the failure
    thresholds are quoted in. The dashboard and the downloaded reports convert
    to °C and kW for display. See backend/units.py for why the conversion lives
    at the edge rather than in here. To send 25 °C, post 298.15.
    """

    air_temp: float = Field(..., ge=250, le=350,
                            description="Ambient air temperature [K] (25 °C = 298.15)")
    process_temp: float = Field(..., ge=250, le=360,
                                description="Process temperature [K]")
    rot_speed: float = Field(..., ge=1, le=5000, description="Spindle speed [rpm]")
    torque: float = Field(..., ge=0, le=200, description="Spindle torque [N·m]")
    tool_wear: float = Field(..., ge=0, le=500, description="Cumulative tool wear [min]")
    product_type: Literal["L", "M", "H"] = Field(
        "M", description="Product quality tier — sets the overstrain limit"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "air_temp": 298.1,
                "process_temp": 308.6,
                "rot_speed": 1551,
                "torque": 42.8,
                "tool_wear": 0,
                "product_type": "M",
            }]
        }
    }


class RuleHitOut(BaseModel):
    rule_id: str
    severity: str
    title: str
    detail: str
    action: str


class Alert(BaseModel):
    severity: Severity
    title: str
    message: str
    recommended_action: str
    triggered_rules: list[RuleHitOut]


class RemainingLife(BaseModel):
    """Remaining useful life. See backend/rul.py for the derivation."""

    remaining_min: float = Field(..., description="Cutting minutes to the first limit")
    binding_constraint: Literal["tool_wear", "overstrain"]
    total_usable_min: float | None = Field(
        None, description="Usable tool life at this operating point; null if unbounded"
    )
    fraction_consumed: float = Field(..., ge=0, le=1)
    wear_limited_min: float
    strain_limited_min: float | None
    band: Literal["ok", "warning", "critical"]
    source: str = "physics"
    # Cross-check from the optional regressor; null when the model is absent.
    model_remaining_min: float | None = None
    model_sigma_min: float | None = None
    # Wall-clock projection from the observed wear rate; null until enough
    # live readings exist to measure a rate.
    wear_rate_per_min: float | None = None
    wallclock_remaining_min: float | None = None

    # `model_` is Pydantic's own namespace; opt out so our field names survive.
    model_config = {"protected_namespaces": ()}


class PredictionResponse(BaseModel):
    timestamp: str
    status: HealthStatus
    confidence: float = Field(..., description="Probability of the predicted class, 0-1")
    probabilities: dict[str, float]
    reading: SensorReading
    derived: dict[str, float] = Field(
        ..., description="Engineered features: temp_diff [K], power [W], strain [min*Nm]"
    )
    alert: Alert | None = None
    remaining_life: RemainingLife | None = None
    prediction_id: int | None = None


class LiveSnapshot(BaseModel):
    running: bool
    interval_seconds: float
    machine_state: str
    readings: list[PredictionResponse]


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    username: str


class AlertRecord(BaseModel):
    id: int
    timestamp: str
    severity: str
    status: str
    title: str
    message: str
    recommended_action: str
    confidence: float
    source: str
    # Which physical rules tripped. The dashboard shows the first one's `detail`
    # so the operator sees the measured value against the limit, not just a label.
    triggered_rules: list[RuleHitOut] = []
