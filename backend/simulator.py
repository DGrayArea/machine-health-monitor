"""
The live sensor simulator, so the system can be demoed without hardware.

Why not just random numbers
    Independent random values would never produce a believable fault, because
    real faults come from channels affecting each other. This simulator
    reproduces the couplings that exist on a real spindle.

    1. Constant-power drive
       The controller holds cutting power roughly constant, so
           P = torque * omega   =>   rpm = P / torque * 60/(2*pi)
       A heavier cut therefore slows the spindle down. That relationship is why
       `power` is a better feature than torque or rpm alone, and the simulator
       follows it rather than faking it.

    2. Wear raises torque
       A blunt tool needs more force for the same cut, so torque climbs with
       wear, which through the first coupling drags rpm down and pushes
       strain = wear x torque towards the overstrain limit. That is the
       degradation ramp that walks the dashboard from Normal to Warning to Fault
       without anyone touching it.

    3. Low rpm means worse cooling
       Less airflow over the spindle, so the process-to-air temperature
       difference shrinks, which is the heat-dissipation failure condition.
       Faults therefore arrive as a cascade the way they do on a real machine,
       rather than one isolated channel drifting out of range.

    4. Maintenance resets the cycle
       After a fault the tool is changed, wear goes back to zero, torque
       recovers and the machine returns to Normal, so a long demo shows
       repeated tool lives.

    The reason this matters: the model was trained on data with these physics in
    it. A simulator that ignored them would be testing the model on a
    distribution it has never seen.

Demo control
    inject(scenario) forces a specific failure mode on demand, so you can turn
    the dashboard red during a presentation instead of waiting for it.
"""

from __future__ import annotations

import math
import random
import threading
from collections import deque
from typing import Any, Callable

from backend import config, database, rul, trends
from backend.alerts import build_alert, reset_suppression, should_log
from backend.predictor import ModelNotAvailable, predict

# --- Nominal operating point (matches the AI4I dataset's centre of mass) ---
NOMINAL_AIR_TEMP_K = 298.5
NOMINAL_TEMP_DIFF_K = 10.0
NOMINAL_POWER_W = 6900.0
NOMINAL_TORQUE_NM = 40.0
WEAR_PER_STEP_MIN = 2.2          # tool minutes consumed per simulated tick
TOOL_CHANGE_WEAR_MIN = 235.0     # tool is swapped at this wear


class MachineSimulator:
    """One virtual milling machine, with a tool-life cycle."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.air_temp = NOMINAL_AIR_TEMP_K
        self.temp_diff = NOMINAL_TEMP_DIFF_K
        self.tool_wear = 0.0
        self.product_type = "M"
        self.target_power = NOMINAL_POWER_W
        self.state = "running"          # running | cooling_fault | overload | maintenance
        self.state_ticks = 0
        self.cycles_completed = 0

    # -- internal helpers -------------------------------------------------

    def _drift(self, value: float, target: float, rate: float, noise: float) -> float:
        """Ease a value towards a target, plus Gaussian sensor noise."""
        return value + (target - value) * rate + self.rng.gauss(0, noise)

    def _maybe_change_state(self) -> None:
        """Occasionally start or end a fault episode."""
        self.state_ticks += 1

        if self.state == "running":
            # Random onset of a cooling problem (clogged filter, fan fault).
            if self.rng.random() < 0.010:
                self.state, self.state_ticks = "cooling_fault", 0
            # Random heavy cut / wrong material -> power overload.
            elif self.rng.random() < 0.008:
                self.state, self.state_ticks = "overload", 0

        elif self.state in ("cooling_fault", "overload"):
            # Episodes end on their own, as an operator would notice and correct.
            if self.state_ticks > self.rng.randint(12, 30):
                self.state, self.state_ticks = "running", 0

        elif self.state == "maintenance":
            if self.state_ticks > 3:
                self.tool_wear = 0.0
                self.temp_diff = NOMINAL_TEMP_DIFF_K
                self.target_power = NOMINAL_POWER_W
                self.state, self.state_ticks = "running", 0
                self.cycles_completed += 1

    # -- the physics ------------------------------------------------------

    def step(self) -> dict[str, Any]:
        """Advance the machine one tick and return a sensor reading."""
        self._maybe_change_state()

        # Tool change once the tool is spent (this is the cycle reset).
        if self.tool_wear >= TOOL_CHANGE_WEAR_MIN and self.state != "maintenance":
            self.state, self.state_ticks = "maintenance", 0

        if self.state == "maintenance":
            # Spindle idles during a tool change: low torque, high free speed.
            torque = max(3.0, self.rng.gauss(6.0, 1.0))
            rot_speed = self.rng.gauss(2400, 60)
            self.air_temp = self._drift(self.air_temp, NOMINAL_AIR_TEMP_K, 0.10, 0.05)
            self.temp_diff = self._drift(self.temp_diff, NOMINAL_TEMP_DIFF_K, 0.15, 0.08)
        else:
            # --- ambient drifts slowly, the way a shop floor does ---
            self.air_temp = self._drift(self.air_temp, NOMINAL_AIR_TEMP_K, 0.02, 0.09)

            # --- the tool wears, and a blunt tool needs more torque ---
            #
            # Wear does not build up at a fixed rate. Taylor's tool life
            # equation (V*T^n = C) says life falls sharply as cutting load
            # rises, so the wear increment is scaled by how hard the machine is
            # working against its nominal duty. The exponent 1.5 is a
            # simplification of that, clipped so one extreme tick cannot destroy
            # the tool outright.
            #
            # This is what makes the clock-time RUL projection worth showing. A
            # machine running hard loses life faster than the clock, so the
            # measured wear rate rises and the projected deadline pulls in.
            severity = (self.target_power / NOMINAL_POWER_W) ** 1.5
            severity = max(0.5, min(3.0, severity))
            self.tool_wear += WEAR_PER_STEP_MIN * severity * self.rng.uniform(0.85, 1.15)
            wear_fraction = self.tool_wear / TOOL_CHANGE_WEAR_MIN
            wear_torque_penalty = 1.0 + 0.55 * wear_fraction ** 2

            if self.state == "overload":
                self.target_power = self._drift(self.target_power, 9600, 0.20, 60)
            else:
                self.target_power = self._drift(self.target_power, NOMINAL_POWER_W,
                                                0.12, 90)

            torque = max(
                5.0,
                NOMINAL_TORQUE_NM * wear_torque_penalty + self.rng.gauss(0, 2.2),
            )

            # --- the constant-power drive sets the speed ---
            omega = self.target_power / torque              # rad/s
            rot_speed = max(150.0, omega * 60.0 / (2 * math.pi) + self.rng.gauss(0, 25))

            # --- cooling depends on airflow, so on rpm ---
            if self.state == "cooling_fault":
                cooling_target = 7.6                        # below the 8.6 K limit
            else:
                # Faster spindle -> more forced convection -> larger delta-T.
                cooling_target = NOMINAL_TEMP_DIFF_K * (0.80 + 0.20 * min(rot_speed / 1500.0, 1.6))
            self.temp_diff = self._drift(self.temp_diff, cooling_target, 0.12, 0.10)

        return {
            "air_temp": round(self.air_temp, 2),
            "process_temp": round(self.air_temp + self.temp_diff, 2),
            "rot_speed": round(max(1.0, rot_speed), 1),
            "torque": round(max(0.0, torque), 2),
            "tool_wear": round(min(self.tool_wear, 500.0), 1),
            "product_type": self.product_type,
        }

    def inject(self, scenario: str) -> str:
        """Force a condition on demand, for live demos."""
        if scenario == "overheat":
            self.state, self.state_ticks = "cooling_fault", 0
            self.temp_diff = 8.0
            self.target_power = 4200.0     # low power -> low rpm -> poor cooling
            return "Cooling fault injected: delta-T driven below the 8.6 K limit."
        if scenario == "overload":
            self.state, self.state_ticks = "overload", 0
            self.target_power = 9800.0
            return "Power overload injected: spindle driven above the 9 kW limit."
        if scenario == "tool_wear":
            self.tool_wear = 205.0
            return "Tool wear jumped to 205 min — inside the tool-failure band."
        if scenario == "reset":
            self.reset()
            return "Machine reset to nominal operating conditions."
        raise ValueError(
            f"Unknown scenario {scenario!r}. "
            "Use one of: overheat, overload, tool_wear, reset."
        )


class SimulationRunner:
    """
    Runs the simulator on a background thread: step, predict, alert, log.

    A daemon thread rather than asyncio keeps this off the web server's event
    loop, so a slow prediction cannot stall an HTTP request. The buffer is a
    deque with a maxlen, which drops the oldest reading on its own, so memory
    stays flat however long the demo runs.
    """

    def __init__(self, interval: float | None = None, buffer_size: int | None = None):
        self.machine = MachineSimulator()
        self.interval = interval or config.SIM_INTERVAL_SECONDS
        self.buffer: deque[dict] = deque(maxlen=buffer_size or config.SIM_BUFFER_SIZE)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.on_reading: Callable[[dict], None] | None = None
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def nominal_wear_rate(self) -> float:
        """
        Wear-minutes per wall-clock minute at nominal duty.

        This is the simulator's time-compression factor. One tick of `interval`
        seconds advances the tool by WEAR_PER_STEP_MIN cutting minutes, so
        dividing the measured rate by this gives a real-machine figure where 1.0
        means wearing at exactly the rate of the clock.
        """
        return WEAR_PER_STEP_MIN / (self.interval / 60.0)

    def tick(self) -> dict | None:
        """
        One full cycle. Tests call this directly, so the pipeline can be checked
        without starting a thread or waiting on the clock.
        """
        reading = self.machine.step()
        try:
            result = predict(reading)
        except ModelNotAvailable as exc:
            self.last_error = str(exc)
            return None

        # Trend rules need history, so they are evaluated here where the buffer
        # lives and handed to build_alert. A single reading cannot support them.
        with self._lock:
            recent = list(self.buffer)
        trend_hits = trends.detect(recent)

        effective_status, alert = build_alert(
            model_status=result["status"],
            confidence=result["confidence"],
            features=result["features"],
            product_type=reading["product_type"],
            extra_hits=trend_hits,
        )

        # Layer 3 of the RUL estimate. The physics gives remaining cutting
        # minutes, but an operator schedules in clock time, so we measure the
        # wear rate actually seen over the recent buffer rather than assuming
        # one. A machine running hard then reports a nearer deadline.
        remaining_life = dict(result["rul"])
        with self._lock:
            history = list(self.buffer)

        wear_rate = rul.estimate_wear_rate(history)
        normalised = rul.normalise_wear_rate(wear_rate, self.nominal_wear_rate)
        remaining_life["wear_rate_per_min"] = (
            round(normalised, 2) if normalised is not None else None
        )
        remaining_life["wallclock_remaining_min"] = (
            round(value, 1)
            if (value := rul.project_wallclock(
                remaining_life["remaining_min"], wear_rate,
                self.nominal_wear_rate)) is not None
            else None
        )

        timestamp = database.utc_now()
        prediction_id = database.log_prediction(
            reading=reading, derived=result["derived"],
            status=effective_status, confidence=result["confidence"],
            probabilities=result["probabilities"], source="simulator",
            timestamp=timestamp, remaining_life=remaining_life,
        )

        # Pop the suppression key before the alert goes anywhere else: it is an
        # internal detail, not part of the record or the API response.
        signature = alert.pop("_signature", None) if alert else None
        if alert is not None and (signature is None or should_log(signature)):
            database.log_alert(
                prediction_id=prediction_id, severity=alert["severity"],
                status=effective_status, title=alert["title"],
                message=alert["message"],
                recommended_action=alert["recommended_action"],
                confidence=result["confidence"], source="simulator",
                triggered_rules=alert["triggered_rules"], timestamp=timestamp,
            )

        record = {
            "timestamp": timestamp,
            "status": effective_status,
            "model_status": result["status"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "reading": reading,
            "derived": result["derived"],
            "alert": alert,
            "remaining_life": remaining_life,
            "prediction_id": prediction_id,
            "machine_state": self.machine.state,
        }
        with self._lock:
            self.buffer.append(record)
        if self.on_reading:
            self.on_reading(record)
        return record

    def _loop(self) -> None:
        # Event.wait() rather than sleep() so stop() takes effect straight away
        # instead of after the current interval finishes.
        while not self._stop.wait(self.interval):
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - one bad tick must not kill the thread
                self.last_error = f"{type(exc).__name__}: {exc}"

    def start(self) -> bool:
        if self.running:
            return False
        # A restart is a fresh run, so nothing should be suppressed because of
        # what the previous run happened to be alerting about.
        reset_suppression()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="sensor-simulator")
        self._thread.start()
        return True

    def stop(self) -> bool:
        if not self.running:
            return False
        self._stop.set()
        self._thread.join(timeout=self.interval + 2)
        self._thread = None
        return True

    def snapshot(self, limit: int | None = None) -> list[dict]:
        with self._lock:
            data = list(self.buffer)
        return data[-limit:] if limit else data


# Module-level instance shared by the API.
runner = SimulationRunner()
