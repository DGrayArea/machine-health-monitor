"""
Step 7 — The live sensor simulator, so the system can be demoed with no hardware.

WHY NOT JUST RANDOM NUMBERS
    Independent random values would never produce a realistic fault, because
    real machine faults come from *coupling* between channels. This simulator
    reproduces the couplings that actually exist on a spindle:

    1. CONSTANT-POWER DRIVE
         The controller holds cutting power roughly constant, so
             P = torque * omega   =>   rpm = P / torque * 60/(2*pi)
         A heavier cut (more torque) therefore SLOWS the spindle. That single
         relationship is why `power` is a better feature than torque or rpm
         alone, and the simulator obeys it rather than faking it.

    2. TOOL WEAR RAISES TORQUE
         A blunt tool needs more force for the same cut. Torque climbs with
         wear, which (via #1) drags rpm down and pushes strain = wear x torque
         toward the overstrain limit. This is the degradation ramp that makes
         the dashboard walk Normal -> Warning -> Fault on its own.

    3. LOW RPM MEANS WORSE COOLING
         Less airflow over the spindle, so the process-to-air temperature
         difference shrinks — which is exactly the heat-dissipation failure
         condition. Faults therefore arrive as a *cascade*, the way they do on
         a real machine, instead of one isolated channel going out of range.

    4. MAINTENANCE RESETS THE CYCLE
         After a fault the tool is changed: wear -> 0, torque recovers, the
         machine returns to Normal. So a long demo shows repeated life cycles.

    The point of all this: the model was trained on data with these physics in
    it, so the simulator must have them too. A simulator that violated them
    would be testing the model on a distribution it never saw.

DEMO CONTROL
    `inject(scenario)` forces a specific failure mode on demand, so you can
    trigger a red dashboard during a presentation instead of waiting for one.
"""

from __future__ import annotations

import math
import random
import threading
from collections import deque
from typing import Any, Callable

from backend import config, database, rul
from backend.alerts import build_alert
from backend.predictor import ModelNotAvailable, predict

# --- Nominal operating point (matches the AI4I dataset's centre of mass) ---
NOMINAL_AIR_TEMP_K = 298.5
NOMINAL_TEMP_DIFF_K = 10.0
NOMINAL_POWER_W = 6900.0
NOMINAL_TORQUE_NM = 40.0
WEAR_PER_STEP_MIN = 2.2          # tool minutes consumed per simulated tick
TOOL_CHANGE_WEAR_MIN = 235.0     # tool is swapped at this wear


class MachineSimulator:
    """A single virtual milling machine with a tool-life cycle."""

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
        """Pull a value gently toward a target, plus Gaussian sensor noise."""
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
            # Episodes are self-limiting: an operator notices and corrects.
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
            # --- ambient drifts slowly, like a real shop floor ---
            self.air_temp = self._drift(self.air_temp, NOMINAL_AIR_TEMP_K, 0.02, 0.09)

            # --- tool wears; a blunt tool demands more torque (coupling #2) ---
            #
            # Wear does NOT accrue at a fixed rate. Taylor's tool life equation
            # (V·T^n = C) says life falls sharply as cutting load rises, so we
            # scale the wear increment by how hard the machine is working
            # relative to its nominal duty. The exponent 1.5 is a simplification
            # of that relationship, clipped so a single extreme tick cannot
            # destroy the tool instantly.
            #
            # This is what makes the wall-clock RUL projection meaningful: a
            # machine running hard burns its life faster than the clock, so the
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

            # --- constant-power drive sets the speed (coupling #1) ---
            omega = self.target_power / torque              # rad/s
            rot_speed = max(150.0, omega * 60.0 / (2 * math.pi) + self.rng.gauss(0, 25))

            # --- cooling depends on airflow, i.e. on rpm (coupling #3) ---
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
        """Force a condition on demand — used for live demos."""
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
    Runs the simulator on a background thread: step -> predict -> alert -> log.

    A daemon thread (not asyncio) keeps this independent of the web server's
    event loop, so a slow prediction can never stall an HTTP request. The ring
    buffer is a `deque(maxlen=N)`, which discards the oldest reading
    automatically — memory stays flat no matter how long the demo runs.
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
        Wear-minutes accrued per wall-clock minute at nominal duty.

        This is the simulator's time-compression factor: one tick of
        `interval` seconds advances the tool by WEAR_PER_STEP_MIN cutting
        minutes. Dividing the measured rate by this recovers a real-machine
        figure where 1.0 means "wearing at exactly the rate of the clock".
        """
        return WEAR_PER_STEP_MIN / (self.interval / 60.0)

    def tick(self) -> dict | None:
        """
        One full cycle. Also called directly by tests, so the pipeline can be
        verified without starting a thread or waiting on wall-clock time.
        """
        reading = self.machine.step()
        try:
            result = predict(reading)
        except ModelNotAvailable as exc:
            self.last_error = str(exc)
            return None

        effective_status, alert = build_alert(
            model_status=result["status"],
            confidence=result["confidence"],
            features=result["features"],
            product_type=reading["product_type"],
        )

        # Layer 3 of the RUL estimate: the physics gives remaining *cutting*
        # minutes, but an operator schedules in wall-clock time. We measure the
        # wear rate actually observed over the recent buffer rather than
        # assuming one, so a machine running hard reports a nearer deadline.
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

        if alert is not None:
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
        # Event.wait() instead of sleep() so stop() takes effect immediately
        # rather than after the current interval finishes.
        while not self._stop.wait(self.interval):
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - a bad tick must not kill the thread
                self.last_error = f"{type(exc).__name__}: {exc}"

    def start(self) -> bool:
        if self.running:
            return False
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
