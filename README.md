# Machine Health Monitoring / Predictive Maintenance

A full-stack predictive maintenance system for a milling machine: it reads sensor
data, classifies machine health as **Normal / Warning / Fault**, estimates the
**remaining useful life** of the cutting tool, raises alerts with a recommended
physical action, logs everything to an audit trail, and shows it all on a live
dashboard.

Built as a mechatronics engineering project. Every threshold in it comes from
documented machine physics, not from arbitrary numbers — the point is that you
can defend each decision, not just demo it.

---

## Table of contents

1. [Quick start](#quick-start)
2. [What it does end to end](#what-it-does-end-to-end)
3. [Units](#units)
4. [Project structure](#project-structure)
5. [Part 1 — Data](#part-1--data)
6. [Part 2 — Model](#part-2--model)
7. [Part 3 — Remaining Useful Life](#part-3--remaining-useful-life)
8. [Part 4 — Backend](#part-4--backend)
9. [Part 5 — Alerts](#part-5--alerts)
10. [Part 6 — Dashboard](#part-6--dashboard)
11. [Part 7 — Authentication](#part-7--authentication)
12. [Part 8 — Testing](#part-8--testing)
13. [API reference](#api-reference)
14. [Results](#results)
15. [Questions you should be ready to answer](#questions-you-should-be-ready-to-answer)
16. [Limitations](#limitations-be-honest-about-these)
17. [Troubleshooting](#troubleshooting)

---

## Quick start

Requires Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Then run the pipeline in order — each step depends on the previous one:

```bash
python data/download_data.py && python scripts/clean_data.py && python scripts/train_model.py && python scripts/evaluate_model.py && python scripts/train_rul_model.py
```

Start the server:

```bash
uvicorn backend.main:app --reload --port 8010
```

Open **http://127.0.0.1:8010** and sign in with `engineer` / `maintenance123`.

The simulator starts automatically, so readings begin appearing within a couple
of seconds. Use the **Inject fault** buttons to force a red dashboard on demand.

Run the tests:

```bash
python -m pytest tests/ -v
```

---

## What it does end to end

```
  data/raw/ai4i2020.csv          10,000 rows of real machine sensor data
          |
          |  scripts/clean_data.py      dedupe, impute, drop implausible,
          v                             derive features, apply threshold rules
  data/processed/machine_health.csv     labelled Normal / Warning / Fault
          |
          |  scripts/train_model.py     stratified 80/20, Random Forest
          v
  model/health_model.pkl                model + feature order + class order
          |
          v
  +------------------- backend (FastAPI) --------------------+
  |                                                          |
  |  simulator.py  --> generates a physically coupled reading |
  |        |                                                 |
  |        v                                                 |
  |  predictor.py  --> Random Forest: status + confidence     |
  |        |                                                 |
  |        v                                                 |
  |  rul.py        --> remaining useful life (physics + trend) |
  |        |                                                 |
  |        v                                                 |
  |  alerts.py     --> combine with physical threshold rules  |
  |        |           (rules can escalate, never suppress)   |
  |        v                                                 |
  |  database.py   --> SQLite + append-only JSONL audit log   |
  +----------------------------|-----------------------------+
                               v
              frontend/  live dashboard, alert history,
                         CSV / PDF report download
```

---

## Units

**Internally the system is strictly SI**: kelvin, watts, newton-metres. The
dataset is in kelvin, the model was trained on kelvin, and the failure
thresholds are quoted in kelvin. **Conversion to readable units happens only at
the presentation layer** — the dashboard, the PDF and the CSV.

| Quantity | Stored / API | Displayed |
|---|---|---|
| Air & process temperature | K | **°C** |
| Temperature difference (ΔT) | K | **°C** (a *difference* of 10 K **is** 10 °C) |
| Spindle power | W | **kW** |
| Torque | N·m | N·m |
| Rotational speed | rpm | rpm |
| Tool wear / RUL | min | min |
| Mechanical strain | min·N·m | min·N·m |

Why convert at the edge instead of just storing Celsius everywhere? Because
there is then exactly **one** place a unit can be wrong. If Celsius leaked into
the feature vector, the model would receive `25.4` where it expects `298.5` and
return a confident wrong answer with no error raised. Unit mismatches are silent
— nothing crashes — which is what makes them dangerous. See
[backend/units.py](backend/units.py); `tests/test_units.py` covers the
conversions, including the one that is easiest to get wrong: subtracting 273.15
from a temperature *difference*.

To POST 25 °C to the API, send `298.15`.

---

## Project structure

```
machine-health-monitor/
├── data/
│   ├── download_data.py        fetches the dataset (no API key needed)
│   ├── raw/ai4i2020.csv        original 10,000-row dataset
│   └── processed/              cleaned + labelled output
│
├── scripts/
│   ├── clean_data.py           cleaning + labelling + RUL target
│   ├── train_model.py          trains Random Forest (+ LogReg baseline)
│   ├── evaluate_model.py       confusion matrix, plots, per-class metrics
│   └── train_rul_model.py      RUL regressor + the negative result about it
│
├── model/
│   ├── health_model.pkl        trained classifier bundle
│   ├── metadata.json           scores, feature importance, training config
│   ├── rul_model.pkl           RUL regressor (optional cross-check)
│   └── rul_metadata.json       RUL scores and uncertainty
│
├── backend/
│   ├── thresholds.py           ** all physical limits, ONE definition **
│   ├── units.py                ** SI <-> display conversion, ONE place **
│   ├── rul.py                  remaining useful life (physics + wear trend)
│   ├── predictor.py            loads the models, runs inference
│   ├── alerts.py               model + rules -> severity + action
│   ├── simulator.py            physically coupled sensor simulator
│   ├── database.py             SQLite + append-only audit log
│   ├── auth.py                 PBKDF2 password hashing + JWT
│   ├── reporting.py            CSV and PDF report generation
│   ├── schemas.py              request/response validation
│   ├── config.py               all settings, env-overridable
│   └── main.py                 the API endpoints
│
├── frontend/
│   ├── index.html              login + dashboard markup
│   ├── styles.css              colour-coded status styling
│   └── app.js                  polling, rendering, canvas chart
│
├── tests/
│   ├── test_alerts.py          alert + threshold logic (no model needed)
│   ├── test_rul.py             RUL physics, wear-rate trend, RUL alert rule
│   ├── test_units.py           unit conversions
│   ├── test_api.py             endpoints, auth, audit trail, reports
│   └── test_thresholds.py      offline labeller == live alerter
│
└── outputs/                    ** everything the system generates **
    ├── figures/                PNG plots from evaluate_model.py
    ├── metrics/                JSON metrics (cleaning + evaluation)
    ├── logs/                   monitoring.db + audit_log.jsonl
    └── exports/                downloaded CSV / PDF reports
```

Nothing generated is ever written next to source code — it all lands in
`outputs/`, so you can delete that folder and rebuild it from scratch.

---

## Part 1 — Data

### The dataset

**AI4I 2020 Predictive Maintenance Dataset**, 10,000 rows. This is the same
table Kaggle publishes as *"Machine Predictive Maintenance Classification"* —
Kaggle mirrors it from the UCI Machine Learning Repository. `download_data.py`
pulls from UCI because that needs no API token.

| Column | Meaning |
|---|---|
| `Type` | product quality tier: L / M / H |
| `Air temperature [K]` | ambient temperature |
| `Process temperature [K]` | temperature at the cutting process |
| `Rotational speed [rpm]` | spindle speed |
| `Torque [Nm]` | spindle torque |
| `Tool wear [min]` | cumulative minutes of cutting on the current tool |
| `Machine failure` + `TWF/HDF/PWF/OSF/RNF` | ground-truth failure flags |

> **On "vibration and pressure":** this dataset does not contain those channels.
> Rather than fabricate them, the system uses the five real ones. The pipeline is
> channel-agnostic — adding a vibration sensor means adding one column in
> `thresholds.py` and retraining, and nothing else changes.

### Cleaning (`scripts/clean_data.py`)

1. **Rename** columns to snake_case.
2. **Deduplicate** — comparing sensor values only, ignoring `UDI` and
   `Product ID`. Those are row counters; two identical readings under different
   IDs are still the same measurement and would double-weight the model.
3. **Impute missing values** — numeric to the *median* (robust to the outliers we
   are specifically trying to detect, unlike the mean), categorical to the
   *mode*. We fill rather than drop, because a sensor dropout on one channel
   should not throw away the other four.
4. **Drop implausible readings** — negative torque, a stopped spindle, a
   temperature below any plausible ambient. These are **sensor** faults, not
   machine faults, and feeding them to the model teaches it nonsense.
5. **Derive three features** that the failure physics is actually written in:

   | Feature | Formula | Unit |
   |---|---|---|
   | `temp_diff` | `process_temp - air_temp` | K |
   | `power` | `torque × ω`, where `ω = rpm × 2π/60` | W |
   | `strain` | `tool_wear × torque` | min·Nm |

6. **Label** each row Normal / Warning / Fault.

### The labelling logic

This is the part you must be able to defend. The dataset was generated from five
documented physical failure modes, so we reuse *those exact thresholds* and
define a Warning band just before each one:

| Failure mode | **Fault** condition | **Warning** band (our margin) |
|---|---|---|
| Heat dissipation (HDF) | `ΔT < 8.6 K` **AND** `rpm < 1380` | `ΔT < 9.5 K` AND `rpm < 1500` |
| Power (PWF) | `power < 3500 W` **OR** `> 9000 W` | `power < 4000 W` OR `> 8500 W` |
| Overstrain (OSF) | `strain > 11000/12000/13000` (L/M/H) | `strain > 85%` of that limit |
| Tool wear (TWF) | `tool_wear` in 200–240 min | `tool_wear > 180 min` |
| Random (RNF) | 0.1% chance, unrelated to sensors | — |

Two details worth pointing out:

- **The heat-dissipation rule is an AND, not an OR.** Cooling depends on forced
  convection, so a small ΔT is only dangerous when the spindle is *also* turning
  slowly. A test in `test_alerts.py` pins this down specifically.
- **The overstrain limit depends on the quality tier.** The same
  `strain = 12100 min·Nm` is a fault on an L-grade tool, a fault on M, and fine
  on H.

**Precedence:** Fault beats Warning beats Normal. A row is a Fault if the
dataset's own `machine_failure` flag is set — we trust ground truth for the
positive class rather than re-deriving it, because RNF failures are not
predictable from the sensors at all.

**Result:** 10,000 rows → 66.96% Normal, 29.65% Warning, 3.39% Fault.

### The honest caveat about the Warning class

The Warning label is *deterministic* given the sensors, because we computed it
from them. A model will therefore learn the Warning boundary almost perfectly.

That is not cheating — it is **rule distillation**. The value is that the same
model also learns the **Fault** class, which is *not* a pure function of the
sensors: it contains a random component (RNF) and a stochastic tool-breaking
point somewhere in 200–240 min. **Fault recall is the number that measures real
learning**, which is why `evaluate_model.py` reports it separately instead of
hiding behind overall accuracy.

---

## Part 2 — Model

### Features

`type_code, air_temp, process_temp, rot_speed, torque, tool_wear, temp_diff, power, strain`

Handing the model `power` and `strain` directly is the single biggest accuracy
lever in the project. A decision tree splits on one variable at a time, so it
would need a deep, brittle staircase of splits to approximate `torque × rpm`.
Giving it the product turns that staircase into one clean split.

`type_code` is **ordinal** (L=0, M=1, H=2) rather than one-hot, because product
quality genuinely is ordered — the overstrain limit rises monotonically with it.

### Why Random Forest

`train_model.py` trains a Logistic Regression baseline *and* a Random Forest, and
prints both. The comparison is the argument, not decoration:

| Model | Accuracy | Macro F1 | **Fault F1** |
|---|---|---|---|
| Logistic Regression | 0.7170 | 0.5917 | **0.3630** |
| **Random Forest** | **0.9920** | **0.9657** | **0.9120** |

Logistic Regression draws one straight line per class. The failure rules are
conjunctions (`ΔT low AND rpm low`) and two-sided bands (`power too low OR too
high`) — shapes no single straight line can express. A Random Forest is a vote
over many axis-aligned trees, which is exactly the shape of a threshold rule.

### Handling the class imbalance

Fault is only 3.4% of rows. Two things address that:

- **Stratified split** — a plain random 80/20 could hand the test set an
  unrepresentative number of the 339 fault rows.
- **`class_weight="balanced"`** — makes each Fault row count roughly 20× a Normal
  row during training, so the model cannot score well by ignoring faults.

Without these, a model that predicted "Normal" forever would already score 67%
accuracy. **This is why accuracy alone proves nothing here.**

### What gets saved

`model/health_model.pkl` is a *bundle*, not a bare estimator:

```python
{"model": rf, "features": [...], "type_code": {...},
 "classes": [...], "trained_at": "..."}
```

The backend rebuilds each input row against the saved `features` list rather than
a hardcoded order. If you retrain with different features, the backend keeps
working instead of silently feeding `torque` into the column the model thinks is
`power` — which would give confident, completely wrong answers with no error.

---

## Part 3 — Remaining Useful Life

*"Is the machine healthy?"* is a classification question with three answers.
*"How long have I got?"* is a regression question with a continuous answer — and
a planner can schedule around "31 minutes" in a way they cannot around the word
"Warning".

### Start with the honest constraint

Classical data-driven prognostics (the NASA C-MAPSS style) needs **run-to-failure
trajectories**: many units, each logged from new until it dies. **AI4I 2020 does
not have that.** Its 10,000 rows are independent samples with no unit id and no
time ordering — you cannot follow one tool from new to worn. Inventing a "cycle"
column and training an LSTM on it would produce an impressive number that means
nothing.

So this does prognostics the other legitimate way: **model-based
(physics-of-failure)** rather than data-driven. Both are standard families in the
prognostics literature, and model-based is the correct choice when you have known
failure physics and no run-to-failure data.

### Layer 1 — the physics (authoritative)

Two constraints limit tool life, and **which one binds changes with load**:

| Constraint | Remaining cutting minutes |
|---|---|
| Tool wear | `200 - tool_wear` |
| Overstrain | `(osf_limit / torque) - tool_wear` |

RUL is the smaller of the two, floored at zero.

**This is why RUL is not just `200 - tool_wear`.** Rearranging the overstrain
condition `tool_wear × torque > limit` for the wear at which it trips gives a
*ceiling that depends on torque*:

| Tool wear | Torque | Ceiling | RUL | Binding |
|---|---|---|---|---|
| 150 min | 40 N·m | 200 min | **50 min** | tool wear |
| 150 min | 75 N·m | 160 min | **10 min** | overstrain |

Same tool, same wear, five times less life — because cutting harder does not just
consume the tool faster, **it lowers the ceiling**. No tool-wear threshold would
ever tell you this. In the cleaned dataset, 452 of 10,000 rows are
overstrain-bound: those are the ones a wear threshold alone would miss.

Deliberately excluded: cooling faults and power overloads. Those are
*instantaneous* failure conditions, not wear mechanisms — they do not consume
tool life, they end it. The alert rules already cover them.

### Layer 2 — the learned model, and why it lost

`scripts/train_rul_model.py` trains a `RandomForestRegressor` on the same 9
features. **We measured it and it does not beat the formula:**

| | MAE | RMSE | R² |
|---|---|---|---|
| Full model | **0.38 min** | 2.10 | 0.9989 |
| Without `tool_wear` | 49.64 min | 60.01 | 0.116 |

An R² of 0.999 is **not an achievement here** — the target is a deterministic
expression, so the forest is learning a division. `tool_wear` alone accounts for
99.7% of feature importance. The spread across its 300 trees is essentially zero,
because there is no noise in the target for them to disagree about:

| Region | median σ |
|---|---|
| All test rows | 0.00 min |
| Tool-wear bound | 0.00 min |
| **Overstrain bound** | **6.21 min** |

The one region where the trees genuinely disagree is the boundary where the
binding constraint switches — and there the model is visibly wrong. At 150 min /
75 N·m the physics says **10.0 min** and the model says **25.9 ± 15.1 min**. The
band is doing its job (it is wide, and it covers the truth), but the formula is
simply correct.

**So production uses the physics formula.** The regressor is reported as a
cross-check on the dashboard and nothing depends on it — the API works with
`rul_model.pkl` deleted. Being able to say that, with numbers, is worth more than
shipping it uncritically.

The second row of that table is the genuinely useful finding: strip out
`tool_wear` and the error explodes 131×. **The tool-wear counter is load-bearing
and has no redundancy** — if the encoder fails or the counter is reset wrongly,
RUL must fall back to the formula, not to this model. That is a real design
conclusion the exercise produced.

### Layer 3 — wall-clock projection

Layers 1 and 2 answer "how many minutes of *cutting* are left". An operator wants
"how long until I have to stop". So `estimate_wear_rate` fits a least-squares
slope to `tool_wear` over the last ~20 live readings — **measured**, not assumed.

Three details that matter:

- **It skips across tool changes.** Wear resets to zero on a tool change; fitting
  across the reset would give a negative slope and a nonsense deadline, so only
  the segment since the last reset is used.
- **The window is ~20 readings.** Fit the whole buffer and the rate lags badly —
  a machine that started cutting hard a minute ago still reports nominal, which
  is exactly when you want the warning.
- **It is normalised.** The simulator compresses time (a 1.5 s tick advances the
  tool ~2.2 cutting minutes so a full tool life is watchable in ~2 min), so its
  raw rate is ~88. Dividing by the nominal rate cancels the compression and
  leaves the figure an engineer wants: **"we are wearing this tool 1.6× faster
  than normal"**. The dashboard shows `1 h 59 min (wear 0.92× nominal)`.

The simulator's wear rate is load-dependent (a simplification of Taylor's tool
life equation `V·Tⁿ = C`), so working the machine hard genuinely pulls the
deadline in — the projection drops below the raw RUL under load and rises above
it when running gently.

### RUL as an alert

The RUL rule fires **only when overstrain is the binding constraint**. When tool
wear binds, the existing `tool_wear` threshold already says the same thing at
180 min, and showing an operator two rows for one problem is how alert lists stop
being trusted. The overstrain-bound case is the gap nothing else catches.

---

## Part 4 — Backend

FastAPI, chosen because Pydantic validates every payload before it reaches your
code and `/docs` gives you interactive API documentation for free.

| Module | Responsibility |
|---|---|
| `thresholds.py` | every physical limit — imported by *both* the offline cleaner and the live API |
| `predictor.py` | loads the model once at startup, runs inference |
| `alerts.py` | combines model verdict + threshold rules into a severity and an action |
| `simulator.py` | generates physically coupled fake readings |
| `database.py` | SQLite (queryable) + JSONL (append-only) audit trail |
| `auth.py` | PBKDF2 hashing, JWT issue/verify |
| `reporting.py` | CSV and PDF generation |

### One definition of the thresholds

`scripts/clean_data.py` imports its constants from `backend/thresholds.py`. If
the offline labeller used one set of thresholds and the live alerter used
another, the model would be *trained* to detect one thing and *deployed* to
explain a different thing — a silent, extremely hard-to-debug bug.
`tests/test_thresholds.py` runs both code paths over all 10,000 rows and asserts
they flag identical rows.

### The sensor simulator

Independent random numbers would never produce a realistic fault, because real
faults come from **coupling** between channels. The simulator reproduces three
couplings that exist on a real spindle:

1. **Constant-power drive** — `rpm = P / torque × 60/2π`. A heavier cut *slows*
   the spindle.
2. **Wear raises torque** — a blunt tool needs more force, which (via #1) drags
   rpm down and pushes `strain` toward the overstrain limit.
3. **Low rpm worsens cooling** — less airflow, so ΔT shrinks, which is the
   heat-dissipation failure condition.

So faults arrive as a **cascade**, the way they do on a real machine. You can see
this in the screenshot below: an injected cooling fault dropped rpm to 974, which
raised torque to 62 Nm, which pushed strain to 14413 min·Nm — over the 12000
limit — tripping a second, different failure mode.

A tool change resets wear to 0 and the cycle restarts, so a long demo shows
repeated tool-life cycles.

### Audit trail

Every prediction and alert goes to **two** sinks:

- **SQLite** (`outputs/logs/monitoring.db`) — queryable; the alert history and
  the downloadable reports are SQL queries against it.
- **JSONL** (`outputs/logs/audit_log.jsonl`) — append-only, one JSON object per
  line, never updated or deleted. If the database is tampered with, the flat log
  still shows what the system actually saw and said.

Login attempts are logged too. In maintenance work this is not optional: if the
machine breaks and the system said "Normal" five minutes earlier, you must be
able to prove exactly what readings it was given.

---

## Part 5 — Alerts

### The core design decision

An alert is **not** just "whatever the model said". It combines two independent
sources of evidence, and the combination rule is:

> **The rules can escalate the model. The model can never suppress the rules.**
> Effective severity = the worst of the two.

Why: a hard physical limit (power above 9 kW, tool past its life) is a *fact*,
not a prediction. If the model has a bad day and says "Normal" while the spindle
draws 9.5 kW, the operator must still be told. Machine safety interlocks work the
same way — a learned layer may add sensitivity, but it is never allowed to
override a hard limit.

The reverse direction is allowed and is the genuinely predictive part: if every
threshold is inside limits but the model recognises a combination that precedes
failures, it raises a Warning on its own. Thresholds alone cannot do that. You
can see this in the generated PDF as *"Model-detected warning condition"*.

### Severity ladder

| Model status | Severity | Meaning |
|---|---|---|
| Normal | — | no alert raised, nothing logged |
| Warning | `Warning` | plan a fix; the machine keeps running |
| Fault | `Critical` | act now; the machine should stop |

### Recommended actions

Every alert carries a concrete physical instruction, taken from the most urgent
rule that tripped, so it names the actual component:

| Condition | Recommended action |
|---|---|
| Heat dissipation failure | Stop the machine. Check coolant flow, clean the heat exchanger and verify the cooling fan is running before restart. |
| Cooling margin low | Inspect coolant level and airflow path. Raise spindle speed to improve forced convection if the process allows. |
| Power overload | Reduce load immediately — cut feed rate or depth of cut. Check for a blunt tool or incorrect material. |
| Power underrun | Check the drive coupling and belt tension — the spindle may be slipping or running unloaded. |
| Mechanical overstrain | Stop and replace the tool. Inspect the spindle bearing and tool holder for damage before resuming. |
| Approaching overstrain | Reduce load or change the tool early. Inspect the bearing at the next available stop. |
| Tool life exceeded | Replace the cutting tool now. Do not start another cycle. |
| Tool nearing end of life | Schedule a tool change within the next N minutes of cutting. |

If several rules trip at once, the message names the secondary ones too — so
nobody fixes one thing and declares victory. A test asserts that **no alert can
ever be raised without an actionable instruction**.

---

## Part 6 — Dashboard

Plain HTML/CSS/JS. No framework, no build step, no CDN — you can read every line
that puts a pixel on the screen, which matters when you have to explain it.

- **Colour-coded status** — green / amber / red maps directly onto
  Normal / Warning / Fault. The Fault dot pulses; the browser tab title changes
  to `(Fault) Machine Health Monitor` so you notice it in a background tab.
- **Eight live tiles** — the five raw channels plus the three derived ones, each
  with a fill bar and its own limit annotation. Tile colours come from the same
  thresholds the backend uses.
- **Recommended action card** — appears only when there is something to do,
  showing the tripped rule, the measured value vs the limit, and the action.
- **Trend chart** — hand-drawn on a `<canvas>`. Power, ΔT and tool wear are
  normalised into one plot area because they have wildly different units; this is
  a *shape* comparison, and the tiles carry exact values. Below it, a status
  strip colours one bar per reading, making the Normal → Warning → Fault
  progression readable at a glance.
- **Alert history** — time, severity, condition, measured value vs limit, action.
- **Report download** — CSV or PDF.
- **Simulator controls** — start/stop, plus inject buttons to force a cooling
  fault, a power overload, or a worn tool on cue for a live demo.

**Why polling, not WebSockets:** the simulator emits one reading every 1.5 s. At
that rate a WebSocket saves almost nothing but adds reconnect logic, heartbeats
and a second code path to debug. If a poll fails, the next one simply succeeds.

**Why downloads go through `fetch()`:** a plain `<a href>` cannot carry the
`Authorization` header, so the file is pulled into a Blob and a temporary
object-URL link is clicked instead.

---

## Part 7 — Authentication

Username + password login returning a signed JWT; every dashboard endpoint
requires `Authorization: Bearer <token>`.

**Password storage** — never plain text. Stored as
`pbkdf2_hmac('sha256', password, salt, 200_000)`:

- The **salt** means two users with the same password get different hashes, so
  they cannot all be cracked at once with one rainbow table.
- **200,000 iterations** make each guess deliberately slow. Plain SHA-256 can be
  guessed billions of times per second on a GPU; this brings that to thousands.
- `hmac.compare_digest` compares in constant time, so an attacker cannot learn
  the hash byte-by-byte from response timing.

**The token** — a JWT is `header.payload.signature`. The payload holds the
username and an expiry; the signature is HMAC-SHA256 over the first two parts
using the secret key. Anyone can *read* a JWT, so no secrets go in it — but
nobody can *forge* one without the key. A test forges a token with the wrong key
and asserts it is rejected.

**Two deliberate details:**

- Login failures return the same vague message whether the username or the
  password was wrong, and an unknown username still runs a dummy hash so the
  response time does not leak which usernames exist.
- If `MHM_SECRET_KEY` is not set, a random key is generated per process. Tokens
  then stop working on restart — inconvenient, but far safer than shipping a
  hardcoded signing key that anyone reading this repo could forge tokens with.

---

## Part 8 — Testing

```bash
python -m pytest tests/ -v
```

**82 tests, all passing.** Tests write to a temp database, never the real audit
trail.

| File | What it pins down |
|---|---|
| `test_alerts.py` | Each threshold rule independently; the AND in the cooling rule; the two-sided power envelope; the per-tier overstrain limit; **that rules escalate the model and the model can never suppress a rule**; that every alert carries an action. Loads no model — pure logic. |
| `test_api.py` | Auth protects every endpoint; forged tokens rejected; prediction shape and probability sum; implausible readings rejected with 422; predictions and alerts reach both audit sinks; simulator tick/buffer/injection; CSV has a header row; PDF starts with `%PDF-`. |
| `test_thresholds.py` | The offline pandas labeller and the live scalar alerter flag identical rows across all 10,000 — the drift guard. |
| `test_rul.py` | RUL is not `200 - wear`; high torque lowers the ceiling; the per-tier ceiling ordering; the wear-rate fit survives a tool change and refuses to guess when wear is flat; time-compression normalisation; the RUL rule does **not** double-alert with the tool-wear threshold but *does* fire for the overstrain-bound gap; the vectorised training target matches the scalar physics row for row. |
| `test_units.py` | Conversions round-trip, and a temperature *difference* has no offset applied. |

---

## API reference

Interactive docs at **http://127.0.0.1:8010/docs**.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/api/health` | — | liveness, model + simulator state |
| `POST` | `/api/auth/login` | — | username/password → JWT |
| `GET` | `/api/auth/me` | ✓ | current user |
| `GET` | `/api/model/info` | ✓ | loaded model metadata |
| `POST` | `/api/predict` | ✓ | sensor reading → status + confidence + alert + RUL |
| `GET` | `/api/live` | ✓ | recent simulated readings |
| `GET` | `/api/alerts` | ✓ | alert history |
| `GET` | `/api/predictions` | ✓ | prediction history |
| `POST` | `/api/simulator/start` / `stop` | ✓ | control the simulator |
| `POST` | `/api/simulator/inject/{scenario}` | ✓ | `overheat` / `overload` / `tool_wear` / `reset` |
| `GET` | `/api/report/csv` / `pdf` | ✓ | download a report |

Example:

```bash
curl -X POST http://127.0.0.1:8010/api/predict -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{"air_temp":300.0,"process_temp":311.0,"rot_speed":1400,"torque":72.0,"tool_wear":190,"product_type":"M"}'
```

Returns `status: "Fault"`, `power: 10556 W` (shown as `10.56 kW`), severity
`Critical`, and:

> Reduce load immediately — cut feed rate or depth of cut. Check for a blunt tool
> or incorrect material.

...plus every rule that tripped, including the secondary overstrain fault, and a
`remaining_life` block:

```json
{
  "remaining_min": 10.0,
  "binding_constraint": "overstrain",
  "total_usable_min": 160.0,
  "fraction_consumed": 0.9375,
  "band": "warning",
  "source": "physics",
  "model_remaining_min": 25.9,
  "model_sigma_min": 15.13
}
```

### Configuration

All settings are env-overridable (see `backend/config.py`):
`MHM_SECRET_KEY`, `MHM_DEMO_USER`, `MHM_DEMO_PASSWORD`, `MHM_SIM_INTERVAL`,
`MHM_SIM_AUTOSTART`, `MHM_TOKEN_TTL_MIN`, `MHM_DB_PATH`, `MHM_MODEL_PATH`,
`MHM_RUL_MODEL_PATH`.

---

## Results

Held-out test set, 2,000 rows never seen during training:

```
              precision    recall  f1-score   support

       Fault      1.000     0.838     0.912        68
      Normal      0.998     0.998     0.998      1339
     Warning      0.978     0.997     0.987       593

    accuracy                          0.992      2000
   macro avg      0.992     0.944     0.966      2000
```

5-fold cross-validation on the training half: macro-F1 **0.9457 ± 0.0113** —
confirming the score is not a fluke of one particular split.

Confusion matrix (rows = actual, columns = predicted):

|            | Normal | Warning | Fault |
|------------|--------|---------|-------|
| **Normal**  | 1336 | 3 | 0 |
| **Warning** | 2 | 591 | 0 |
| **Fault**   | 1 | 10 | 57 |

### Reading this operationally

The two error types are **not** equally bad:

- **1 missed fault** (Fault → Normal) out of 68. This is the dangerous cell: a
  real failure reported as healthy costs a breakdown.
- **10 faults downgraded to Warning.** Less serious than it looks — the operator
  is still alerted and still told to inspect the machine; the alert is just amber
  instead of red.
- **0 false alarms** (Normal → Fault). A false alarm only costs an unnecessary
  inspection.

So of 68 real faults, **67 produced an alert** and exactly one was silent.

Feature importance: `tool_wear` 0.192, `rot_speed` 0.190, `temp_diff` 0.168,
`power` 0.132, `torque` 0.132, `strain` 0.110 — the derived features
(`temp_diff`, `power`, `strain`) account for **41%** of all decisions, which is
the quantitative justification for engineering them.

Generated figures in `outputs/figures/`:

| File | Shows |
|---|---|
| `confusion_matrix.png` | counts + row-normalised percentages |
| `actual_vs_predicted.png` | actual vs predicted as a step trace, mismatches marked |
| `feature_importance.png` | which channels drive decisions |
| `confidence_distribution.png` | confidence when right vs wrong — is the score meaningful? |
| `rul_predicted_vs_actual.png` | RUL regressor against the physics target |
| `rul_error_distribution.png` | RUL error; negative = predicted too little life (safe) |
| `rul_uncertainty.png` | RUL with the ±1.96σ tree-spread band |

---

## Questions you should be ready to answer

**"Your accuracy is 99% — isn't that suspiciously high?"**
Partly, yes, and I report why. The Warning class is derived from the sensors, so
the model learns that boundary almost perfectly. The honest number is Fault
recall: 0.838. I also report a Logistic Regression baseline at 0.363 Fault F1 to
show the task is not trivially easy.

**"Why not just use the threshold rules? Why do you need ML at all?"**
Two reasons. The rules cannot detect combinations they were not written for —
the system does raise "Model-detected" alerts where no single limit is crossed.
And the Fault class contains a stochastic tool-breaking point that no fixed
threshold captures. But I do not *replace* the rules with the model — the rules
still run, and they can override the model upward.

**"What happens if the model is wrong?"**
If it is wrongly optimistic, the threshold rules still fire — the model cannot
suppress a hard limit. If it is wrongly pessimistic, you get a false alarm, which
costs an inspection. The asymmetry is deliberate.

**"Why median imputation instead of mean?"**
The median is robust to outliers. The mean is dragged by exactly the extreme
readings we are trying to detect, so using it would partly erase the signal.

**"Why is heat dissipation an AND and not an OR?"**
Cooling depends on forced convection. A small ΔT with a fast spindle is fine —
there is enough airflow. It is only a failure when the spindle is *also* slow.

**"Your RUL model has R² of 0.999 — isn't that just the physics formula?"**
Yes, and I say so up front. The target is deterministic, so the forest learns a
division. I keep it as a cross-check and the API returns the physics value as
authoritative. What the exercise actually produced is the second model: strip out
`tool_wear` and the error goes from 0.4 to 50 minutes, which tells me the
tool-wear counter has no redundancy and is a single point of failure.

**"Why didn't you use an LSTM for RUL?"**
Because AI4I 2020 has no run-to-failure trajectories — no unit id, no time
ordering. A sequence model needs to see degradation unfold. I would have had to
fabricate the cycle structure, and the resulting metric would measure nothing.
Model-based prognostics is the correct family when you have known failure physics
and no run-to-failure data.

**"Why is the temperature difference in °C when you store kelvin?"**
Because a *difference* of 10 K is exactly 10 °C — the offset cancels. Only
absolute temperatures need the 273.15 shift. That distinction has its own
function and its own test, because getting it wrong would turn a healthy ΔT of 10
into -263 and trip a permanent cooling fault.

---

## Limitations (be honest about these)

- **The dataset is synthetic**, albeit physically grounded. Real machine data is
  noisier and has sensor drift, calibration offsets and missing periods.
- **No vibration or pressure channels** — the dataset does not have them.
- **The classifier is not time-aware.** It classifies each reading
  independently. Only the RUL wear-rate estimator looks at history, and it fits a
  simple linear slope. A production system would model the degradation
  trajectory, not just its current gradient.
- **RUL covers tool life only.** Bearings, spindle and drivetrain have their own
  wear mechanisms that this dataset does not instrument, so "remaining useful
  life" here means "remaining tool life" and the README says so rather than
  implying whole-machine prognostics.
- **Auth is deliberately basic** — one in-memory user, no registration, no
  password reset, no refresh tokens, no login rate limiting, no HTTPS. It meets
  "only logged-in users can view the dashboard" and nothing more.
- **SQLite with a single writer lock** is fine for one machine at 1.5 s intervals
  and would not be for a factory-scale fleet.
- **The demo password is in the source** and is intended for local use only. Set
  `MHM_SECRET_KEY` and `MHM_DEMO_PASSWORD` before exposing this anywhere.

---

## Troubleshooting

**`No trained model at model/health_model.pkl`** — run the pipeline steps in
order; `train_model.py` needs `clean_data.py` to have run first.

**Dashboard shows "disconnected"** — the token expires after 8 hours; sign out
and back in. If `MHM_SECRET_KEY` is unset, restarting the server also invalidates
tokens (this is intentional, and the server prints a warning at startup).

**CSS or JS changes do not appear** — bump `?v=` in `frontend/index.html`. The
static routes send `Cache-Control: no-cache`, but a copy cached before that
header was added can persist.

**Simulator not producing readings** — check `GET /api/health` for
`last_simulator_error`, or set `MHM_SIM_AUTOSTART=1`.

**Port 8010 already in use** — `uvicorn backend.main:app --port 8011`.
