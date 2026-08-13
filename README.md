# Machine Health Monitoring / Predictive Maintenance

A predictive maintenance system for a milling machine. It reads sensor data,
classifies the machine as Normal, Warning or Fault, estimates how much cutting
life the tool has left, raises alerts with a recommended action, logs everything,
and shows it on a live dashboard.

Built as a mechatronics project. The thresholds all come from documented machine
physics rather than numbers picked to make the results look good, so each one can
be justified.

---

## Contents

1. [Quick start](#quick-start)
2. [How it fits together](#how-it-fits-together)
3. [Units](#units)
4. [Folder layout](#folder-layout)
5. [Part 1 — Data](#part-1--data)
6. [Part 2 — Model](#part-2--model)
7. [Part 3 — Remaining useful life](#part-3--remaining-useful-life)
8. [Part 4 — Backend](#part-4--backend)
9. [Part 5 — Alerts](#part-5--alerts)
10. [Part 6 — Dashboard](#part-6--dashboard)
11. [Part 7 — Login](#part-7--login)
12. [Part 8 — Tests](#part-8--tests)
13. [API reference](#api-reference)
14. [Results](#results)
15. [Likely questions](#likely-questions)
16. [Limitations](#limitations)
17. [Troubleshooting](#troubleshooting)

---

## Quick start

Needs Python 3.10 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Run the pipeline. Each step needs the one before it:

```bash
python data/download_data.py && python scripts/clean_data.py && python scripts/train_model.py && python scripts/evaluate_model.py && python scripts/train_rul_model.py
```

Start the server:

```bash
uvicorn backend.main:app --reload --port 8010
```

Open http://127.0.0.1:8010 and log in with `engineer` / `maintenance123`.

The simulator starts on its own, so readings appear after a second or two. The
"Inject fault" buttons force a fault when you want to show one.

Run the tests:

```bash
python -m pytest tests/ -v
```

Optional, and worth running once for the write-up — it measures how much
accuracy is lost when the sensors are imperfect:

```bash
python scripts/test_robustness.py
```

---

## How it fits together

```
  data/raw/ai4i2020.csv          10,000 rows of machine sensor data
          |
          |  scripts/clean_data.py      dedupe, fill gaps, drop bad readings,
          v                             derive features, apply threshold rules
  data/processed/machine_health.csv     labelled Normal / Warning / Fault
          |
          |  scripts/train_model.py     stratified 80/20 split, Random Forest
          v
  model/health_model.pkl                model + feature order + class order
          |
          v
  +------------------ backend (FastAPI) ---------------------+
  |                                                          |
  |  simulator.py  -> generates a physically coupled reading  |
  |        |                                                  |
  |  predictor.py  -> Random Forest: status + confidence       |
  |        |                                                  |
  |  rul.py        -> remaining tool life + wear-rate trend    |
  |        |                                                  |
  |  alerts.py     -> combine with the threshold rules         |
  |        |                                                  |
  |  database.py   -> SQLite + append-only JSONL audit log     |
  +----------------------------|-----------------------------+
                               v
              frontend/  dashboard, alert history,
                         CSV / PDF download
```

---

## Units

Inside the system everything is SI: kelvin, watts, newton-metres. That is what
the dataset uses and what the model was trained on. Conversion to friendlier
units happens only where a person reads the number, meaning the dashboard, the
PDF and the CSV.

| Quantity | Stored and sent over the API | Shown on screen |
|---|---|---|
| Air and process temperature | K | °C |
| Temperature difference (ΔT) | K | °C |
| Spindle power | W | kW |
| Torque | N·m | N·m |
| Rotational speed | rpm | rpm |
| Tool wear and RUL | min | min |
| Mechanical strain | min·N·m | min·N·m |

One thing worth knowing when reading the code: a temperature *difference* of
8.6 K is the same as 8.6 °C, because the offset cancels. That is why the constant
is called `HDF_TEMP_DIFF_K` but the dashboard prints "8.6 °C" with the same
number. Only absolute temperatures need the 273.15 shift.

Converting at the edge rather than storing Celsius everywhere keeps it to one
place where a unit can go wrong. If Celsius reached the feature vector the model
would get 25.4 where it expects 298.5 and return a wrong answer without any
error. Unit bugs are quiet, which is what makes them expensive. The conversions
live in [backend/units.py](backend/units.py) and are covered by
`tests/test_units.py`.

To send 25 °C to the API, post `298.15`.

---

## Folder layout

```
machine-health-monitor/
├── data/
│   ├── download_data.py        fetches the dataset, no API key needed
│   ├── raw/ai4i2020.csv        original 10,000-row dataset
│   └── processed/              cleaned and labelled output
│
├── scripts/
│   ├── clean_data.py           cleaning, labelling, RUL target
│   ├── train_model.py          Random Forest and a Logistic Regression baseline
│   ├── evaluate_model.py       confusion matrix, plots, per-class scores
│   ├── train_rul_model.py      RUL regressor and the measurements on it
│   └── test_robustness.py      how the model holds up on imperfect sensors
│
├── model/
│   ├── health_model.pkl        trained classifier bundle
│   ├── metadata.json           scores, feature importance, training config
│   ├── rul_model.pkl           RUL regressor, used as a cross-check
│   └── rul_metadata.json       RUL scores and uncertainty
│
├── backend/
│   ├── thresholds.py           every physical limit, defined once
│   ├── units.py                SI to display conversion, defined once
│   ├── rul.py                  remaining tool life and wear-rate trend
│   ├── trends.py               projects when a channel crosses its limit
│   ├── predictor.py            loads the models, runs inference
│   ├── alerts.py               model plus rules to severity and action
│   ├── simulator.py            sensor simulator with realistic coupling
│   ├── database.py             SQLite and append-only audit log
│   ├── auth.py                 PBKDF2 password hashing and JWT
│   ├── reporting.py            CSV and PDF generation
│   ├── schemas.py              request and response validation
│   ├── config.py               settings, all overridable by env var
│   └── main.py                 the API endpoints
│
├── frontend/
│   ├── index.html              login and dashboard markup
│   ├── styles.css              colour-coded status styling
│   └── app.js                  polling, rendering, canvas chart
│
├── tests/
│   ├── test_alerts.py          alert logic, thresholds, repeat suppression
│   ├── test_rul.py             RUL physics, wear trend, RUL alert rule
│   ├── test_trends.py          trend projection and its significance test
│   ├── test_units.py           unit conversions
│   ├── test_api.py             endpoints, login, rate limits, audit, reports
│   └── test_thresholds.py      offline labeller vs live alerter
│
└── outputs/                    everything the system generates
    ├── figures/                plots from evaluate_model.py
    ├── metrics/                JSON metrics
    ├── logs/                   monitoring.db and audit_log.jsonl
    └── exports/                reports downloaded from the dashboard
```

Nothing generated is written next to source. Delete `outputs/` and the pipeline
rebuilds it.

---

## Part 1 — Data

### The dataset

AI4I 2020 Predictive Maintenance Dataset, 10,000 rows. Kaggle publishes the same
table as "Machine Predictive Maintenance Classification"; it originally comes
from the UCI repository. `download_data.py` pulls from UCI because that needs no
API token.

| Column | Meaning |
|---|---|
| `Type` | product quality tier, L / M / H |
| `Air temperature [K]` | ambient temperature, about 25 °C |
| `Process temperature [K]` | temperature at the cutting process, about 35 °C |
| `Rotational speed [rpm]` | spindle speed |
| `Torque [Nm]` | spindle torque, N·m |
| `Tool wear [min]` | minutes of cutting on the current tool |
| `Machine failure` plus `TWF/HDF/PWF/OSF/RNF` | ground-truth failure flags |

The dataset has no vibration or pressure channel. Rather than invent one, the
system uses the five real channels. Nothing about the pipeline is specific to
them, so adding a vibration sensor would mean adding a column in `thresholds.py`
and retraining.

### Cleaning

`scripts/clean_data.py` does six things.

1. Renames columns to snake_case.
2. Removes duplicate rows, comparing sensor values only and ignoring `UDI` and
   `Product ID`. Those are just row counters, and two identical readings logged
   under different IDs are still one measurement.
3. Fills missing values, numeric with the median and categorical with the mode.
   The median is used because it is not dragged around by outliers, and outliers
   are exactly what we are trying to detect. Filling rather than dropping means
   one dead channel does not throw away the other four.
4. Drops readings that break physics, such as negative torque or a stopped
   spindle. Those are broken sensors, not broken machines, and training on them
   teaches the model nonsense.
5. Derives three features that the failure physics is actually written in.

   | Feature | Formula | Stored as | Shown as |
   |---|---|---|---|
   | `temp_diff` | `process_temp - air_temp` | K | °C |
   | `power` | `torque × ω`, with `ω = rpm × 2π/60` | W | kW |
   | `strain` | `tool_wear × torque` | min·N·m | min·N·m |

6. Labels each row Normal, Warning or Fault.

### How the labels are decided

The dataset was generated from five documented failure modes, so the same
thresholds are reused and a Warning band is placed just before each one.

| Failure mode | Fault condition | Warning band |
|---|---|---|
| Heat dissipation (HDF) | ΔT below 8.6 °C **and** speed below 1380 rpm | ΔT below 9.5 °C and speed below 1500 rpm |
| Power (PWF) | power below 3.5 kW **or** above 9.0 kW | below 4.0 kW or above 8.5 kW |
| Overstrain (OSF) | strain above 11000 / 12000 / 13000 min·N·m for L / M / H | above 85% of that limit |
| Tool wear (TWF) | wear between 200 and 240 min | wear above 180 min |
| Random (RNF) | 0.1% chance, nothing to do with the sensors | — |

Two details are worth pointing out.

The heat dissipation rule is an AND, not an OR. Cooling relies on air being
pushed over the spindle, so a small temperature difference only becomes dangerous
when the spindle is also turning slowly. There is a test for this specifically.

The overstrain limit depends on the quality tier. A strain of 12100 min·N·m is a
fault on an L-grade tool and on an M-grade one, but fine on H.

Fault beats Warning beats Normal. A row counts as a Fault if the dataset's own
failure flag is set. We trust the ground truth for that class rather than
re-deriving it, since RNF failures cannot be predicted from the sensors at all.

Result: 66.96% Normal, 29.65% Warning, 3.39% Fault.

### One caveat about the Warning class

The Warning label is worked out from the sensors, so it is a fixed function of
them and a model will learn that boundary almost perfectly. That is not cheating,
but it does mean the headline accuracy is flattering.

The Fault class is different. It is not a clean function of the sensors: it
contains a random component and a tool that breaks somewhere between 200 and 240
minutes rather than at a fixed point. Fault recall is the number that shows
whether the model learned anything, which is why `evaluate_model.py` reports it
on its own instead of leaving it inside the overall accuracy.

---

## Part 2 — Model

### Features

`type_code, air_temp, process_temp, rot_speed, torque, tool_wear, temp_diff,
power, strain`

Giving the model `power` and `strain` directly makes the biggest difference of
anything in the project. A decision tree splits on one variable at a time, so it
would need a deep and fragile staircase of splits to approximate `torque × rpm`.
Handing it the product turns that into one clean split.

`type_code` is ordinal (L=0, M=1, H=2) rather than one-hot, because the quality
tiers really are ordered. The overstrain limit rises with them.

### Why Random Forest

`train_model.py` trains a Logistic Regression baseline alongside the forest and
prints both, because the comparison is the argument for the choice.

| Model | Accuracy | Macro F1 | Fault F1 |
|---|---|---|---|
| Logistic Regression | 0.7170 | 0.5917 | 0.3630 |
| Random Forest | 0.9920 | 0.9657 | 0.9120 |

Logistic Regression draws one straight line per class. The failure rules are
conjunctions ("ΔT low AND rpm low") and two-sided bands ("power too low OR too
high"), and no single straight line can express either. A Random Forest is a vote
across many axis-aligned trees, which is the same shape as a threshold rule.

### The class imbalance

Fault is only 3.4% of the rows, so two things are needed.

The split is stratified, because a plain random 80/20 could easily leave the test
set with an unrepresentative number of the 339 fault rows.

`class_weight="balanced"` makes each Fault row count roughly 20 times a Normal
one during training, so the model cannot score well by ignoring faults.

Without both, a model that answered "Normal" every time would already reach 67%
accuracy. That is why accuracy on its own proves nothing here.

### What gets saved

`model/health_model.pkl` holds a bundle rather than a bare estimator:

```python
{"model": rf, "features": [...], "type_code": {...},
 "classes": [...], "trained_at": "..."}
```

The backend rebuilds each input row against the saved `features` list instead of
a hardcoded order. Retrain with a different feature set and the backend still
works, rather than quietly feeding `torque` into the column the model thinks is
`power` and returning a confident wrong answer.

---

## Part 3 — Remaining useful life

"Is the machine healthy" is a classification question with three answers. "How
long have I got" is a regression question with a continuous answer, and a planner
can schedule around "31 minutes" in a way they cannot around the word "Warning".

### The constraint to be upfront about

The usual data-driven approach to RUL, the NASA C-MAPSS style, needs
run-to-failure trajectories: many units, each logged from new until it dies.
AI4I 2020 does not have that. Its rows are independent samples with no unit ID
and no time ordering, so you cannot follow one tool from new to worn. Adding a
made-up "cycle" column and training an LSTM on it would give a good-looking
number that means nothing.

So this uses the other standard approach, model-based prognostics, working from
the failure physics instead. That is the right family to pick when the physics is
known and run-to-failure data is not available.

### Layer 1: the physics, and what the system actually uses

Two things limit tool life, and which one bites depends on the load.

| Constraint | Remaining cutting minutes |
|---|---|
| Tool wear | `200 - tool_wear` |
| Overstrain | `(limit / torque) - tool_wear` |

RUL is the smaller of the two, floored at zero.

This is why RUL is not simply `200 - tool_wear`. Rearranging the overstrain
condition for the wear at which it trips gives a ceiling that moves with torque:

| Tool wear | Torque | Ceiling | RUL | What binds |
|---|---|---|---|---|
| 150 min | 40 N·m | 200 min | 50 min | tool wear |
| 150 min | 75 N·m | 160 min | 10 min | overstrain |

Same tool, same wear, five times less life left. Cutting harder does not just use
the tool up faster, it lowers the ceiling, and no tool-wear threshold would tell
you that. In the cleaned dataset 452 rows out of 10,000 are overstrain-bound, and
those are the ones a wear threshold on its own would miss.

Cooling faults and power overloads are deliberately left out of RUL. They are
instant failure conditions rather than wear mechanisms, so they do not eat tool
life, they end it. The alert rules already handle them.

### Layer 2: the learned model, and why it did not win

`scripts/train_rul_model.py` trains a `RandomForestRegressor` on the same nine
features. It was measured and it does not beat the formula.

| | MAE | RMSE | R² |
|---|---|---|---|
| Full model | 0.38 min | 2.10 | 0.9989 |
| Without `tool_wear` | 49.64 min | 60.01 | 0.116 |

An R² of 0.999 is not an achievement here. The target is a fixed formula, so the
forest is learning a division, and `tool_wear` accounts for 99.7% of the feature
importance. The spread across the 300 trees is close to zero as well, because
there is no noise in the target for them to disagree about.

| Region | Median σ |
|---|---|
| All test rows | 0.00 min |
| Tool-wear bound | 0.00 min |
| Overstrain bound | 6.21 min |

The only place the trees disagree is the boundary where the binding constraint
switches, and there the model is visibly off. At 150 min and 75 N·m the physics
gives 10.0 min while the model gives 25.9 ± 15.1 min. The band is behaving
correctly, in that it is wide and it does cover the truth, but the formula is
simply right.

So the system uses the physics value. The regressor is shown on the dashboard as
a cross-check and nothing depends on it; delete `rul_model.pkl` and the API still
works.

The second row of that table is the finding worth keeping. Take `tool_wear` away
and the error grows by a factor of 131. The tool-wear counter has no backup, so
if the encoder fails or the counter is reset by mistake, RUL has to fall back to
the formula rather than to this model.

### Layer 3: turning cutting minutes into clock time

Layers 1 and 2 answer how many minutes of cutting are left. An operator wants to
know how long until they have to stop. `estimate_wear_rate` fits a least-squares
slope through `tool_wear` over the last 20 or so readings, so the rate is
measured rather than assumed.

Three details matter here.

It skips over tool changes. Wear resets to zero when the tool is swapped, and
fitting across that reset would give a negative slope and a nonsense answer, so
only the readings since the last reset are used.

The window is about 20 readings. Fit the whole buffer and the rate lags badly,
so a machine that started cutting hard a minute ago still reports a normal rate,
which is exactly when the warning is wanted.

The rate is normalised. The simulator compresses time, advancing the tool about
2.2 cutting minutes per 1.5 second tick so a full tool life is watchable in a
couple of minutes. Its raw rate is therefore around 88, which is arithmetically
right and useless on a screen. Dividing by the nominal rate cancels the
compression and leaves something readable: "wearing this tool 1.6× faster than
normal". The dashboard shows it as `1 h 59 min (wear 0.92× nominal)`.

The simulator's wear rate depends on load, following the shape of Taylor's tool
life equation, so working the machine hard genuinely pulls the deadline in. The
projection drops below the raw RUL under load and sits above it when running
gently.

### RUL as an alert

The RUL rule only fires when overstrain is the binding constraint. When tool wear
binds, the existing wear threshold already says the same thing at 180 minutes,
and showing an operator two rows for one problem is how alert lists stop getting
read. The overstrain case is the gap nothing else covers.

---

## Part 4 — Backend

FastAPI, chosen because Pydantic validates every payload before it reaches your
code and `/docs` gives interactive API documentation for free.

| Module | What it does |
|---|---|
| `thresholds.py` | every physical limit, imported by both the offline cleaner and the live API |
| `units.py` | the one place SI values are converted for display |
| `rul.py` | remaining tool life and the wear-rate trend |
| `predictor.py` | loads the models once at startup and runs inference |
| `alerts.py` | combines the model verdict with the threshold rules |
| `simulator.py` | generates fake readings with realistic coupling |
| `database.py` | SQLite plus an append-only JSONL log |
| `auth.py` | PBKDF2 hashing, JWT issue and verify |
| `reporting.py` | CSV and PDF generation |

### Thresholds defined once

`scripts/clean_data.py` imports its constants from `backend/thresholds.py`. If
the offline labeller used one set of numbers and the live alerter used another,
the model would be trained to spot one thing and deployed to explain something
slightly different. That is a quiet bug and a horrible one to track down, so
`tests/test_thresholds.py` runs both code paths over all 10,000 rows and checks
they flag the same ones.

### The simulator

Independent random numbers would never produce a believable fault, because real
faults come from channels affecting each other. The simulator reproduces three
couplings that exist on a real spindle.

The drive holds power roughly constant, so `rpm = P / torque × 60/2π` and a
heavier cut slows the spindle down.

A blunt tool needs more force, so torque climbs with wear, which through the
first coupling drags rpm down and pushes strain towards the overstrain limit.

Lower rpm means less airflow, so ΔT shrinks, which is the heat dissipation
failure condition.

Faults therefore arrive as a cascade, the way they do on a real machine. In
testing, an injected cooling fault dropped rpm to 974, which raised torque to
62 N·m, which pushed strain to 14413 min·N·m and tripped a second, different
failure mode.

A tool change resets wear to zero and the cycle starts again, so a long demo
shows repeated tool lives.

### The audit trail

Every prediction and alert is written twice.

SQLite (`outputs/logs/monitoring.db`) is queryable, and the alert history and the
downloadable reports are both queries against it.

JSONL (`outputs/logs/audit_log.jsonl`) is append-only, one JSON object per line,
never updated or deleted. If the database is tampered with, the flat log still
shows what the system saw and what it said.

Login attempts are logged too. For maintenance work this is not optional. If the
machine breaks and the system said "Normal" five minutes earlier, you need to be
able to show exactly what readings it was given.

---

## Part 5 — Alerts

### How the model and the rules combine

An alert is not just whatever the model said. It combines two independent sources
of evidence, and the rule is that the thresholds can escalate the model but the
model can never overrule a threshold. Effective severity is the worse of the two.

The reasoning: a hard physical limit is a fact, not a prediction. If the model has
a bad moment and says "Normal" while the spindle is drawing 9.5 kW, the operator
still needs to be told. Safety interlocks work the same way, in that a learned
layer can add sensitivity but never gets to switch off a hard limit.

The other direction is allowed, and it is the genuinely predictive part. If every
threshold is inside limits but the model recognises a combination that tends to
come before failures, it raises a Warning by itself. Thresholds cannot do that.
It shows up in the generated PDF as "Model-detected warning condition".

### Severity

| Model status | Severity | Meaning |
|---|---|---|
| Normal | — | no alert, nothing logged |
| Warning | Warning | plan a fix, the machine keeps running |
| Fault | Critical | act now, the machine should stop |

### Recommended actions

Every alert carries a physical instruction taken from the most urgent rule that
tripped, so it names the actual component.

| Condition | Recommended action |
|---|---|
| Heat dissipation failure | Stop the machine. Check coolant flow, clean the heat exchanger and verify the cooling fan is running before restart. |
| Cooling margin low | Inspect coolant level and airflow path. Raise spindle speed to improve forced convection if the process allows. |
| Power overload | Reduce load immediately, cutting feed rate or depth of cut. Check for a blunt tool or incorrect material. |
| Power underrun | Check the drive coupling and belt tension, the spindle may be slipping or running unloaded. |
| Mechanical overstrain | Stop and replace the tool. Inspect the spindle bearing and tool holder for damage before resuming. |
| Approaching overstrain | Reduce load or change the tool early. Inspect the bearing at the next available stop. |
| Tool life exceeded | Replace the cutting tool now. Do not start another cycle. |
| Tool nearing end of life | Schedule a tool change within the next N minutes of cutting. |
| Tool life shortened by high load | Reduce torque to extend tool life, or schedule a change within N minutes. |

If several rules trip at once the message names the others too, so nobody fixes
one thing and assumes they are done. A test checks that no alert can be raised
without an instruction attached.

### The trend layer

Every threshold rule looks at one reading and asks whether it is out of range.
That is detection, not prediction. A spindle whose cooling has been degrading
for four minutes reads perfectly normal right up until it does not.

`backend/trends.py` asks the other question: fit a line through the recent
readings, and at this rate, when does the channel cross its limit? If the answer
is inside the horizon, say so now.

Two gates keep it quiet, and both have to pass:

- **The window must cover enough real time.** Twenty readings at four ticks a
  second span five seconds, and five seconds cannot measure a per-minute trend.
  This was a real bug, found by running the simulator fast: the fitted slope was
  almost pure noise and flipped sign every tick.
- **The slope must be statistically distinguishable from zero.** Scattered points
  still fit *some* line, so the fit reports `t = |slope| / standard error` and
  anything under 2 is discarded. This is better than a fixed threshold on the
  slope because it adapts to how noisy the data is and how much of it there is,
  instead of needing a hand-tuned constant per channel and per tick rate.

A trend hit is capped at Warning and can never declare a Fault. It is a
straight-line extrapolation and assumes the trend continues, which will not
always be true, so only a measured limit gets to stop the machine.

In a typical run the trend layer fires roughly half a minute before the
threshold it is predicting:

```
09:36:26  Warning  Cooling is degrading    [trend_temp_diff_falling]
09:36:53  Warning  Cooling margin low      [cooling]
```

The first line is the projection. The second is the measured threshold catching
up 27 seconds later. That gap is the whole point of the layer.

### Not repeating yourself

A fault condition persists for many readings. Unsuppressed, at a 1.5 second tick
one cooling fault writes about forty identical rows a minute and the alert
history becomes one condition repeated until it scrolls off the screen, which is
the fastest way to get an operator to stop reading alerts.

So an alert is logged when what it *says* changes, or when the same condition has
persisted for `MHM_ALERT_REPEAT_SEC` (60 by default) and is worth restating. The
key is the effective status plus which rules tripped, not the message text, since
the message embeds live measurements and differs on every tick.

Trend rules are deliberately excluded from that key whenever a measured rule is
present. A projection naturally comes and goes as the fit wobbles near its
threshold, and if that were part of the key every flicker would re-log the
underlying fault and defeat the suppression entirely.

Measured over 100 seconds of simulation: **63 predictions, 4 alert rows**, each a
distinct condition. The audit trail is unaffected — every prediction is still
logged, every tick. Suppression only trims the alerts table, which is a
human-facing summary rather than the record of what was seen.

---

## Part 6 — Dashboard

Plain HTML, CSS and JavaScript. No framework, no build step, no CDN, so every
line that puts something on the screen can be read directly.

The status is colour-coded, with green, amber and red mapping onto Normal,
Warning and Fault. The Fault dot pulses and the browser tab title changes to
"(Fault) Machine Health Monitor" so it is noticeable in a background tab.

Eight tiles show the five raw channels and the three derived ones, each with a
fill bar and its own limit written underneath. The colours come from the same
thresholds the backend uses, and the bands are evaluated on the raw SI value so a
display change cannot pull them out of step.

The remaining-life panel shows minutes of cutting left, a bar that fills as life
is used up, what is limiting it, the projection in clock time with the measured
wear rate, and the model's cross-check with its uncertainty.

The action card appears only when there is something to do, showing the rule that
tripped, the measured value against the limit, and the action.

The trend chart is drawn on a canvas. Power, ΔT and remaining life are normalised
into one plot area because their units are nothing like each other, so it is a
comparison of shapes while the tiles carry the exact values. Underneath it a
strip colours one bar per reading, which makes the Normal to Warning to Fault
progression readable at a glance.

Below that are the alert history, the report download buttons, and simulator
controls including buttons to force a cooling fault, a power overload or a worn
tool.

Two implementation notes. Polling is used rather than WebSockets: the simulator
produces a reading every 1.5 seconds, so a socket would save almost nothing while
adding reconnect logic and heartbeats to debug, and if a poll fails the next one
just works. Downloads go through `fetch()` rather than a plain link, because a
link cannot carry the `Authorization` header, so the file is pulled into a Blob
and a temporary object URL is clicked instead.

---

## Part 7 — Login

Username and password login returning a signed JWT. Every dashboard endpoint
requires `Authorization: Bearer <token>`.

Passwords are stored as `pbkdf2_hmac('sha256', password, salt, 200_000)`, never
in plain text. The salt means two users with the same password get different
hashes, so they cannot all be cracked at once from one rainbow table. The 200,000
iterations make each guess deliberately slow; plain SHA-256 can be guessed
billions of times a second on a GPU and this brings that down to thousands.
`hmac.compare_digest` compares in constant time so the hash cannot be worked out
byte by byte from response timing.

A JWT is three base64 parts: header, payload, signature. The payload holds the
username and an expiry, and the signature is HMAC-SHA256 over the first two parts
using the secret key. Anyone can read a JWT, so nothing secret goes in it, but
nobody can forge one without the key. There is a test that forges a token with
the wrong key and checks it is rejected.

Two deliberate details. Login failures give the same vague message whether the
username or the password was wrong, and an unknown username still runs a dummy
hash so the response time does not reveal which usernames exist. And if
`MHM_SECRET_KEY` is not set, a random key is generated per process, so tokens
stop working after a restart. That is mildly annoying but much safer than
shipping a signing key that anyone reading the repo could forge tokens with.

---

## Part 8 — Tests

```bash
python -m pytest tests/ -v
```

117 tests, all passing. They write to a temporary database, never the real audit
trail.

| File | What it covers |
|---|---|
| `test_alerts.py` | each threshold rule on its own, the AND in the cooling rule, the two-sided power envelope, the per-tier overstrain limit, that rules escalate the model and the model cannot suppress a rule, that every alert carries an action, and that repeat suppression collapses a persisting condition without letting a flickering trend re-log it. Loads no model, so it is pure logic. |
| `test_rul.py` | that RUL is not `200 - wear` and high torque lowers the ceiling, the per-tier ceiling ordering, that the wear-rate fit survives a tool change and refuses to guess when wear is flat, the time-compression normalisation, that the RUL rule does not double up with the tool-wear threshold but does fire for the overstrain gap, and that the vectorised training target matches the scalar physics row for row. |
| `test_trends.py` | that a projection fires only for a channel genuinely heading at its limit, that noise and a too-short window produce nothing, that the t-statistic separates a real slope from scatter, and that a trend can raise a Warning but never a Fault. |
| `test_units.py` | conversions round-trip, and a temperature difference has no offset applied. |
| `test_api.py` | login protects every endpoint, forged tokens are rejected, failed logins are rate limited while successful ones never are, prediction shape and probabilities, implausible readings rejected with 422, predictions and alerts reaching both audit sinks, simulator tick and buffer and injection, CSV columns carrying their units and the values actually being converted, the PDF starting with `%PDF-`, and the database running in WAL mode. |
| `test_thresholds.py` | the offline pandas labeller and the live scalar alerter flag identical rows across all 10,000. |

---

## API reference

Interactive docs at http://127.0.0.1:8010/docs.

| Method | Path | Login | Purpose |
|---|---|---|---|
| `GET` | `/api/health` | — | liveness, model and simulator state |
| `POST` | `/api/auth/login` | — | username and password to JWT |
| `GET` | `/api/auth/me` | yes | current user |
| `GET` | `/api/model/info` | yes | loaded model metadata |
| `POST` | `/api/predict` | yes | reading to status, confidence, alert, RUL |
| `GET` | `/api/live` | yes | recent simulated readings |
| `GET` | `/api/alerts` | yes | alert history |
| `GET` | `/api/predictions` | yes | prediction history |
| `POST` | `/api/simulator/start` and `/stop` | yes | control the simulator |
| `POST` | `/api/simulator/inject/{scenario}` | yes | `overheat`, `overload`, `tool_wear`, `reset` |
| `GET` | `/api/report/csv` and `/pdf` | yes | download a report |

Example, remembering that temperatures go over the API in kelvin:

```bash
curl -X POST http://127.0.0.1:8010/api/predict -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{"air_temp":300.0,"process_temp":311.0,"rot_speed":1400,"torque":72.0,"tool_wear":190,"product_type":"M"}'
```

That returns `status: "Fault"` at 10.56 kW, severity `Critical`, and:

> Reduce load immediately — cut feed rate or depth of cut. Check for a blunt tool
> or incorrect material.

along with every rule that tripped, including the secondary overstrain fault, and
a `remaining_life` block:

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

Everything is overridable by environment variable, listed in
`backend/config.py`: `MHM_SECRET_KEY`, `MHM_DEMO_USER`, `MHM_DEMO_PASSWORD`,
`MHM_SIM_INTERVAL`, `MHM_SIM_AUTOSTART`, `MHM_TOKEN_TTL_MIN`, `MHM_DB_PATH`,
`MHM_MODEL_PATH`, `MHM_RUL_MODEL_PATH`, `MHM_ALERT_REPEAT_SEC`,
`MHM_LOGIN_MAX_ATTEMPTS`, `MHM_LOGIN_WINDOW_SEC`.

Two worth knowing about when demoing: `MHM_SIM_INTERVAL` speeds the simulator up,
and `MHM_ALERT_REPEAT_SEC` controls how often a persisting condition is
restated in the alert history.

---

## Results

Held-out test set, 2,000 rows the model never saw during training:

```
              precision    recall  f1-score   support

       Fault      1.000     0.838     0.912        68
      Normal      0.998     0.998     0.998      1339
     Warning      0.978     0.997     0.987       593

    accuracy                          0.992      2000
   macro avg      0.992     0.944     0.966      2000
```

Five-fold cross-validation on the training half gives a macro F1 of 0.9457 ±
0.0113, which says the score is not a fluke of one particular split.

Confusion matrix, rows actual and columns predicted:

|            | Normal | Warning | Fault |
|------------|--------|---------|-------|
| **Normal**  | 1336 | 3 | 0 |
| **Warning** | 2 | 591 | 0 |
| **Fault**   | 1 | 10 | 57 |

### What that means in practice

The two kinds of error are not equally bad.

One real fault out of 68 was reported as Normal. That is the dangerous cell,
since a missed failure costs a breakdown.

Ten faults were downgraded to Warning. Less serious than it looks, because the
operator is still alerted and still told to inspect the machine; the alert is
amber rather than red.

No Normal rows were reported as Fault. A false alarm only costs an unnecessary
inspection.

So of 68 real faults, 67 produced an alert and one was silent.

Feature importance comes out as `tool_wear` 0.192, `rot_speed` 0.190,
`temp_diff` 0.168, `power` 0.132, `torque` 0.132, `strain` 0.110, `air_temp`
0.046, `process_temp` 0.024, `type_code` 0.005. The three derived features
account for 41% of all decisions, which is the argument for engineering them in
the first place.

Figures land in `outputs/figures/`:

| File | Shows |
|---|---|
| `confusion_matrix.png` | counts and row percentages |
| `actual_vs_predicted.png` | actual against predicted as a step trace, mismatches marked |
| `feature_importance.png` | which channels drive decisions |
| `confidence_distribution.png` | confidence when right against when wrong |
| `rul_predicted_vs_actual.png` | RUL regressor against the physics target |
| `rul_error_distribution.png` | RUL error, negative meaning it predicted too little life |
| `rul_uncertainty.png` | RUL with the ±1.96σ tree-spread band |

---

## Likely questions

**Your accuracy is 99%, isn't that suspiciously high?**
Partly, and the README says why. The Warning class is derived from the sensors so
the model learns that boundary almost perfectly. The honest number is Fault
recall at 0.838. The Logistic Regression baseline at 0.363 Fault F1 is there to
show the task is not trivially easy.

**Why use ML at all when you already have threshold rules?**
Two reasons. The rules cannot catch combinations nobody wrote a rule for, and the
system does raise model-only alerts where no single limit has been crossed. And
the Fault class contains a tool that breaks at an unpredictable point, which no
fixed threshold captures. The model does not replace the rules though. They still
run, and they can override it upwards.

**What happens if the model is wrong?**
If it is wrongly optimistic the threshold rules still fire, because the model
cannot suppress a hard limit. If it is wrongly pessimistic you get a false alarm
and an unnecessary inspection. The asymmetry is on purpose.

**Why median imputation instead of mean?**
The median is not moved much by outliers. The mean gets dragged by exactly the
extreme readings we are trying to detect, so using it would partly erase the
signal.

**Why is heat dissipation an AND rather than an OR?**
Cooling depends on airflow over the spindle. A small ΔT with a fast spindle is
fine because there is enough air moving. It only becomes a failure when the
spindle is slow as well.

**Your RUL model scores R² 0.999, isn't that just the formula?**
Yes, and that is stated up front. The target is deterministic so the forest
learns a division. It is kept as a cross-check and the API returns the physics
value. What the exercise actually produced is the second model: remove
`tool_wear` and the error goes from 0.4 to 50 minutes, which says the tool-wear
counter is a single point of failure with no redundancy.

**Why not an LSTM for RUL?**
Because AI4I 2020 has no run-to-failure trajectories, no unit ID and no time
ordering. A sequence model needs to watch degradation unfold. The cycle structure
would have to be fabricated and the resulting score would measure nothing.

**Why does the dashboard show °C when the code constant is called `_K`?**
A temperature difference of 8.6 K is 8.6 °C, because the offset cancels. Only
absolute temperatures need the 273.15 shift. That distinction has its own
function and its own test, since getting it wrong would turn a healthy ΔT of 10
into -263 and trip a permanent cooling fault.

---

## Limitations

The dataset is synthetic, though grounded in real physics. Real machine data is
noisier and has sensor drift, calibration offsets and missing periods. How much
that costs is measured rather than guessed at, in
[Sensor robustness](#sensor-robustness) below.

There are no vibration or pressure channels, because the dataset does not have
them. Adding a real accelerometer would be the single biggest improvement to
this project, and the architecture is ready for it: because alerts combine a
learned model with independent physical rules, a vibration channel can go in as
a threshold rule with no retraining and no labelled vibration data. ISO 20816
gives vibration-severity limits by machine class, so the thresholds would be
justified the same way the AI4I ones are.

The classifier is not time-aware, and cannot be made so from this dataset. Its
rows have no unit id and no time ordering, so there is no sequence to learn
from, and training on the simulator's own output would only teach the model the
simulator. The trend layer in `backend/trends.py` addresses this from the rules
side instead: it fits a line through recent readings and projects when a channel
will cross its limit. That is genuinely predictive and needs no training data,
but it is a straight-line extrapolation and assumes the trend continues, which
is why a trend hit can never be worse than a Warning.

RUL covers tool life only. Bearings, spindle and drivetrain have their own wear
mechanisms that this dataset does not instrument, so "remaining useful life"
here means remaining tool life.

Login is deliberately basic: one in-memory user, no registration, no password
reset, no refresh tokens and no HTTPS. Failed logins are rate limited, but the
counter is per-process and in memory, so two workers means two windows and a
restart clears it. Doing that properly needs shared state.

SQLite runs in WAL mode, so the dashboard reads while the simulator writes.
There is still one writer at a time, which is fine for one machine at 1.5 second
intervals and would not be for a factory of them.

The demo password is in the source and is meant for local use. Set
`MHM_SECRET_KEY` and `MHM_DEMO_PASSWORD` before putting this anywhere else.

---

## Sensor robustness

`python scripts/test_robustness.py` corrupts the held-out test set the way real
instruments fail and re-scores without retraining. It turns "the dataset is
synthetic" from a caveat into a number.

Four corruptions: Gaussian noise, calibration drift, ADC quantisation, and
dropout where a channel freezes on its last value. Levels come from typical
instrument specifications, not from whatever made the results look good — a type
K thermocouple has a standard tolerance of about ±1.5 K, a rotary torque sensor
0.1–0.5% of full scale, an encoder well under 0.1% on speed.

Watch Fault recall, not accuracy. Accuracy is dominated by the Normal class and
barely moves.

| Corruption | Fault recall | Change |
|---|---|---|
| clean baseline | 0.838 | — |
| noise, torque σ = 2.0 N·m | 0.750 | −0.088 |
| every channel noisy, good sensors | 0.765 | −0.074 |
| every channel noisy, poor sensors | 0.706 | −0.132 |
| dropout, torque 10% of readings | 0.750 | −0.088 |
| **drift, process temp +1 K** | **0.559** | **−0.279** |
| **drift, air temp −1 K** | **0.559** | **−0.279** |

### The finding

Calibration drift on a temperature channel is far more damaging than noise on
anything, and the direction decides whether it is dangerous.

`temp_diff = process_temp − air_temp`, so the two channels enter with opposite
signs. A process-temp sensor reading **high**, or an air-temp sensor reading
**low**, inflates the apparent cooling margin and hides heat-dissipation faults.
One kelvin in that direction — inside a type K thermocouple's tolerance — costs
a third of all fault detection.

Drift the other way and Fault recall goes slightly *up*: the machine looks worse
than it is, so the system raises false alarms. That costs an inspection. The
first direction costs a machine.

The practical conclusion is that calibrating the two thermocouples against each
other matters more than buying quieter sensors, and it is not a conclusion
anyone would reach by looking at accuracy.

---

## Troubleshooting

**`No trained model at model/health_model.pkl`** — run the pipeline steps in
order, since `train_model.py` needs `clean_data.py` to have run first.

**Dashboard says "disconnected"** — the token lasts 8 hours, so log out and back
in. If `MHM_SECRET_KEY` is unset, restarting the server also invalidates tokens.
That is intentional and the server prints a warning at startup.

**CSS or JS changes do not show up** — bump the `?v=` number in
`frontend/index.html`. The static routes send `Cache-Control: no-cache`, but a
copy cached before that header existed can hang around.

**Simulator not producing readings** — check `GET /api/health` for
`last_simulator_error`, or set `MHM_SIM_AUTOSTART=1`.

**Port 8010 already in use** — `uvicorn backend.main:app --port 8011`.
