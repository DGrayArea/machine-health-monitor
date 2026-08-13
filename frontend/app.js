/* Machine Health Monitor — dashboard logic.
 *
 * Deliberately plain JavaScript: no framework, no build step, no CDN. You can
 * open this file and read every line that puts a pixel on the screen, which
 * matters when you have to explain the project.
 *
 * Flow:
 *   1. Log in  -> POST /api/auth/login -> keep the JWT in sessionStorage.
 *   2. Poll    -> GET /api/live every ~1.5 s (the simulator's tick rate).
 *   3. Render  -> status banner, sensor tiles, trend chart, alert table.
 *
 * WHY POLLING AND NOT WEBSOCKETS
 *   The simulator produces one reading every 1.5 s. At that rate a WebSocket
 *   saves almost nothing but adds reconnect logic, heartbeats and a second code
 *   path to debug. Polling is the right amount of machinery for the job — and
 *   if the connection drops, the next poll simply succeeds.
 *
 * WHY sessionStorage AND NOT localStorage
 *   sessionStorage is cleared when the tab closes, so an unattended machine on
 *   the shop floor does not stay logged in indefinitely.
 */

'use strict';

const TOKEN_KEY = 'mhm_token';
const POLL_MS = 1500;
const MAX_POINTS = 80;

let pollTimer = null;
let history = [];

/* ------------------------------------------------------------------ */
/* API helper                                                          */
/* ------------------------------------------------------------------ */

const getToken = () => sessionStorage.getItem(TOKEN_KEY);

async function api(path, options = {}) {
  const headers = Object.assign({}, options.headers);
  const token = getToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;
  if (options.body) headers['Content-Type'] = 'application/json';

  const res = await fetch(path, Object.assign({}, options, { headers }));

  // 401 means the token expired or was never valid -> back to the login screen.
  if (res.status === 401) {
    logout();
    throw new Error('Session expired — please sign in again.');
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res;
}

const apiJson = (path, options) => api(path, options).then((r) => r.json());

/* ------------------------------------------------------------------ */
/* Auth                                                                */
/* ------------------------------------------------------------------ */

const $ = (id) => document.getElementById(id);

document.getElementById('login-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const btn = $('login-btn');
  const err = $('login-error');
  err.hidden = true;
  btn.disabled = true;
  btn.textContent = 'Signing in…';

  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: $('username').value,
        password: $('password').value,
      }),
    });
    if (!res.ok) throw new Error('Incorrect username or password.');
    const data = await res.json();
    sessionStorage.setItem(TOKEN_KEY, data.access_token);
    $('who').textContent = data.username;
    showDashboard();
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Sign in';
  }
});

$('logout-btn').addEventListener('click', logout);

function logout() {
  sessionStorage.removeItem(TOKEN_KEY);
  clearInterval(pollTimer);
  pollTimer = null;
  history = [];
  $('dash-view').hidden = true;
  $('login-view').hidden = false;
}

async function showDashboard() {
  $('login-view').hidden = true;
  $('dash-view').hidden = false;
  await refresh();
  await loadAlerts();
  clearInterval(pollTimer);
  pollTimer = setInterval(tick, POLL_MS);
}

/* ------------------------------------------------------------------ */
/* Polling                                                             */
/* ------------------------------------------------------------------ */

let alertPollCounter = 0;

async function tick() {
  await refresh();
  // The alert table changes far less often than the readings, so refresh it
  // every 4th poll instead of every one. Fewer queries, same user experience.
  if (++alertPollCounter % 4 === 0) await loadAlerts();
}

async function refresh() {
  try {
    const snap = await apiJson(`/api/live?limit=${MAX_POINTS}`);
    $('conn-state').textContent = snap.running ? 'live' : 'simulator stopped';
    $('machine-state').textContent = snap.machine_state || '—';
    $('sim-start').disabled = snap.running;
    $('sim-stop').disabled = !snap.running;

    history = snap.readings || [];
    if (history.length) {
      renderCurrent(history[history.length - 1]);
      drawChart(history);
    }
  } catch (e) {
    $('conn-state').textContent = 'disconnected';
  }
}

/* ------------------------------------------------------------------ */
/* Rendering — current status                                          */
/* ------------------------------------------------------------------ */

const SUBTITLE = {
  Normal: 'All monitored parameters are inside their operating limits.',
  Warning: 'A parameter is approaching its limit — plan a corrective action.',
  Fault: 'A physical limit has been exceeded — act now.',
};

function renderCurrent(record) {
  const status = record.status;
  const lower = status.toLowerCase();

  $('status-dot').className = `dot dot-${lower}`;
  const banner = $('status-banner');
  banner.className = `banner banner-${lower}`;
  $('banner-status').textContent = status.toUpperCase();
  $('banner-sub').textContent = SUBTITLE[status] || '';
  $('banner-conf').textContent = `${Math.round(record.confidence * 100)}%`;

  document.title = status === 'Normal'
    ? 'Machine Health Monitor'
    : `(${status}) Machine Health Monitor`;

  // Recommended action card — only shown when there is something to do.
  const card = $('action-card');
  if (record.alert) {
    card.hidden = false;
    $('action-title').textContent = record.alert.title;
    const rules = record.alert.triggered_rules || [];
    $('action-detail').textContent = rules.length ? rules[0].detail : '';
    $('action-text').textContent = record.alert.recommended_action;
  } else {
    card.hidden = true;
  }

  renderTiles(record);
  renderRemainingLife(record.remaining_life);
}

/* Remaining useful life. The headline number is the PHYSICS estimate — see
 * backend/rul.py for why the learned model is shown only as a cross-check. */
const BINDING_LABEL = {
  tool_wear: 'tool wear (200 min life)',
  overstrain: 'overstrain at this torque',
};

function renderRemainingLife(life) {
  const card = $('rul-card');
  if (!life) {
    card.className = 'rul-card';
    return;
  }

  card.className = `rul-card is-${life.band}`;
  $('rul-min').textContent = Math.round(life.remaining_min);

  // The bar fills as life is consumed, so a full bar means "change it now".
  $('rul-fill').style.width = `${Math.round(life.fraction_consumed * 100)}%`;

  const ceiling = life.total_usable_min;
  $('rul-meta').textContent = ceiling == null
    ? 'Spindle idle — no tool life is being consumed.'
    : `${Math.round(life.fraction_consumed * 100)}% of this tool's `
      + `${Math.round(ceiling)} min usable life consumed.`;

  // Wall-clock projection. Null until enough readings exist to measure a rate —
  // we show a dash rather than inventing a deadline.
  // wear_rate_per_min is normalised: 1.0 = wearing at exactly the rate of the
  // clock, 1.6 = burning tool life 60% faster than nominal because of the load.
  const wallclock = life.wallclock_remaining_min;
  const rate = life.wear_rate_per_min;
  $('rul-wallclock').textContent = wallclock == null
    ? 'measuring…'
    : `${formatDuration(wallclock)}  (wear ${rate.toFixed(2)}× nominal)`;

  $('rul-binding').textContent = BINDING_LABEL[life.binding_constraint]
    || life.binding_constraint;

  // The model agrees to within a fraction of a minute almost everywhere; where
  // it does not, sigma is the honest signal. Show both.
  $('rul-model').textContent = life.model_remaining_min == null
    ? 'not loaded'
    : `${life.model_remaining_min} ± ${life.model_sigma_min} min`;
}

function formatDuration(minutes) {
  if (minutes < 1) return '< 1 min';
  if (minutes < 90) return `${Math.round(minutes)} min`;
  const hours = Math.floor(minutes / 60);
  return `${hours} h ${Math.round(minutes % 60)} min`;
}

/* UNITS: the API speaks SI (kelvin, watts) because that is what the model was
 * trained on — see backend/units.py. The dashboard converts to what an engineer
 * actually reads off a panel (°C, kW).
 *
 * `show` converts raw -> display. `min`/`max` are in DISPLAY units so the bar
 * fill matches the number above it. `status` deliberately takes the RAW value,
 * so the colour bands stay tied to the backend's thresholds and cannot drift
 * out of sync with a display change. */
const K_OFFSET = 273.15;

const TILES = {
  air_temp:     { show: (v) => v - K_OFFSET, min: 20, max: 35, digits: 1 },
  process_temp: { show: (v) => v - K_OFFSET, min: 30, max: 45, digits: 1 },
  // A temperature DIFFERENCE of 10 K is 10 °C — the offset cancels, so no
  // conversion here. Only the label changes.
  temp_diff:    { show: (v) => v, min: 0, max: 15, digits: 1,
                  status: (v) => (v < 8.6 ? 'fault' : v < 9.5 ? 'warning' : 'ok') },
  rot_speed:    { show: (v) => v, min: 800, max: 2800, digits: 0 },
  torque:       { show: (v) => v, min: 0, max: 90, digits: 1 },
  power:        { show: (v) => v / 1000, min: 0, max: 11, digits: 2,
                  status: (v) => (v < 3500 || v > 9000 ? 'fault'
                                : v < 4000 || v > 8500 ? 'warning' : 'ok') },
  tool_wear:    { show: (v) => v, min: 0, max: 245, digits: 0,
                  status: (v) => (v >= 240 ? 'fault' : v > 180 ? 'warning' : 'ok') },
  strain:       { show: (v) => v, min: 0, max: 14000, digits: 0,
                  status: (v) => (v > 12000 ? 'fault' : v > 10200 ? 'warning' : 'ok') },
};

function renderTiles(record) {
  const values = Object.assign({}, record.reading, record.derived);
  document.querySelectorAll('.tile').forEach((tile) => {
    const key = tile.dataset.key;
    const cfg = TILES[key];
    const raw = values[key];
    if (cfg == null || raw == null) return;

    const display = cfg.show(raw);
    tile.querySelector('.tile-value span').textContent =
      display.toFixed(cfg.digits);

    const pct = Math.max(0, Math.min(100,
      ((display - cfg.min) / (cfg.max - cfg.min)) * 100));
    tile.querySelector('.tile-bar i').style.width = `${pct}%`;

    // Status bands are evaluated on the RAW SI value, matching the backend.
    const state = cfg.status ? cfg.status(raw) : 'ok';
    tile.classList.toggle('is-warning', state === 'warning');
    tile.classList.toggle('is-fault', state === 'fault');
  });
}

/* ------------------------------------------------------------------ */
/* Rendering — trend chart (hand-drawn on a canvas)                    */
/* ------------------------------------------------------------------ */

/* `from` names the sub-object on each record that holds the value. Remaining
 * life is plotted instead of raw tool wear because it is the decision-relevant
 * quantity: it already accounts for torque lowering the ceiling, so a heavy cut
 * makes this line drop faster AND from a lower starting point. */
const SERIES = [
  { key: 'power',         from: 'derived',        colour: '#4c8dd6', min: 2000, max: 11000 },
  { key: 'temp_diff',     from: 'derived',        colour: '#58c0b0', min: 6,    max: 14 },
  { key: 'remaining_min', from: 'remaining_life', colour: '#c07fd6', min: 0,    max: 205 },
];

const STATUS_COLOUR = { Normal: '#2e9e5b', Warning: '#e0a800', Fault: '#e0463f' };

function drawChart(records) {
  const canvas = $('chart');
  const ctx = canvas.getContext('2d');

  // Match the canvas backing store to the CSS size AND the device pixel ratio,
  // otherwise lines look blurry on a retina screen.
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = 220;
  if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const pad = { top: 12, right: 12, bottom: 16, left: 12 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  // Horizontal gridlines
  ctx.strokeStyle = '#282d38';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  }

  if (records.length < 2) return;
  const stepX = plotW / (records.length - 1);

  // Each series is normalised into the same 0..1 plot area, because power (W),
  // ΔT (K) and wear (min) have wildly different magnitudes and units. This is a
  // shape comparison, not a value comparison — the tiles show exact values.
  SERIES.forEach((series) => {
    ctx.beginPath();
    ctx.strokeStyle = series.colour;
    ctx.lineWidth = 1.8;
    records.forEach((rec, i) => {
      const raw = (rec[series.from] || {})[series.key];
      if (raw == null) return;
      const norm = Math.max(0, Math.min(1,
        (raw - series.min) / (series.max - series.min)));
      const x = pad.left + stepX * i;
      const y = pad.top + plotH * (1 - norm);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  // A status strip under the chart: one bar per reading, coloured by verdict.
  // This makes the Normal -> Warning -> Fault progression readable at a glance.
  const strip = $('chart-status');
  strip.innerHTML = '';
  records.forEach((rec) => {
    const bar = document.createElement('i');
    bar.style.background = STATUS_COLOUR[rec.status] || '#8b93a5';
    bar.title = `${rec.timestamp} — ${rec.status}`;
    strip.appendChild(bar);
  });
}

window.addEventListener('resize', () => { if (history.length) drawChart(history); });

/* ------------------------------------------------------------------ */
/* Alert history                                                       */
/* ------------------------------------------------------------------ */

async function loadAlerts() {
  try {
    const alerts = await apiJson('/api/alerts?limit=40');
    const body = $('alert-body');
    $('alert-count').textContent = alerts.length
      ? `${alerts.length} most recent` : '';

    if (!alerts.length) {
      body.innerHTML =
        '<tr><td colspan="5" class="muted center">No alerts yet.</td></tr>';
      return;
    }

    body.innerHTML = '';
    alerts.forEach((a) => {
      const rules = Array.isArray(a.triggered_rules) ? a.triggered_rules : [];
      const tr = document.createElement('tr');
      tr.appendChild(cell(a.timestamp.slice(11, 19), 'ts'));
      tr.appendChild(cell(a.severity, `sev sev-${a.severity}`));
      tr.appendChild(cell(a.title));
      tr.appendChild(cell(rules.length ? rules[0].detail : '—', 'detail'));
      tr.appendChild(cell(a.recommended_action));
      body.appendChild(tr);
    });
  } catch (_) { /* transient — the next poll retries */ }
}

/* textContent, never innerHTML: alert text originates from the backend, and
 * building rows as text means a stray "<" can never become markup. */
function cell(text, className) {
  const td = document.createElement('td');
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

/* ------------------------------------------------------------------ */
/* Controls                                                            */
/* ------------------------------------------------------------------ */

function say(message, isError) {
  const el = $('control-msg');
  el.textContent = message;
  el.style.color = isError ? 'var(--fault)' : 'var(--muted)';
}

$('sim-start').addEventListener('click', async () => {
  try { await api('/api/simulator/start', { method: 'POST' }); say('Simulator started.'); }
  catch (e) { say(e.message, true); }
  refresh();
});

$('sim-stop').addEventListener('click', async () => {
  try { await api('/api/simulator/stop', { method: 'POST' }); say('Simulator stopped.'); }
  catch (e) { say(e.message, true); }
  refresh();
});

document.querySelectorAll('[data-inject]').forEach((btn) => {
  btn.addEventListener('click', async () => {
    try {
      const res = await apiJson(`/api/simulator/inject/${btn.dataset.inject}`,
                                { method: 'POST' });
      say(res.message);
    } catch (e) { say(e.message, true); }
  });
});

/* Downloads go through fetch(), not a plain <a href>, because a link cannot
 * carry the Authorization header. We pull the file into a Blob and click a
 * temporary object-URL link instead. */
async function download(path, fallbackName) {
  say('Generating report…');
  try {
    const res = await api(path);
    const blob = await res.blob();

    const disposition = res.headers.get('Content-Disposition') || '';
    const match = disposition.match(/filename="?([^"]+)"?/);
    const filename = match ? match[1] : fallbackName;

    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    say(`Downloaded ${filename}`);
  } catch (e) {
    say(e.message, true);
  }
}

$('dl-csv').addEventListener('click', () => download('/api/report/csv', 'report.csv'));
$('dl-pdf').addEventListener('click', () => download('/api/report/pdf', 'report.pdf'));

/* ------------------------------------------------------------------ */
/* Boot — restore an existing session if the token is still valid       */
/* ------------------------------------------------------------------ */

(async function boot() {
  if (!getToken()) return;
  try {
    const user = await apiJson('/api/auth/me');
    $('who').textContent = user.username;
    showDashboard();
  } catch (_) {
    logout();
  }
})();
