// script.js -- talks to the FastAPI backend in app.py.
// Per latest requirements: Predict panel and Event Log removed (not
// adding value / not fetching reliably); replaced with a percentage-
// based 6-month device forecast and a red-zone area risk panel.
// Every field name here is verified against real, live-run backend
// responses -- not guessed.

const state = { stats: null };

function apiUrl(path) {
  return new URL(API_BASE_URL + path, window.location.origin);
}

async function apiGet(path, params = {}) {
  const url = apiUrl(path);
  Object.entries(params).forEach(([k, v]) => { if (v !== "" && v != null) url.searchParams.set(k, v); });
  let res;
  try {
    res = await fetch(url);
  } catch (networkErr) {
    throw new Error(`Cannot reach the API at ${url.origin} -- is the backend running? (uvicorn app:app --port 8000)`);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

function fmtMin(n) {
  if (n == null || isNaN(n)) return "—";
  return Math.round(n).toLocaleString() + " min";
}
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function loadStats() {
  const data = await apiGet("/stats");
  state.stats = data;
  renderKpis(data);
  renderParetoBars(data);
  renderSeverityTable(data);
  renderLowSampleTable(data);
  populateForecastDeviceSelect(data);
}

// loadModelInfo() removed per user request -- the "MODEL: CATBOOST ·
// HOLDOUT ACCURACY..." footer line is no longer shown.

async function loadForecast() {
  const data = await apiGet("/forecast/all_devices", { months: 6 });
  renderForecastBars(data);
  return data;
}

async function loadAreaBreakdown() {
  const data = await apiGet("/device_area_breakdown", { top_n: 12 });
  renderAreaTable(data);
  return data;
}

async function loadAreaRisk() {
  const data = await apiGet("/area_risk", { top_n: 6 });
  renderAreaRisk(data);
  return data;
}

async function loadReviewQueue() {
  const data = await apiGet("/review_queue", { limit: 30 });
  renderReviewQueue(data);
  return data;
}

async function loadShadowStats() {
  const data = await apiGet("/shadow_stats");
  renderShadowStats(data);
  return data;
}

// NEW: device health score / recurrence risk -- follows the exact same
// loadX()/renderX() pattern as loadReviewQueue()/loadShadowStats()
// above, using apiGet() like every other GET call in this file.
async function loadHealthScore() {
  const data = await apiGet("/health_score", { top_n: 10 });
  renderHealthScore(data);
  return data;
}

// --------------------------------------------------- Shadow review form --
let shadowCurrentId = null;

async function populateShadowDropdowns() {
  try {
    const data = await apiGet("/devices");
    const sel = document.getElementById("shadowActualSelect");
    (data.target_devices || []).forEach(dev => {
      const opt = document.createElement("option");
      opt.value = dev;
      opt.textContent = formatDevice(dev);
      sel.appendChild(opt);
    });
  } catch (e) {
    // non-critical -- form still usable, dropdown just stays empty
  }
}

function populateShadowAreaOptions(breakdown) {
  const areas = [...new Set((breakdown?.top_device_area_combinations || []).map(r => r.area))].sort();
  const sel = document.getElementById("shadowAreaInput");
  areas.forEach(a => {
    const opt = document.createElement("option");
    opt.value = a; opt.textContent = a;
    sel.appendChild(opt);
  });
}

async function shadowLogDelay() {
  const reasonText = document.getElementById("shadowReasonInput").value.trim();
  const area = document.getElementById("shadowAreaInput").value || null;
  const minsVal = document.getElementById("shadowMinsInput").value;
  const errorEl = document.getElementById("shadowStep1Error");
  const btn = document.getElementById("shadowLogBtn");
  errorEl.innerHTML = "";

  if (reasonText.length < 3) {
    errorEl.innerHTML = `<div class="error-banner">Enter a delay description (at least 3 characters).</div>`;
    return;
  }

  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Logging…`;
  try {
    const payload = { reason_text: reasonText, area };
    if (minsVal !== "") payload.mins = parseFloat(minsVal);
    // POST directly -- apiGet is GET-only, this needs its own POST call
    const res = await fetch(apiUrl("/shadow_predict"), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    const result = await res.json();
    shadowCurrentId = result.shadow_id;

    document.getElementById("shadowStep1").style.display = "none";
    document.getElementById("shadowStep2").style.display = "block";
  } catch (e) {
    errorEl.innerHTML = `<div class="error-banner">${escapeHtml(e.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Log This Delay";
  }
}

async function shadowResolveDelay() {
  const actualDevice = document.getElementById("shadowActualSelect").value;
  const note = document.getElementById("shadowNoteInput").value.trim();
  const errorEl = document.getElementById("shadowStep2Error");
  const btn = document.getElementById("shadowResolveBtn");
  errorEl.innerHTML = "";

  if (!actualDevice) {
    errorEl.innerHTML = `<div class="error-banner">Select what you actually found.</div>`;
    return;
  }
  if (!shadowCurrentId) {
    errorEl.innerHTML = `<div class="error-banner">No delay logged yet — start over.</div>`;
    return;
  }

  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Comparing…`;
  try {
    const res = await fetch(apiUrl("/shadow_resolve"), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ shadow_id: shadowCurrentId, actual_device: actualDevice, reviewer_note: note || null }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    const result = await res.json();
    renderShadowResult(result);
    await loadShadowStats(); // refresh the running agreement-rate panel
  } catch (e) {
    errorEl.innerHTML = `<div class="error-banner">${escapeHtml(e.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Record My Finding & Compare";
  }
}

function renderShadowResult(r) {
  document.getElementById("shadowStep2").style.display = "none";
  const dot = document.getElementById("shadowResultDot");
  const outcome = document.getElementById("shadowResultOutcome");
  dot.style.background = r.agreement ? "var(--teal)" : "var(--accent)";
  outcome.textContent = r.agreement ? "Model agreed" : "Model disagreed";
  document.getElementById("shadowResultPredicted").textContent = formatDevice(r.predicted_device);
  document.getElementById("shadowResultActual").textContent = formatDevice(r.actual_device);
  document.getElementById("shadowResultBlock").classList.add("show");
}

function shadowStartNewReview() {
  shadowCurrentId = null;
  document.getElementById("shadowReasonInput").value = "";
  document.getElementById("shadowAreaInput").value = "";
  document.getElementById("shadowMinsInput").value = "";
  document.getElementById("shadowActualSelect").value = "";
  document.getElementById("shadowNoteInput").value = "";
  document.getElementById("shadowStep1Error").innerHTML = "";
  document.getElementById("shadowStep2Error").innerHTML = "";
  document.getElementById("shadowResultBlock").classList.remove("show");
  document.getElementById("shadowStep2").style.display = "none";
  document.getElementById("shadowStep1").style.display = "block";
}

async function loadUnresolvedHotspots() {
  return apiGet("/unresolved_hotspots", { top_n: 3 });
}

function renderKpis(d) {
  document.getElementById("kpiTotalMinutes").textContent = fmtMin(d.total_field_device_delay_minutes);
  document.getElementById("kpiTotalMinutesFoot").textContent =
    `${d.months_covered.length} months · ${d.months_covered[0]}–${d.months_covered[d.months_covered.length-1]}`;
  document.getElementById("kpiTaggedEvents").textContent = d.pareto_by_device.length;
  document.getElementById("kpiTaggedEventsFoot").textContent = `${d.total_tagged_events.toLocaleString()} tracked events`;

  if (d.top_contributor_volume) {
    document.getElementById("kpiTopVolume").textContent = formatDevice(d.top_contributor_volume.device);
    document.getElementById("kpiTopVolumeFoot").textContent =
      `${fmtMin(d.top_contributor_volume.total_minutes)} · ${d.top_contributor_volume.events} events`;
  }
  if (d.top_contributor_severity) {
    document.getElementById("kpiTopSeverity").textContent = formatDevice(d.top_contributor_severity.device);
    document.getElementById("kpiTopSeverityFoot").textContent =
      `${d.top_contributor_severity.avg_minutes.toFixed(1)} min avg per incident`;
  }
}

function renderParetoBars(d) {
  const rows = [...d.pareto_by_device].sort((a, b) => b.total_minutes - a.total_minutes);
  const max = rows.length ? rows[0].total_minutes : 1;
  document.getElementById("paretoBars").innerHTML = rows.map(r => `
    <div class="bar-row">
      <span class="name">${formatDevice(r.device)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(r.total_minutes / max * 100).toFixed(1)}%; background:${deviceColor(r.device)}"></div></div>
      <span class="val">${Math.round(r.total_minutes)}</span>
    </div>
  `).join("");
}

function renderForecastBars(d) {
  const rows = d.ranked_by_expected_delay || [];
  const total = rows.reduce((sum, r) => sum + r.expected_minutes, 0);
  document.getElementById("forecastBars").innerHTML = rows.map(r => {
    const pct = total > 0 ? (r.expected_minutes / total * 100) : 0;
    return `
    <div class="bar-row">
      <span class="name">${formatDevice(r.device)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${pct.toFixed(1)}%; background:${deviceColor(r.device)}"></div></div>
      <span class="val">${pct.toFixed(1)}%</span>
    </div>`;
  }).join("");

  const top = rows[0];
  document.getElementById("forecastFootnote").textContent = top && total > 0
    ? `${formatDevice(top.device)} is expected to account for the largest share of delay over the next 6 months (~${((top.expected_minutes/total)*100).toFixed(1)}% of total predicted field-device delay).`
    : "";
}

function renderAreaRisk(d) {
  const rows = d.top_risk_areas || [];
  const max = rows.length ? rows[0].total_minutes : 1;
  document.getElementById("areaRiskBars").innerHTML = rows.map(r => `
    <div class="bar-row">
      <span class="name">${escapeHtml(r.area)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(r.total_minutes / max * 100).toFixed(1)}%; background:var(--accent)"></div></div>
      <span class="val">${r.pct_share}%</span>
    </div>
  `).join("");
}

function renderReviewQueue(d) {
  const rows = d.queue || [];
  document.getElementById("reviewQueueNote").textContent =
    `${d.total_unresolved} unresolved delays · ${Math.round(d.total_unresolved_minutes).toLocaleString()} min total · showing top ${rows.length} by impact`;
  document.getElementById("reviewQueueBody").innerHTML = rows.length ? rows.map(r => `
    <tr>
      <td class="mono">${r.date || "—"}</td>
      <td>${escapeHtml(r.area)}</td>
      <td class="mins">${r.mins != null ? Math.round(r.mins) : "—"}</td>
      <td class="reason">${escapeHtml(r.reason_text || "")}</td>
    </tr>
  `).join("") : `<tr><td colspan="4" class="text-dim">Nothing to review.</td></tr>`;
}

function renderShadowStats(d) {
  const el = document.getElementById("shadowStatsBody");
  if (!d.total_resolved) {
    el.innerHTML = `<p class="text-dim">No resolved shadow-mode entries yet. Use /shadow_predict and /shadow_resolve to begin tracking agreement before this system drives unsupervised decisions.</p>`;
    return;
  }
  const byDeviceRows = (d.by_device || []).map(r => `
    <div class="bar-row">
      <span class="name">${formatDevice(r.predicted_device)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${r.agreement_rate}%; background:${r.agreement_rate >= 70 ? 'var(--teal)' : 'var(--accent)'}"></div></div>
      <span class="val">${r.agreement_rate}% (n=${r.n})</span>
    </div>
  `).join("");
  el.innerHTML = `
    <div class="meta-card" style="margin-bottom:14px;">
      <p class="mk-label">Overall Agreement Rate</p>
      <p class="mk-value mk-value-lg">${d.overall_agreement_rate}%</p>
      <p class="fc-note">${d.total_resolved} resolved · ${d.total_pending} pending review</p>
    </div>
    ${byDeviceRows}
  `;
}

// NEW: device health score / recurrence risk table -- ranked list of
// device+area combinations most likely to fail again soon. Follows the
// exact same rendering pattern as renderReviewQueue()/renderShadowStats()
// above. Field names verified against the real /health_score response
// shape (top_at_risk[], survival_model_available, note).
function renderHealthScore(d) {
  const tbody = document.getElementById("healthScoreBody");
  const noteEl = document.getElementById("healthScoreNote");
  const noteTextEl = document.getElementById("healthScoreNoteText");

  if (noteEl && noteTextEl) {
    if (d.note) {
      noteTextEl.textContent = d.note;
      noteEl.style.display = "block";
    } else {
      noteEl.style.display = "none";
    }
  }

  const rows = d.top_at_risk || [];
  tbody.innerHTML = rows.length ? rows.map(r => {
    const scoreColor = r.health_score >= 75 ? "var(--accent)" : r.health_score >= 50 ? "#f0ad4e" : "var(--teal)";
    const warningBadge = r.low_sample_warning
      ? ` <span title="Fewer than 5 total events -- treat this score cautiously" style="color:#f0ad4e;">&#9888;</span>`
      : "";
    const survivalCell = (d.survival_model_available && r.survival_model_14d_recurrence_pct != null)
      ? `${r.survival_model_14d_recurrence_pct}%`
      : `<span class="text-dim">—</span>`;
    return `
      <tr>
        <td><span class="device-chip"><span class="device-dot" style="background:${deviceColor(r.device)}"></span>${formatDevice(r.device)}</span>${warningBadge}</td>
        <td>${escapeHtml(r.area)}</td>
        <td class="num" style="color:${scoreColor}; font-weight:bold;">${r.health_score.toFixed(1)}</td>
        <td class="num">${r.days_since_last_event}</td>
        <td class="num">${r.events_last_90d}</td>
        <td class="num">${r.total_events}</td>
        <td class="num">${survivalCell}</td>
      </tr>
    `;
  }).join("") : `<tr><td colspan="7" class="text-dim">No health score data available yet.</td></tr>`;
}

// NEW: populate the "Check 6-Month Forecast" dropdown with every
// reliable field device from the current Pareto data.
function populateForecastDeviceSelect(statsData) {
  const sel = document.getElementById("forecastDeviceSelect");
  const existing = new Set(Array.from(sel.options).map(o => o.value));
  (statsData.pareto_by_device || []).forEach(r => {
    if (!existing.has(r.device)) {
      const opt = document.createElement("option");
      opt.value = r.device;
      opt.textContent = formatDevice(r.device);
      sel.appendChild(opt);
    }
  });
}

// NEW: on-demand single-device 6-month forecast check.
async function checkDeviceForecast() {
  const device = document.getElementById("forecastDeviceSelect").value;
  const resultEl = document.getElementById("forecastCheckResult");
  if (!device) return;
  resultEl.innerHTML = `<p class="text-dim">Checking…</p>`;
  try {
    const data = await apiGet("/forecast", { device, months: 6 });
    resultEl.innerHTML = `
      <div class="forecast-check-card">
        <p class="fc-device">${formatDevice(device)}</p>
        <p class="fc-value">${fmtMin(data.recommended_forecast)}</p>
        <p class="fc-note">Predicted over the next 6 months.</p>
      </div>
    `;
  } catch (e) {
    resultEl.innerHTML = `<p style="color:var(--accent);">${escapeHtml(e.message)}</p>`;
  }
}

// NEW: render the low-sample-size devices table (RFID/TT/HIP etc.) --
// shown separately, never mixed into the main reliable-device ranking.
function renderLowSampleTable(d) {
  const rows = d.pareto_low_sample_devices || [];
  const tbody = document.getElementById("lowSampleTableBody");
  if (!tbody) return;
  tbody.innerHTML = rows.length ? rows.map(r => `
    <tr>
      <td><span class="device-chip"><span class="device-dot" style="background:${deviceColor(r.device)}"></span>${formatDevice(r.device)}</span></td>
      <td class="num">${r.avg_minutes.toFixed(1)}</td>
      <td class="num">${Math.round(r.max_minutes)} min</td>
      <td class="num">${r.events}</td>
      <td><span class="risk-badge" style="background:${riskColor(r.risk_level)}22; color:${riskColor(r.risk_level)};">${r.risk_level}${r.is_estimated ? " · no sheet" : ""}</span></td>
    </tr>
  `).join("") : `<tr><td colspan="5" class="text-dim">None.</td></tr>`;
}

let trendChartInstance = null;
function renderTrendChart(forecastData) {
  const canvas = document.getElementById("trendChart");
  if (!canvas || typeof Chart === "undefined") return;

  const forecasts = forecastData.forecasts || {};
  const topDevices = (forecastData.ranked_by_expected_delay || []).slice(0, 5).map(r => r.device);
  if (topDevices.length === 0) return;

  const monthSet = new Set();
  topDevices.forEach(dev => {
    (forecasts[dev]?.monthly_totals || []).forEach(m => monthSet.add(m.month));
  });
  const labels = sortMonthsChronologically([...monthSet]);

  const datasets = topDevices.map(dev => {
    const series = forecasts[dev]?.monthly_totals || [];
    const byMonth = Object.fromEntries(series.map(s => [s.month, s.mins]));
    return {
      label: formatDevice(dev),
      data: labels.map(m => byMonth[m] ?? 0),
      borderColor: deviceColor(dev),
      backgroundColor: deviceColor(dev) + "33",
      tension: 0.35,
      pointRadius: 3,
      borderWidth: 2,
    };
  });

  if (trendChartInstance) trendChartInstance.destroy();
  trendChartInstance = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "bottom", labels: { color: "#8194A6", boxWidth: 10, font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: "#5A6C7E" }, grid: { color: "#182230" } },
        y: { ticks: { color: "#5A6C7E" }, grid: { color: "#182230" } },
      },
    },
  });
}

// "Nov-25" -> 202511. Parsed rather than a hardcoded month list, which
// went stale (any month after Jul-26 sorted as unknown).
const MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
function monthKey(label) {
  const [mon, yy] = String(label).split("-");
  const mi = MONTH_ABBR.indexOf(mon);
  return (mi === -1 || isNaN(+yy)) ? Infinity : (2000 + +yy) * 100 + mi + 1;
}
function sortMonthsChronologically(months) {
  return [...months].sort((a, b) => monthKey(a) - monthKey(b));
}

function renderAreaTable(d) {
  const rows = d.top_device_area_combinations || [];
  document.getElementById("areaTableBody").innerHTML = rows.length ? rows.map(r => `
    <tr>
      <td><span class="device-chip"><span class="device-dot" style="background:${deviceColor(r.device)}"></span>${formatDevice(r.device)}</span></td>
      <td>${escapeHtml(r.area)}</td>
      <td class="num">${Math.round(r.total_minutes)}</td>
      <td class="num">${r.avg_minutes.toFixed(1)}</td>
      <td class="num">${r.events}</td>
    </tr>
  `).join("") : `<tr><td colspan="5" class="text-dim">No area data yet.</td></tr>`;
}

function renderSeverityTable(d) {
  const rows = [...d.pareto_by_device].sort((a, b) => b.avg_minutes - a.avg_minutes);
  document.getElementById("sevTableBody").innerHTML = rows.map(r => `
    <tr>
      <td><span class="device-chip"><span class="device-dot" style="background:${deviceColor(r.device)}"></span>${formatDevice(r.device)}</span></td>
      <td class="num">${r.avg_minutes.toFixed(1)}</td>
      <td class="num">${Math.round(r.max_minutes)} min</td>
      <td class="num">${r.events}</td>
      <td><span class="risk-badge" style="background:${riskColor(r.risk_level)}22; color:${riskColor(r.risk_level)};">${r.risk_level}${r.is_estimated ? " · no sheet" : ""}</span></td>
      <td style="font-size:11.5px; color:var(--text-muted); max-width:260px;">${escapeHtml(r.recommendation || "")}</td>
    </tr>
  `).join("");
}

function renderAlerts(stats, forecast, areaBreakdown, areaRisk, hotspots) {
  const alerts = [];
  const rows = stats.pareto_by_device;
  const bySeverity = [...rows].sort((a, b) => b.avg_minutes - a.avg_minutes);
  const gaps = rows.filter(r => r.is_estimated);
  const topForecast = (forecast.ranked_by_expected_delay || [])[0];
  const forecastTotal = (forecast.ranked_by_expected_delay || []).reduce((s, r) => s + r.expected_minutes, 0);
  const topAreaCombo = (areaBreakdown.top_device_area_combinations || [])[0];
  const topRiskAreas = (areaRisk.top_risk_areas || []).slice(0, 2);
  const topHotspot = (hotspots?.top_unresolved_hotspots || [])[0];

  if (topRiskAreas.length) {
    alerts.push({
      type: "critical",
      title: `Red-zone areas: ${topRiskAreas.map(a => a.area).join(", ")}`,
      body: `${topRiskAreas.map(a => `${a.area} (${a.pct_share}%)`).join(" and ")} together account for the largest share of current delay — prioritize maintenance attention here first.`,
    });
  }
  if (topForecast && forecastTotal > 0) {
    const pct = (topForecast.expected_minutes / forecastTotal * 100).toFixed(1);
    alerts.push({
      type: "warning",
      title: `${formatDevice(topForecast.device)} — highest predicted delay share, next 6 months`,
      body: `Expected to account for <span class="pct">~${pct}%</span> of total field-device delay over the next 6 months. This is where the 50% reduction effort should focus first.`,
    });
  }
  if (topAreaCombo) {
    alerts.push({
      type: "warning",
      title: `${formatDevice(topAreaCombo.device)} @ ${topAreaCombo.area} — highest combined risk`,
      body: `${Math.round(topAreaCombo.total_minutes)} min across ${topAreaCombo.events} events — the single worst device+area combination.`,
    });
  }
  if (bySeverity[0]) {
    alerts.push({
      type: "info",
      title: `${formatDevice(bySeverity[0].device)} — highest severity per incident`,
      body: `Averages ${bySeverity[0].avg_minutes.toFixed(1)} min per event` +
        (bySeverity[0].rpn ? ` and carries FMEA RPN ${bySeverity[0].rpn} (${bySeverity[0].risk_level}).` : `, with no FMEA sheet on file.`),
    });
  }
  if (gaps.length) {
    alerts.push({
      type: "info",
      title: `FMEA gap: ${gaps.map(g => formatDevice(g.device)).join(", ")}`,
      body: `${gaps.length} device(s) have no FMEA sheet -- reported as UNKNOWN risk, never fabricated as "Low".`,
    });
  }
  if (topHotspot) {
    alerts.push({
      type: "info",
      title: `Untagged delay hotspot: ${topHotspot.area}`,
      body: `${Math.round(topHotspot.total_minutes)} min across ${topHotspot.events} events in this area could not be auto-matched to a specific field device — review this area's delay logging to close the gap.`,
    });
  }

  document.getElementById("alertsContainer").innerHTML = alerts.map(a => `
    <div class="alert ${a.type}"><p class="alert-title">${a.title}</p><p class="alert-body">${a.body}</p></div>
  `).join("");
}

async function init() {
  setupTabs();
  document.getElementById("checkForecastBtn").addEventListener("click", checkDeviceForecast);
  document.getElementById("shadowLogBtn").addEventListener("click", shadowLogDelay);
  document.getElementById("shadowResolveBtn").addEventListener("click", shadowResolveDelay);
  document.getElementById("shadowNewReviewBtn").addEventListener("click", shadowStartNewReview);
  populateShadowDropdowns();

  let statsOk = false, areaBreakdown = null, forecast = null, areaRisk = null;

  try {
    await loadStats();
    statsOk = true;
  } catch (e) {
    showPanelError("alertsContainer", `/stats failed: ${e.message}`);
  }

  try {
    areaBreakdown = await loadAreaBreakdown();
    populateShadowAreaOptions(areaBreakdown);
  } catch (e) {
    showPanelError("areaTableBody", `/device_area_breakdown failed: ${e.message}`, 5);
  }

  try {
    forecast = await loadForecast();
    renderTrendChart(forecast);
  } catch (e) {
    showPanelError("forecastBars", `/forecast/all_devices failed: ${e.message}`);
  }

  try {
    areaRisk = await loadAreaRisk();
  } catch (e) {
    showPanelError("areaRiskBars", `/area_risk failed: ${e.message}`);
  }

  try {
    await loadReviewQueue();
  } catch (e) {
    showPanelError("reviewQueueBody", `/review_queue failed: ${e.message}`, 4);
  }

  try {
    await loadShadowStats();
  } catch (e) {
    showPanelError("shadowStatsBody", `/shadow_stats failed: ${e.message}`);
  }

  // NEW: health score panel -- same try/catch pattern as every other
  // panel above, so a failure here never blocks the rest of the page.
  try {
    await loadHealthScore();
  } catch (e) {
    showPanelError("healthScoreBody", `/health_score failed: ${e.message}`, 7);
  }

  let hotspots = null;
  try {
    hotspots = await loadUnresolvedHotspots();
  } catch (e) {
    hotspots = null; // non-critical -- alerts still render without it
  }

  if (statsOk) {
    try {
      renderAlerts(
        state.stats,
        forecast || { ranked_by_expected_delay: [] },
        areaBreakdown || { top_device_area_combinations: [] },
        areaRisk || { top_risk_areas: [] },
        hotspots
      );
    } catch (e) {
      showPanelError("alertsContainer", `Could not render alerts: ${e.message}`);
    }
  }
}

// ------------------------------------------------------------- Tabs --
// Fleet / plant tabs load lazily on first open: keeps the Combi Mill
// view fast, and Chart.js can't size a canvas inside a hidden tab.
const tabLoaders = { "tab-fleet": loadFleetTab, "tab-plants": loadPlantsTab };
const tabsLoaded = new Set(["tab-mill"]);

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => showTab(btn.dataset.tab));
  });
  // Hash is "#fleet", not "#tab-fleet" -- a hash matching an element id
  // makes the browser jump-scroll past the header on load.
  const fromHash = "tab-" + location.hash.replace("#", "");
  if (document.getElementById(fromHash)) showTab(fromHash);
}

function showTab(id) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === id));
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.toggle("active", p.id === id));
  history.replaceState(null, "", "#" + id.replace("tab-", ""));
  if (!tabsLoaded.has(id) && tabLoaders[id]) {
    tabsLoaded.add(id);
    tabLoaders[id]();
  }
}

// ------------------------------------------------------- Power fleet --
const fmtInt = n => (n == null || isNaN(n)) ? "—" : Math.round(n).toLocaleString();
const fmtPct = n => (n == null || isNaN(n)) ? "—" : `${(+n).toFixed(1)}%`;
const MODE_LABELS = {
  CALIBRATION_DRIFT: "calibration / drift", WIRING_CONNECTION: "wiring / connection",
  MOISTURE_CORROSION: "moisture / corrosion", MECHANICAL_VIBRATION: "mechanical / vibration",
  POWER_SUPPLY: "power supply", COMPONENT_AGING: "aging", MAINTENANCE_HUMAN_ERROR: "maintenance / human error",
  CONTAMINATION_BLOCKAGE: "contamination / blockage",
};

async function loadFleetTab() {
  try {
    const o = await apiGet("/fleet/overview");
    if (!o.available) {
      showPanelError("lessonGrid", o.message || "No fleet data available.");
      return;
    }
    document.getElementById("fleetKpiEvents").textContent = fmtInt(o.total_events);
    document.getElementById("fleetKpiEventsFoot").textContent = `${o.first_date} → ${o.last_date}`;
    document.getElementById("fleetKpiDevice").textContent = fmtInt(o.field_device_events);
    document.getElementById("fleetKpiDeviceFoot").textContent = `${fmtInt(o.field_device_trips)} caused a reactor/turbine trip`;
    document.getElementById("fleetKpiPlants").textContent = fmtInt(o.plants);
    document.getElementById("fleetKpiPlantsFoot").textContent = `US power reactor sites · ${o.states} states`;
    document.getElementById("fleetKpiGrid").textContent = fmtInt(o.grid_related_events);
    document.getElementById("fleetKpiGridFoot").textContent = `${fmtInt(o.reactor_trips)} total unit trips in fleet`;
    document.getElementById("fleetLabelNote").textContent = o.label_method + " Source: US NRC Event Notification Reports.";
  } catch (e) {
    showPanelError("lessonGrid", `/fleet/overview failed: ${e.message}`);
    return;
  }

  const jobs = [
    ["/fleet/lessons", renderLessons, "lessonGrid"],
    ["/fleet/device_ranking", renderFleetRanking, "fleetRankBody", 7],
    ["/fleet/trend", renderFleetTrend, "fleetTrendNote"],
    ["/fleet/grid", renderGrid, "gridCauseBars"],
    ["/fleet/plant_reliability", renderPlantReliability, "plantRelBody", 6],
  ];
  await Promise.all(jobs.map(async ([path, fn, errId, colspan]) => {
    try { fn(await apiGet(path)); }
    catch (e) { showPanelError(errId, `${path} failed: ${e.message}`, colspan); }
  }));
}

function renderLessons(d) {
  const el = document.getElementById("lessonGrid");
  const rows = d.lessons || [];
  el.innerHTML = rows.length ? rows.map(r => `
    <div class="lesson-card">
      <h3><span class="device-dot" style="background:${deviceColor(r.device)}"></span>${formatDevice(r.device)}</h3>
      <p class="lesson-stats">Mill: ${fmtInt(r.home_events)} events · ${fmtInt(r.home_minutes)} min<br>
        Fleet: ${fmtInt(r.fleet_events)} failures at ${fmtInt(r.fleet_plants_affected)} plants · ${fmtPct(r.fleet_trip_rate_pct)} tripped the unit</p>
      <div class="mode-chips">${(r.top_failure_modes || []).map(m =>
        `<span class="mode-chip">${escapeHtml(MODE_LABELS[m.mode] || m.mode)} ${m.pct}%</span>`).join("") ||
        `<span class="text-dim">no failure mode stated in narratives</span>`}</div>
      ${r.recommended_actions?.length ? `<ul>${r.recommended_actions.map(a => `<li>${escapeHtml(a)}</li>`).join("")}</ul>` : ""}
    </div>`).join("") : `<p class="text-dim">No devices shared between the mill and the fleet data yet.</p>`;
}

function renderFleetRanking(d) {
  const rows = (d.devices || []).filter(r => r.fleet_events > 0 || r.home_events > 0);
  document.getElementById("fleetRankBody").innerHTML = rows.map(r => `
    <tr>
      <td><span class="device-chip"><span class="device-dot" style="background:${deviceColor(r.device)}"></span>${formatDevice(r.device)}</span>${r.low_sample && r.fleet_events ? ` <span title="Fewer than 5 fleet events -- treat rates as indicative only" style="color:#f0ad4e;">&#9888;</span>` : ""}</td>
      <td class="num">${fmtInt(r.fleet_events)}</td>
      <td class="num">${fmtInt(r.plants_affected)}</td>
      <td class="num">${fmtPct(r.trip_rate_pct)}</td>
      <td class="num">${r.avg_power_lost_pct != null ? fmtPct(r.avg_power_lost_pct) : "—"}</td>
      <td class="num">${r.home_events ? fmtInt(r.home_events) : "—"}</td>
      <td class="num">${r.home_minutes ? fmtInt(r.home_minutes) : "—"}</td>
    </tr>`).join("");
}

let fleetChart = null;
function renderFleetTrend(d) {
  const canvas = document.getElementById("fleetTrendChart");
  document.getElementById("fleetTrendNote").textContent = d.note || "";
  if (!canvas || typeof Chart === "undefined") return;
  // Fleet-level totals, not per-device lines: keyword-labeled device
  // failures are only a handful per year, too sparse to read as a trend.
  const line = (label, data, color) => ({
    label, data, borderColor: color, backgroundColor: color + "33", tension: 0.3, pointRadius: 2, borderWidth: 2,
  });
  const datasets = [
    line("All reported events", d.total_events, "#4A90E2"),
    line("Reactor / turbine trips", d.reactor_trips, "#F4623A"),
    line("Grid / offsite-power events", d.grid_events, "#3ED6A0"),
  ];
  if (fleetChart) fleetChart.destroy();
  fleetChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { labels: d.years, datasets },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "bottom", labels: { color: "#8194A6", boxWidth: 10, font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: "#5A6C7E" }, grid: { color: "#182230" } },
        y: { ticks: { color: "#5A6C7E" }, grid: { color: "#182230" }, beginAtZero: true },
      },
    },
  });
}

function renderGrid(d) {
  const causes = d.causes || [];
  const max = causes.length ? causes[0].events : 1;
  document.getElementById("gridNote").textContent =
    `${fmtInt(d.events)} events at ${fmtInt(d.plants_affected)} plants · ${fmtPct(d.trip_rate_pct)} tripped the unit`;
  document.getElementById("gridCauseBars").innerHTML = causes.map(c => `
    <div class="bar-row">
      <span class="name">${escapeHtml(c.cause.replace(/_/g, " ").toLowerCase())}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(c.events / max * 100).toFixed(1)}%; background:var(--blue)"></div></div>
      <span class="val">${c.pct}%</span>
    </div>`).join("");
  document.getElementById("gridFootnote").textContent =
    `An event can have more than one cause. ${fmtPct(d.uncategorized_pct)} of narratives name no recognizable cause.`;
  document.getElementById("gridPlantsBody").innerHTML = (d.top_plants || []).map(p => `
    <tr><td>${escapeHtml(p.site)}</td><td>${escapeHtml(p.state || "")}</td>
    <td class="num">${fmtInt(p.events)}</td><td class="num">${fmtInt(p.trips)}</td></tr>`).join("");
}

function renderPlantReliability(d) {
  const rows = d.plants || [];
  document.getElementById("plantRelBody").innerHTML = rows.length ? rows.map(p => `
    <tr><td>${escapeHtml(p.site)}</td><td>${escapeHtml(p.state || "")}</td>
    <td class="num">${fmtInt(p.capacity_mw)} MW</td><td class="num">${fmtInt(p.events)}</td>
    <td class="num">${fmtInt(p.trips)}</td><td class="num">${p.trips_per_gw}</td></tr>`).join("")
    : `<tr><td colspan="6" class="text-dim">${escapeHtml(d.message || "No data.")}</td></tr>`;
  if (d.matched_plants != null) {
    document.getElementById("plantRelNote").textContent =
      `Only plants with ≥${d.min_events} reported events are ranked. Capacity was matched by plant name to the WRI database for ${d.matched_plants} plants; ${d.unmatched_plants} could not be matched and are left out (not guessed).`;
  }
}

// ----------------------------------------------------- Global plants --
async function loadPlantsTab() {
  const sel = document.getElementById("countrySelect");
  sel.addEventListener("change", () => loadPlants(sel.value));
  await loadPlants("");
}

async function loadPlants(country) {
  try {
    const d = await apiGet("/power_plants", { country, top_n: 10 });
    if (!d.available) { showPanelError("fuelBars", d.message); return; }
    document.getElementById("plantsTitle").textContent = `Installed Capacity by Fuel — ${country ? d.scope : "World"}`;
    document.getElementById("plantsKpiCount").textContent = fmtInt(d.plants);
    document.getElementById("plantsKpiGw").textContent = `${fmtInt(d.capacity_gw)} GW`;
    document.getElementById("plantsKpiFuel").textContent = d.by_fuel[0]?.primary_fuel || "—";
    document.getElementById("plantsKpiFuels").textContent = d.by_fuel.length;
    const max = d.by_fuel.length ? d.by_fuel[0].capacity_gw : 1;
    document.getElementById("fuelBars").innerHTML = d.by_fuel.map(f => `
      <div class="bar-row">
        <span class="name">${escapeHtml(f.primary_fuel)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${(f.capacity_gw / max * 100).toFixed(1)}%; background:var(--gold)"></div></div>
        <span class="val">${f.share_pct}%</span>
      </div>`).join("");
    document.getElementById("largestPlantsBody").innerHTML = d.largest_plants.map(p => `
      <tr><td>${escapeHtml(p.name)}</td><td>${escapeHtml(p.country_long)}</td>
      <td>${escapeHtml(p.primary_fuel)}</td><td class="num">${fmtInt(p.capacity_mw)}</td></tr>`).join("");
    const panel = document.getElementById("topCountriesPanel");
    panel.style.display = d.top_countries ? "" : "none";
    if (d.top_countries) {
      const cmax = d.top_countries[0].capacity_gw;
      document.getElementById("countryBars").innerHTML = d.top_countries.map(c => `
        <div class="bar-row">
          <span class="name">${escapeHtml(c.country_long)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${(c.capacity_gw / cmax * 100).toFixed(1)}%; background:var(--blue)"></div></div>
          <span class="val">${fmtInt(c.capacity_gw)}</span>
        </div>`).join("");
    }
  } catch (e) {
    showPanelError("fuelBars", `/power_plants failed: ${e.message}`);
  }
}

function showPanelError(elementId, message, colspan) {
  const el = document.getElementById(elementId);
  if (!el) return;
  if (colspan) {
    el.innerHTML = `<tr><td colspan="${colspan}" style="color:var(--accent);">${escapeHtml(message)}</td></tr>`;
  } else {
    el.innerHTML = `<div class="alert critical"><p class="alert-title">Could not load this section</p><p class="alert-body">${escapeHtml(message)}</p></div>`;
  }
}

init();