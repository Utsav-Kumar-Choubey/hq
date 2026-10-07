/* ===========================================================================
   app.js - FairLens front-end interaction layer
   ---------------------------------------------------------------------------
   Connects the static page to the FastAPI engine:
     1. Upload: show the file name and fill the column controls.
     2. Run audit: send files and selections to /audit with a loading state.
     3. Render: replace the sample numbers, bars, tables and badges.
     4. Advisor and report: metric recommendation and report download.
   Plain JavaScript, no framework or build step.
   =========================================================================== */

"use strict";

const API = "";
const state = { dataset: null, reference: null, result: null, grouping: null };

/* -- helpers --------------------------------------------------------------- */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
const pct = (v) => (v == null ? "n/a" : `${Math.round(v * 100)}%`);
const num = (v, d = 2) => (v == null ? "n/a" : Number(v).toFixed(d));

const BADGES = {
  pass: ["badge-ok", "Pass"], fail: ["badge-bad", "Fail"],
  review: ["badge-warn", "Review (small sample)"],
  not_applicable: ["badge-warn", "n/a"],
  stable: ["badge-ok", "Stable"], moderate: ["badge-warn", "Moderate"],
  significant: ["badge-bad", "Significant"],
  high: ["badge-bad", "High"], medium: ["badge-warn", "Medium"], low: ["badge-ok", "Low"],
  approved: ["badge-ok", "Approved"], rejected: ["badge-bad", "Rejected"],
};
function badge(key, text) {
  const [cls, label] = BADGES[key] || ["badge-warn", key];
  return `<span class="badge ${cls}">${esc(text ?? label)}</span>`;
}

function setStatus(message, kind = "") {
  const node = $("#audit-status");
  if (!node) return;
  node.textContent = message;
  node.className = `hint ${kind ? `is-${kind}` : ""}`;
}

async function apiError(res) {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
  } catch (_) { /* fall through */ }
  return `${res.status} ${res.statusText}`;
}

/* ===========================================================================
   1. Upload
   =========================================================================== */
function setDropzoneLabel(inputId, filename) {
  const label = document.querySelector(`label.dropzone[for="${inputId}"]`);
  if (label) {
    label.innerHTML = `<span class="dropzone-icon">&#10003;</span><strong>${esc(filename)}</strong>
      <small>Click to choose a different file</small>`;
  }
}

async function onDatasetChosen(event) {
  const file = event.target.files[0];
  if (!file) return;
  if (!/\.csv$/i.test(file.name)) {
    setStatus("The dataset must be a .csv file.", "error");
    event.target.value = "";
    return;
  }
  state.dataset = file;
  setDropzoneLabel("dataset", file.name);
  setStatus("Reading columns...");

  const fd = new FormData();
  fd.append("dataset", file);
  try {
    const res = await fetch(`${API}/upload`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(await apiError(res));
    const info = await res.json();
    fillColumnControls(info);
    setStatus(`Loaded ${info.rows.toLocaleString()} rows and ${info.columns.length} columns. `
      + "Check the selections below, then run the audit.", "ok");
  } catch (err) {
    state.dataset = null;
    setStatus(`Could not read that CSV: ${err.message}`, "error");
  }
}

function fillSelect(sel, columns, selected) {
  const node = $(sel);
  if (!node) return;
  node.innerHTML = `<option value="">Select a column</option>`
    + columns.map((c) => `<option value="${esc(c.name)}" ${c.name === selected ? "selected" : ""}>`
      + `${esc(c.name)}${c.is_binary ? "" : " (not yes/no)"}</option>`).join("");
}

function fillColumnControls(info) {
  const s = info.suggestions;
  fillSelect("#target", info.columns, s.suggested_outcome);
  fillSelect("#prediction-col", info.columns, s.suggested_prediction);

  const box = $("#sensitive");
  if (!box) return;
  box.innerHTML = info.columns.map((c) => {
    const checked = s.suggested_sensitive.includes(c.name) ? "checked" : "";
    return `<label class="chip"><input type="checkbox" name="sensitive" value="${esc(c.name)}" ${checked}>`
      + `<span>${esc(c.name)}</span></label>`;
  }).join("");
}

/* ===========================================================================
   2. Run audit
   =========================================================================== */
async function onRunAudit(event) {
  event.preventDefault();
  if (!state.dataset) return setStatus("Upload a dataset CSV first.", "error");

  const sensitive = $$('input[name="sensitive"]:checked').map((c) => c.value);
  const predictionCol = $("#prediction-col")?.value || "";
  const outcomeCol = $("#target")?.value || "";
  if (!predictionCol) return setStatus("Choose the column that holds the model decision.", "error");
  if (!sensitive.length) return setStatus("Choose at least one sensitive attribute.", "error");
  if (sensitive.includes(predictionCol) || sensitive.includes(outcomeCol)) {
    return setStatus("A sensitive attribute cannot also be the decision or outcome column.", "error");
  }

  const fd = new FormData();
  fd.append("dataset", state.dataset);
  fd.append("sensitive", JSON.stringify(sensitive));
  fd.append("prediction_col", predictionCol);
  fd.append("outcome_col", outcomeCol);
  fd.append("intersectional", $('input[name="intersectional"]')?.checked ? "true" : "false");
  fd.append("model_name", state.dataset.name.replace(/\.csv$/i, ""));
  fd.append("use_case", $("#use-case")?.value || "");
  fd.append("metric", $("#metric")?.value || "");
  if (state.reference) fd.append("reference", state.reference);

  const btn = $("#run-audit");
  const label = btn.textContent;
  btn.textContent = "Running audit...";
  btn.disabled = true;
  setStatus("Running the bias, drift and explainability checks...");

  try {
    const res = await fetch(`${API}/audit`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(await apiError(res));
    state.result = await res.json();
    renderResults(state.result);
    setStatus("Audit complete. Results, explanations and the report are updated below.", "ok");
    $("#results").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    setStatus(`Audit failed: ${err.message}`, "error");
  } finally {
    btn.textContent = label;
    btn.disabled = false;
  }
}

/* ===========================================================================
   3. Render results
   =========================================================================== */
function groupings(fairness) {
  const out = {};
  Object.entries(fairness.single).forEach(([attr, rows]) => { out[attr] = rows; });
  fairness.intersectional.forEach((row) => {
    (out[row.grouping] = out[row.grouping] || []).push(row);
  });
  return out;
}

function renderResults(result) {
  const f = result.fairness;
  const s = f.summary;
  const d = result.drift;

  $("#results-sub").textContent = `${result.model_name}: ${result.dataset.rows.toLocaleString()} records, `
    + `audited with ${f.metric.name.toLowerCase()} across ${result.dataset.sensitive_attributes.join(", ")}.`;

  const stats = $$("#results .stats .stat-value");
  if (stats.length >= 4) {
    stats[0].innerHTML = `${s.fairness_score}<small>/100</small>`;
    stats[1].textContent = s.groups_tested;
    stats[2].textContent = s.groups_flagged + (s.groups_review ? ` (+${s.groups_review})` : "");
    stats[3].textContent = d.overall_band === "not_tested" ? "Not tested"
      : d.overall_band.charAt(0).toUpperCase() + d.overall_band.slice(1);
  }

  const note = $("#metric-note");
  if (note) {
    note.hidden = !f.metric.note;
    note.textContent = f.metric.note || "";
  }

  const all = groupings(f);
  const keys = Object.keys(all);
  // Default view: the widest intersectional grouping, otherwise the first attribute.
  const inter = keys.filter((k) => k.includes(" x "));
  state.grouping = inter.length ? inter[inter.length - 1] : keys[0];
  renderTabs(keys);
  renderGrouping();
  renderDrift(d);
  renderExplainability(result);
  renderReport(result);
}

function renderTabs(keys) {
  const box = $("#group-tabs");
  if (!box) return;
  box.innerHTML = keys.map((k) => `<button type="button" data-key="${esc(k)}" `
    + `aria-pressed="${k === state.grouping}">${esc(k)}</button>`).join("");
  box.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    state.grouping = b.dataset.key;
    box.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    renderGrouping();
  }));
}

function renderGrouping() {
  const f = state.result.fairness;
  const rows = groupings(f)[state.grouping] || [];
  const metric = f.metric;

  // Bars: selection rate with the four-fifths threshold of the reference group.
  const sized = rows.filter((r) => r.size >= 30);
  const top = Math.max(...(sized.length ? sized : rows).map((r) => r.selection_rate || 0), 0.0001);
  const threshold = Math.min(100, top * 0.8 * 100);
  $("#bars-note").textContent = `Grouping: ${state.grouping}. Bars show the approval rate; `
    + `the dashed line marks 80% of the best-off group (${pct(top)}).`;
  $("#results .bars").innerHTML = rows.slice(0, 12).map((g) => {
    const cls = g.status === "fail" ? "bar-bad" : g.status === "pass" ? "bar-ok" : "bar-warn";
    return `<div class="bar-row"><span class="bar-label" title="${esc(g.group)}">${esc(g.group)}</span>`
      + `<div class="bar-track"><div class="bar ${cls}" style="width:${Math.round((g.selection_rate || 0) * 100)}%"></div>`
      + `<span class="threshold" style="left:${threshold.toFixed(1)}%"></span></div>`
      + `<span class="bar-value">${pct(g.selection_rate)}</span></div>`;
  }).join("") || `<p class="muted">No groups to display.</p>`;

  // Table: primary metric plus the others for context.
  const head = $("#results .table thead tr");
  const hasOutcome = rows.some((r) => r.true_positive_rate != null);
  head.innerHTML = "<th>Group</th><th>Size</th><th>Approval rate (95% CI)</th>"
    + "<th>Selection ratio</th>"
    + (hasOutcome ? "<th>TPR ratio</th><th>Odds gap</th><th>Precision ratio</th>" : "")
    + `<th>${esc(metric.name)}</th><th>Confidence</th><th>Status</th>`;
  $("#groups-caption").textContent = `Fairness detail by ${state.grouping}`;
  $("#groups-body").innerHTML = rows.map((g) => `<tr>
      <td>${esc(g.group)}${g.is_reference ? ' <span class="ci">reference group</span>' : ""}</td>
      <td>${g.size.toLocaleString()}</td>
      <td>${pct(g.selection_rate)}<span class="ci">${pct(g.selection_ci[0])} to ${pct(g.selection_ci[1])}</span></td>
      <td>${num(g.dp_ratio)}</td>
      ${hasOutcome ? `<td>${num(g.eo_ratio)}</td><td>${num(g.odds_gap)}</td><td>${num(g.pp_ratio)}</td>` : ""}
      <td><strong>${num(g.primary_value)}</strong></td>
      <td>${g.confidence === "high" ? "High" : "Low"}</td>
      <td>${badge(g.status)}</td></tr>`).join("");
}

function renderDrift(d) {
  const list = $("#results .drift-list");
  if (!list) return;
  if (!d.features.length) {
    list.innerHTML = `<li class="muted">No reference dataset was uploaded, so drift was not tested. `
      + `Add the training data in field 8 to compare distributions.</li>`;
    return;
  }
  list.innerHTML = d.features.slice(0, 8).map((x) => {
    const width = Math.min(100, Math.round((x.psi / 0.4) * 100));
    const cls = x.band === "significant" ? "bar-bad" : x.band === "moderate" ? "bar-warn" : "bar-ok";
    const ks = x.ks_pvalue != null ? ` title="KS test p-value ${x.ks_pvalue}"` : "";
    return `<li${ks}><span>${esc(x.feature)}</span><div class="meter">`
      + `<div class="meter-fill ${cls}" style="width:${width}%"></div></div>${badge(x.band, num(x.psi))}</li>`;
  }).join("");
}

/* -- explanations ---------------------------------------------------------- */
function renderExplanation(ex) {
  const card = $("#explain .decision");
  if (!card || !ex) return;
  card.querySelector(".decision-head").innerHTML = `<div><p class="muted">Record #${esc(ex.index)}</p>`
    + `<h3>Decision: ${esc(ex.decision)}</h3></div>${badge(ex.decision)}`;
  card.querySelector(".plain").textContent = ex.plain_language;

  const max = Math.max(...ex.factors.map((f) => f.weight), 0.001);
  card.querySelector(".factors").innerHTML = ex.factors.map((f) => `<li><span>${esc(f.label)}`
    + `<span class="factor-value">${esc(f.value)}</span></span>`
    + `<div class="factor-track"><div class="factor ${f.effect === "helped" ? "factor-pos" : "factor-neg"}" `
    + `style="width:${Math.round((f.weight / max) * 100)}%"></div></div>`
    + `<span class="factor-note">${f.effect === "helped" ? "Helped" : "Hurt"}</span></li>`).join("");
  $("#explain-caveat").textContent = ex.caveat
    || `The explanation model agrees with the audited model on ${pct(ex.fidelity)} of records.`;

  const target = ex.decision === "approved" ? "rejected" : "approved";
  $("#cf-title").textContent = ex.counterfactuals.length
    ? `Changes that would likely make this ${target}`
    : "No single small change flips this decision";
  $("#cf-list").innerHTML = ex.counterfactuals.length
    ? ex.counterfactuals.map((c) => `<li>${esc(c.text)}</li>`).join("")
    : `<li class="muted">The outcome is driven by several factors together, so no single realistic change reverses it.</li>`;
  const input = $("#record-index");
  if (input) input.value = ex.index;
}

function renderExplainability(result) {
  const info = result.explainability || {};
  const status = $("#explain-status");
  const input = $("#record-index");
  const btn = $("#explain-btn");
  if (!info.available) {
    if (status) { status.textContent = info.reason || "Explanations are unavailable."; status.className = "hint is-error"; }
    return;
  }
  if (input) { input.disabled = false; input.max = info.records - 1; }
  if (btn) btn.disabled = false;
  if (status) { status.textContent = `Enter a record number from 0 to ${(info.records - 1).toLocaleString()}.`; status.className = "hint"; }

  const samples = $("#sample-records");
  if (samples) {
    const items = [...info.samples.rejected.slice(0, 2).map((i) => [i, "rejected"]),
      ...info.samples.approved.slice(0, 2).map((i) => [i, "approved"])];
    samples.innerHTML = items.map(([i, d]) => `<button type="button" data-index="${i}">#${i} (${d})</button>`).join("");
    samples.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => explainRecord(b.dataset.index)));
  }

  $("#fidelity-note").textContent = `Share of influence across all ${info.records.toLocaleString()} records. `
    + `The explanation model reproduces the audited model's decisions ${pct(info.fidelity)} of the time.`;
  const top = Math.max(...info.global_importance.map((g) => g.share), 0.001);
  $("#global-importance").innerHTML = info.global_importance.map((g) => `<li><span>${esc(g.label)}</span>`
    + `<div class="factor-track"><div class="factor factor-pos" style="width:${Math.round((g.share / top) * 100)}%"></div></div>`
    + `<span class="factor-note">${pct(g.share)}</span></li>`).join("");

  const proxies = info.proxies || [];
  $("#proxy-list").innerHTML = proxies.length
    ? proxies.map((p) => `<li><strong>${esc(p.feature)}</strong> is strongly associated with `
      + `<strong>${esc(p.sensitive)}</strong> (Cramer's V ${num(p.strength)}).</li>`).join("")
    : `<li>No feature is strongly associated with the sensitive attributes.</li>`;

  const first = (result.explanations || []).find((e) => !e.error);
  if (first) renderExplanation(first);
}

async function explainRecord(index) {
  const status = $("#explain-status");
  const value = Number(index);
  if (!Number.isInteger(value) || value < 0) {
    status.textContent = "Enter a whole record number of 0 or more.";
    status.className = "hint is-error";
    return;
  }
  status.textContent = `Explaining record #${value}...`;
  status.className = "hint";
  try {
    const res = await fetch(`${API}/explain/${value}`);
    if (!res.ok) throw new Error(await apiError(res));
    renderExplanation(await res.json());
    status.textContent = `Showing record #${value}.`;
  } catch (err) {
    status.textContent = err.message;
    status.className = "hint is-error";
  }
}

/* -- report ---------------------------------------------------------------- */
function renderReport(result) {
  const list = $("#report .reco");
  if (list) {
    list.innerHTML = result.recommendations
      .map((r) => `<li>${badge(r.priority)} <span>${esc(r.text)}</span></li>`).join("");
  }
  const title = $("#report .paper h3");
  if (title) title.textContent = `Model Fairness Audit: ${result.model_name}`;
}

async function onDownloadReport(event) {
  event.preventDefault();
  const res = await fetch(`${API}/report`, { method: "POST" });
  if (!res.ok) {
    alert("No report is available yet. Run an audit first.");
    return;
  }
  const url = URL.createObjectURL(await res.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = "fairlens_report.html";
  a.click();
  URL.revokeObjectURL(url);
}

/* ===========================================================================
   4. Metric advisor
   =========================================================================== */
let lastAdvice = null;

async function showAdvice(useCase) {
  const res = await fetch(`${API}/advisor/${encodeURIComponent(useCase)}`);
  if (!res.ok) return;
  const rec = await res.json();
  lastAdvice = rec;
  $("#rec-metric").textContent = rec.metric;
  $("#rec-why").textContent = rec.why;
  $("#rec-best").textContent = rec.best_for;
  $("#rec-threshold").textContent = rec.threshold;
  $("#rec-watch").textContent = rec.watch;
  $("#rec-backup").textContent = rec.backup;
}

function selectedAdvisorCase() {
  return $('input[name="usecase"]:checked')?.value || "hiring";
}

function applyAdviceToAudit() {
  const useCase = selectedAdvisorCase();
  const useCaseSelect = $("#use-case");
  if (useCaseSelect) useCaseSelect.value = useCase;
  const metricSelect = $("#metric");
  if (metricSelect && lastAdvice) metricSelect.value = lastAdvice.metric_id;
  updateMetricHint();
  $("#audit").scrollIntoView({ behavior: "smooth" });
}

function updateMetricHint() {
  const hint = $("#metric-hint");
  const metric = $("#metric");
  if (!hint || !metric) return;
  hint.textContent = metric.value
    ? "This metric decides pass or fail; the others are shown for context."
    : "The metric recommended for the selected use case will be applied.";
}

/* ===========================================================================
   wiring
   =========================================================================== */
document.addEventListener("DOMContentLoaded", () => {
  $("#dataset")?.addEventListener("change", onDatasetChosen);
  $("#predictions")?.addEventListener("change", (e) => {
    if (e.target.files[0]) setDropzoneLabel("predictions", e.target.files[0].name);
  });
  $("#reference")?.addEventListener("change", (e) => { state.reference = e.target.files[0] || null; });
  $("#audit-form")?.addEventListener("submit", onRunAudit);
  $("#audit-form")?.addEventListener("reset", () => {
    state.dataset = null;
    state.reference = null;
    setStatus("");
  });
  $("#metric")?.addEventListener("change", updateMetricHint);

  $(".advisor-form")?.addEventListener("submit", (e) => { e.preventDefault(); showAdvice(selectedAdvisorCase()); });
  $$('input[name="usecase"]').forEach((r) => r.addEventListener("change", () => showAdvice(r.value)));
  $("#use-recommendation")?.addEventListener("click", applyAdviceToAudit);
  showAdvice("hiring");

  $("#explain-btn")?.addEventListener("click", () => explainRecord($("#record-index").value));
  $("#record-index")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); explainRecord(e.target.value); }
  });

  $$("#report .btn").forEach((b) => b.addEventListener("click", onDownloadReport));
});
