/* ===========================================================================
   app.js  -  FairLens front-end interaction layer
   ---------------------------------------------------------------------------
   Connects the static HTML page to the FastAPI engine. Four jobs:
     1. On file upload: show the filename and fill the column dropdowns.
     2. On "Run audit": POST files to /audit and show a loading state.
     3. Replace the sample numbers / bars / tables / badges with real results.
     4. Wire the metric-advisor radios and the report download button.

   Everything is plain JavaScript - no framework, no build step.
   =========================================================================== */

"use strict";

const API = "";                 // same origin as the server
let datasetFile = null;         // remembers the uploaded dataset
let referenceFile = null;       // optional drift reference

/* -- tiny helpers ---------------------------------------------------------- */
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};
function badge(status) {
  const map = { pass: ["badge-ok", "Pass"], fail: ["badge-bad", "Fail"],
                small_sample: ["badge-warn", "Small sample"],
                stable: ["badge-ok", "Stable"], moderate: ["badge-warn", "Moderate"],
                significant: ["badge-bad", "High"] };
  const [cls, text] = map[status] || ["badge-warn", status];
  return `<span class="badge ${cls}">${text}</span>`;
}

/* ===========================================================================
   JOB 1: file upload -> show name, fetch columns, fill dropdowns
   =========================================================================== */
function setDropzoneLabel(inputId, filename) {
  const label = document.querySelector(`label.dropzone[for="${inputId}"]`);
  if (label) label.innerHTML =
    `<span class="dropzone-icon">&#10003;</span><strong>${filename}</strong>
     <small>Click to choose a different file</small>`;
}

async function onDatasetChosen(e) {
  const file = e.target.files[0];
  if (!file) return;
  datasetFile = file;
  setDropzoneLabel("dataset", file.name);

  const fd = new FormData();
  fd.append("dataset", file);
  try {
    const res = await fetch(`${API}/upload`, { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const info = await res.json();
    fillColumnControls(info);
  } catch (err) {
    alert("Could not read that CSV: " + err.message);
  }
}

function fillColumnControls(info) {
  const cols = info.columns.map((c) => c.name);
  const s = info.suggestions;

  // Outcome + prediction dropdowns
  fillSelect("#target", cols, s.suggested_outcome);
  fillSelect("#prediction-col", cols, s.suggested_prediction);

  // Sensitive-attribute chips: rebuild from the real columns
  const chipBox = $("#sensitive");
  if (chipBox) {
    chipBox.innerHTML = "";
    cols.forEach((c) => {
      const checked = s.suggested_sensitive.includes(c) ? "checked" : "";
      chipBox.insertAdjacentHTML("beforeend",
        `<label class="chip"><input type="checkbox" name="sensitive" value="${c}" ${checked}><span>${c}</span></label>`);
    });
  }
}

function fillSelect(sel, cols, selected) {
  const node = $(sel);
  if (!node) return;
  node.innerHTML = `<option value="">Select a column</option>` +
    cols.map((c) => `<option value="${c}" ${c === selected ? "selected" : ""}>${c}</option>`).join("");
}

/* ===========================================================================
   JOB 2 + 3: run audit, show loading, render real results
   =========================================================================== */
async function onRunAudit(e) {
  e.preventDefault();
  if (!datasetFile) { alert("Please upload a dataset CSV first."); return; }

  const sensitive = [...document.querySelectorAll('input[name="sensitive"]:checked')]
    .map((c) => c.value);
  const predictionCol = $("#prediction-col")?.value;
  const outcomeCol = $("#target")?.value || "";
  const intersectional = $('input[name="intersectional"]')?.checked ?? true;

  if (!predictionCol) { alert("Please choose the model's decision column."); return; }
  if (sensitive.length === 0) { alert("Please pick at least one sensitive attribute."); return; }

  const btn = e.submitter || $(".form .btn-primary");
  const original = btn.textContent;
  btn.textContent = "Running audit...";
  btn.disabled = true;

  const fd = new FormData();
  fd.append("dataset", datasetFile);
  fd.append("sensitive", JSON.stringify(sensitive));
  fd.append("prediction_col", predictionCol);
  fd.append("outcome_col", outcomeCol);
  fd.append("intersectional", intersectional);
  fd.append("model_name", datasetFile.name.replace(/\.csv$/i, ""));
  if (referenceFile) fd.append("reference", referenceFile);

  try {
    const res = await fetch(`${API}/audit`, { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const result = await res.json();
    renderResults(result);
    $("#results").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    alert("Audit failed: " + err.message);
  } finally {
    btn.textContent = original;
    btn.disabled = false;
  }
}

function renderResults(result) {
  const f = result.fairness, s = f.summary, d = result.drift;

  // -- top stat cards --
  const stats = document.querySelectorAll("#results .stats .stat-value");
  if (stats.length >= 4) {
    stats[0].innerHTML = `${s.fairness_score}<small>/100</small>`;
    stats[1].textContent = s.groups_tested;
    stats[2].textContent = s.groups_flagged;
    stats[3].textContent = d.overall_band === "not_tested"
      ? "N/A" : d.overall_band;
  }

  // -- bias bars (use intersectional if present, else first attribute) --
  const groups = f.intersectional.length ? f.intersectional
                 : Object.values(f.single)[0] || [];
  const bars = $("#results .bars");
  if (bars) {
    bars.innerHTML = "";
    groups.slice(0, 8).forEach((g) => {
      const pct = Math.round(g.selection_rate * 100);
      const cls = g.status === "fail" ? "bar-bad"
                : g.status === "small_sample" ? "bar-warn" : "bar-ok";
      bars.insertAdjacentHTML("beforeend",
        `<div class="bar-row"><span class="bar-label">${g.group}</span>
         <div class="bar-track"><div class="bar ${cls}" style="width:${pct}%"></div>
         <span class="threshold"></span></div><span class="bar-value">${pct}%</span></div>`);
    });
  }

  // -- drift list --
  const driftList = $("#results .drift-list");
  if (driftList) {
    driftList.innerHTML = "";
    if (!d.features.length) {
      driftList.innerHTML = `<li class="muted">No reference dataset uploaded - drift not tested.</li>`;
    }
    d.features.slice(0, 6).forEach((x) => {
      const w = Math.min(100, Math.round((x.psi / 0.4) * 100));
      const cls = x.band === "significant" ? "bar-bad"
                : x.band === "moderate" ? "bar-warn" : "bar-ok";
      driftList.insertAdjacentHTML("beforeend",
        `<li><span>${x.feature}</span><div class="meter">
         <div class="meter-fill ${cls}" style="width:${w}%"></div></div>
         ${badge(x.band).replace(/>[^<]+</, ">" + x.psi + "<")}</li>`);
    });
  }

  // -- intersectional table --
  const tbody = document.querySelector("#results .table tbody");
  if (tbody) {
    tbody.innerHTML = "";
    groups.forEach((g) => {
      tbody.insertAdjacentHTML("beforeend",
        `<tr><td>${g.group}</td><td>${g.size.toLocaleString()}</td>
         <td>${Math.round(g.selection_rate * 100)}%</td>
         <td>${g.dp_ratio}</td>
         <td>${g.size < 100 ? "Low" : "High"}</td>
         <td>${badge(g.status)}</td></tr>`);
    });
  }

  renderExplanations(result.explanations);
  renderReport(result);
}

/* -- explanations ---------------------------------------------------------- */
function renderExplanations(explanations) {
  const card = $("#explain .decision");
  if (!card || !explanations.length) return;
  const ex = explanations.find((e) => !e.error) || explanations[0];
  if (ex.error) return;

  const head = card.querySelector(".decision-head");
  if (head) head.innerHTML =
    `<div><p class="muted">Sample applicant #${ex.index}</p>
     <h3>Decision: ${ex.decision}</h3></div>${badge(ex.decision === "approved" ? "pass" : "fail")}`;

  const plain = card.querySelector(".plain");
  if (plain) plain.innerHTML = ex.plain_language;

  const factors = card.querySelector(".factors");
  if (factors) {
    factors.innerHTML = "";
    const max = Math.max(...ex.factors.map((f) => f.weight), 0.001);
    ex.factors.forEach((f) => {
      const w = Math.round((f.weight / max) * 100);
      const cls = f.effect === "helped" ? "factor-pos" : "factor-neg";
      factors.insertAdjacentHTML("beforeend",
        `<li><span>${f.feature}</span><div class="factor-track">
         <div class="factor ${cls}" style="width:${w}%"></div></div>
         <span class="factor-note">${f.effect === "helped" ? "Helped" : "Hurt"}</span></li>`);
    });
  }
  const cf = $("#explain .counterfactual .check-list");
  if (cf) cf.innerHTML = `<li>${ex.counterfactual}</li>`;
}

/* -- report preview + enable download ------------------------------------- */
function renderReport(result) {
  const toc = $("#report .reco");
  if (toc) {
    toc.innerHTML = "";
    result.recommendations.forEach((r) => {
      toc.insertAdjacentHTML("beforeend",
        `<li>${badge(r.priority === "high" ? "fail" : r.priority === "medium" ? "moderate" : "pass")
          .replace(/>[^<]+</, ">" + r.priority + "<")} ${r.text}</li>`);
    });
  }
  const title = $("#report .paper h3");
  if (title) title.textContent = `Model Fairness Audit: ${result.model_name}`;
}

async function onDownloadReport(e) {
  e.preventDefault();
  const res = await fetch(`${API}/report`, { method: "POST" });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = el("a");
  a.href = url; a.download = "fairlens_report.html";
  a.click();
  URL.revokeObjectURL(url);
}

/* ===========================================================================
   JOB 4: metric advisor
   =========================================================================== */
const USECASE_MAP = ["hiring", "lending", "triage", "scoring"];

async function onAdvisorSubmit(e) {
  e.preventDefault();
  const radios = document.querySelectorAll('input[name="usecase"]');
  let idx = 0;
  radios.forEach((r, i) => { if (r.checked) idx = i; });
  const res = await fetch(`${API}/advisor/${USECASE_MAP[idx] || "hiring"}`);
  const rec = await res.json();

  const card = $("#advisor .recommend");
  if (card) {
    card.querySelector("h3").textContent = rec.metric;
    card.querySelector("p:not(.eyebrow)").textContent = rec.why;
    const facts = card.querySelectorAll(".facts dd");
    if (facts.length >= 3) {
      facts[0].textContent = USECASE_MAP[idx];
      facts[1].textContent = rec.watch;
      facts[2].textContent = rec.backup;
    }
  }
}

/* ===========================================================================
   wire everything up on page load
   =========================================================================== */
document.addEventListener("DOMContentLoaded", () => {
  $("#dataset")?.addEventListener("change", onDatasetChosen);
  $("#predictions")?.addEventListener("change", (e) => {
    if (e.target.files[0]) setDropzoneLabel("predictions", e.target.files[0].name);
  });
  $("#reference")?.addEventListener("change", (e) => {
    referenceFile = e.target.files[0] || null;
  });
  $(".form")?.addEventListener("submit", onRunAudit);
  $(".advisor-form")?.addEventListener("submit", onAdvisorSubmit);

  // Report download buttons (the two links in the report section)
  document.querySelectorAll("#report .btn").forEach((b) =>
    b.addEventListener("click", onDownloadReport));
});
