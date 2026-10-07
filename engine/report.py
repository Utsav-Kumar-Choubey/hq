"""
report.py
---------
Turns the audit results into (a) a plain-language recommendations list and
(b) a self-contained HTML compliance report that can be downloaded or printed
to PDF from the browser.
"""

from __future__ import annotations

import html
import platform
from datetime import datetime, timezone


def build_recommendations(fairness: dict, drift: dict,
                          explainability: dict = None) -> list[dict]:
    """Produce prioritised, human recommendations from the numbers."""
    recs = []
    s = fairness["summary"]

    metric = fairness["metric"]
    rows = [r for g in fairness["single"].values() for r in g] + fairness["intersectional"]
    failing = sorted((r for r in rows if r["status"] == "fail"),
                     key=lambda r: r["compliance"])
    for r in failing[:3]:
        recs.append({
            "priority": "high",
            "text": (f"Investigate the {metric['name'].lower()} gap for "
                     f"'{r['group']}' ({r['grouping']}): measured {r['primary_value']}; "
                     f"required: {metric['threshold'][0].lower() + metric['threshold'][1:]}."),
        })
    review = [r for r in rows if r["status"] == "review"]
    if review:
        recs.append({
            "priority": "medium",
            "text": (f"{len(review)} small group(s) fall below the threshold but have "
                     f"fewer than 100 people, for example '{review[0]['group']}'. "
                     "Validate with more data before drawing conclusions."),
        })
    if metric.get("note"):
        recs.append({"priority": "medium",
                     "text": metric["note"] + " Supply the actual outcome to test it."})

    if drift["n_drifted"] > 0:
        top = drift["features"][0]
        recs.append({
            "priority": "medium",
            "text": (f"Retrain with recent data: '{top['feature']}' has drifted "
                     f"(PSI {top['psi']}, {top['band']})."),
        })

    ex = explainability or {}
    for proxy in ex.get("proxies", [])[:2]:
        if proxy["strength"] >= 0.9:
            text = (f"'{proxy['feature']}' carries essentially the same information as "
                    f"'{proxy['sensitive']}' (Cramer's V {proxy['strength']}). If the model "
                    "uses it, the sensitive attribute is effectively a direct input; "
                    "confirm this is lawful and justified for the use case.")
        else:
            text = (f"'{proxy['feature']}' is strongly associated with "
                    f"'{proxy['sensitive']}' (Cramer's V {proxy['strength']}). It may act "
                    "as a proxy; review whether the model should use it.")
        recs.append({"priority": "medium", "text": text})
    if ex.get("available") and ex.get("fidelity", 1) < 0.85:
        recs.append({
            "priority": "medium",
            "text": (f"The explanation surrogate matches the model on only "
                     f"{ex['fidelity']:.0%} of records; treat per-decision explanations "
                     "as indicative and consider a model-specific explainer."),
        })

    if s["small_sample_groups"] > 0 and not review:
        recs.append({
            "priority": "low",
            "text": (f"{s['small_sample_groups']} group(s) have fewer than 100 people; "
                     "their results are low-confidence."),
        })

    recs.append({"priority": "low",
                 "text": "Schedule a fairness and drift re-check every quarter."})
    return recs


TOOL_VERSION = "FairLens 1.2"

LIMITATIONS = [
    "Fairness metrics can conflict mathematically when groups have different base "
    "rates; passing the primary metric does not imply passing every metric.",
    "Results describe the supplied data only. If outcomes were recorded under a "
    "biased historical process, outcome-based metrics inherit that bias.",
    "Groups with fewer than 100 records are reported with low confidence and are "
    "not counted as firm failures.",
    "Per-decision explanations come from a transparent surrogate model that "
    "approximates the audited model; its agreement rate is stated in section 6.",
    "Counterfactuals change one feature at a time within the observed range and "
    "indicate sensitivity; they are not guaranteed routes to a different outcome.",
    "Proxy detection measures statistical association only and does not show that "
    "the model relies on the feature.",
    "This report supports, but does not replace, legal and domain review.",
]


# --------------------------------------------------------------------------
# structured report
# --------------------------------------------------------------------------
def _key_findings(fairness: dict, drift: dict, explainability: dict) -> list:
    s, m = fairness["summary"], fairness["metric"]
    out = []
    if s["groups_flagged"]:
        out.append(f"{s['groups_flagged']} of {s['groups_tested']} groups fail {m['name'].lower()}; "
                   f"the most affected is '{s['worst_group']}' (value {s['worst_value']}).")
    else:
        out.append(f"No group with adequate sample size fails {m['name'].lower()}.")
    if s["groups_review"]:
        out.append(f"{s['groups_review']} small group(s) fall below the threshold and need "
                   "more data before a conclusion can be drawn.")
    if s.get("worst_ratio") is not None and m["id"] != "demographic_parity":
        out.append(f"For context, the lowest selection-rate ratio is {s['worst_ratio']} "
                   "(the four-fifths rule requires 0.80).")
    if drift["overall_band"] == "not_tested":
        out.append("Drift was not tested because no reference dataset was supplied.")
    else:
        out.append(f"Overall drift is {drift['overall_band']} (highest PSI {drift['worst_psi']}); "
                   f"{drift['n_drifted']} feature(s) shifted.")
    if explainability.get("available"):
        out.append(f"Decision explanations reproduce the model's decisions on "
                   f"{explainability['fidelity']:.0%} of records.")
    proxies = explainability.get("proxies") or []
    if proxies:
        p = proxies[0]
        out.append(f"'{p['feature']}' is strongly associated with '{p['sensitive']}' "
                   f"(Cramer's V {p['strength']}) and may act as a proxy.")
    return out


def build_report(model_name: str, dataset_info: dict, fairness: dict,
                 drift: dict, recommendations: list,
                 explainability: dict = None, explanations: list = None) -> dict:
    """Return a structured report (JSON-serialisable) plus a rendered HTML string."""
    explainability = explainability or {}
    explanations = [e for e in (explanations or []) if "error" not in e]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = {
        "title": f"Model Fairness Audit: {model_name}",
        "model_name": model_name,
        "generated": generated,
        "tool_version": TOOL_VERSION,
        "dataset": dataset_info,
        "metric": fairness["metric"],
        "overall": fairness.get("overall", {}),
        "fairness_summary": fairness["summary"],
        "key_findings": _key_findings(fairness, drift, explainability),
        "drift_summary": {"overall": drift["overall_band"], "worst_psi": drift["worst_psi"],
                          "features_drifted": drift["n_drifted"]},
        "explainability_summary": {k: explainability.get(k) for k in
                                   ("available", "fidelity", "global_importance", "proxies")},
        "limitations": LIMITATIONS,
        "recommendations": recommendations,
        "sections": ["Executive summary", "Scope: data and model", "Method and metric selection",
                     "Fairness results: single attributes", "Fairness results: intersectional groups",
                     "Drift analysis", "Explainability", "Limitations", "Recommendations",
                     "Audit trail and sign-off"],
    }
    report["html"] = render_html(report, fairness, drift, explainability, explanations)
    return report


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------
def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def _num(v, digits: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


_STATUS = {"pass": ("ok", "Pass"), "fail": ("bad", "Fail"),
           "review": ("warn", "Review"), "not_applicable": ("warn", "n/a"),
           "stable": ("ok", "Stable"), "moderate": ("warn", "Moderate"),
           "significant": ("bad", "Significant"),
           "high": ("bad", "High"), "medium": ("warn", "Medium"), "low": ("ok", "Low"),
           "approved": ("ok", "Approved"), "rejected": ("bad", "Rejected")}


def _pill(key: str) -> str:
    cls, label = _STATUS.get(key, ("warn", key))
    return f'<span class="pill {cls}">{_e(label)}</span>'


def _group_table(rows: list, has_outcome: bool, metric_name: str) -> str:
    head = ("<tr><th>Group</th><th>Size</th><th>Selection rate (95% CI)</th>"
            "<th>Selection ratio</th>")
    if has_outcome:
        head += "<th>TPR</th><th>FPR</th><th>TPR ratio</th><th>Odds gap</th><th>Precision ratio</th>"
    head += f"<th>{_e(metric_name)}</th><th>Status</th></tr>"
    body = []
    for r in rows:
        ref = ' <span class="muted">(reference)</span>' if r.get("is_reference") else ""
        cells = (f"<td>{_e(r['group'])}{ref}</td><td>{r['size']:,}</td>"
                 f"<td>{_pct(r['selection_rate'])} "
                 f"<span class='muted'>({_pct(r['selection_ci'][0])}-{_pct(r['selection_ci'][1])})</span></td>"
                 f"<td>{_num(r['dp_ratio'])}</td>")
        if has_outcome:
            cells += (f"<td>{_pct(r['true_positive_rate'])}</td><td>{_pct(r['false_positive_rate'])}</td>"
                      f"<td>{_num(r['eo_ratio'])}</td><td>{_num(r['odds_gap'])}</td>"
                      f"<td>{_num(r['pp_ratio'])}</td>")
        cells += f"<td><strong>{_num(r['primary_value'])}</strong></td><td>{_pill(r['status'])}</td>"
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


def render_html(report: dict, fairness: dict, drift: dict,
                explainability: dict, explanations: list) -> str:
    s, m, d = report["fairness_summary"], report["metric"], report["dataset"]
    overall = report["overall"]
    has_outcome = bool(d.get("outcome_column"))
    verdict = s["verdict"].replace("_", " ").capitalize()
    verdict_cls = {"pass": "ok", "needs_attention": "warn", "fail": "bad"}[s["verdict"]]

    findings = "".join(f"<li>{_e(f)}</li>" for f in report["key_findings"])
    top_recs = "".join(f"<li>{_pill(r['priority'])} {_e(r['text'])}</li>"
                       for r in report["recommendations"][:3])

    scope_rows = [
        ("Model", report["model_name"]),
        ("Intended use", (d.get("use_case") or "not specified").capitalize()),
        ("Records audited", f"{d['rows']:,}"),
        ("Columns", f"{len(d['columns'])}"),
        ("Model decision column", d.get("prediction_column")),
        ("Actual outcome column", d.get("outcome_column") or "not supplied"),
        ("Sensitive attributes", ", ".join(d["sensitive_attributes"])),
        ("Overall selection rate", _pct(overall.get("selection_rate"))),
    ]
    if has_outcome:
        scope_rows += [("Overall accuracy", _pct(overall.get("accuracy"))),
                       ("Positive outcome base rate", _pct(overall.get("base_rate")))]
    scope = "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in scope_rows)

    single = "".join(
        f"<h3>{_e(attr)}</h3>{_group_table(rows, has_outcome, m['name'])}"
        for attr, rows in fairness["single"].items())

    groupings = {}
    for r in fairness["intersectional"]:
        groupings.setdefault(r["grouping"], []).append(r)
    inter = "".join(f"<h3>{_e(g)}</h3>{_group_table(rows, has_outcome, m['name'])}"
                    for g, rows in groupings.items()) or \
        "<p>Intersectional analysis requires at least two sensitive attributes.</p>"

    if drift["features"]:
        drift_rows = "".join(
            f"<tr><td>{_e(x['feature'])}</td><td>{_num(x['psi'], 3)}</td>"
            f"<td>{_e(x.get('ks_pvalue', 'n/a'))}</td><td>{_pill(x['band'])}</td></tr>"
            for x in drift["features"])
        drift_html = ("<p>Population Stability Index (PSI): below 0.10 stable, 0.10 to 0.25 "
                      "moderate, above 0.25 significant. The Kolmogorov-Smirnov p-value is "
                      "given for numeric features (below 0.05 indicates a significant shift).</p>"
                      "<table><thead><tr><th>Feature</th><th>PSI</th><th>KS p-value</th>"
                      f"<th>Band</th></tr></thead><tbody>{drift_rows}</tbody></table>")
    else:
        drift_html = ("<p>Not tested: no reference (training-time) dataset was supplied. "
                      "Re-run the audit with the training data to assess drift.</p>")

    if explainability.get("available"):
        importance = "".join(
            f"<tr><td>{_e(g['label'])}</td><td>{_pct(g['share'])}</td></tr>"
            for g in explainability.get("global_importance", []))
        cases = []
        for ex in explanations:
            cfs = "".join(f"<li>{_e(c['text'])}</li>" for c in ex.get("counterfactuals", [])) \
                or "<li>No single realistic change reverses this decision.</li>"
            factors = ", ".join(f"{f['label']} = {f['value']} ({f['effect']})"
                                for f in ex["factors"])
            cases.append(f"<div class='case'><p><strong>Record #{ex['index']}</strong> "
                         f"{_pill(ex['decision'])}</p><p>{_e(ex['plain_language'])}</p>"
                         f"<p class='muted'>Main factors: {_e(factors)}</p>"
                         f"<p><strong>What would change the outcome</strong></p><ul>{cfs}</ul></div>")
        explain_html = (f"<p>Explanations are produced by a transparent surrogate model that "
                        f"reproduces the audited model's decisions on "
                        f"<strong>{_pct(explainability['fidelity'])}</strong> of records. "
                        "Sensitive attributes are excluded from explanations and counterfactuals.</p>"
                        "<h3>Overall drivers of the model's decisions</h3>"
                        f"<table><thead><tr><th>Feature</th><th>Share of influence</th></tr></thead>"
                        f"<tbody>{importance}</tbody></table>"
                        f"<h3>Example decisions</h3>{''.join(cases)}")
    else:
        explain_html = f"<p>{_e(explainability.get('reason', 'Explanations unavailable.'))}</p>"
    proxies = explainability.get("proxies") or []
    explain_html += "<h3>Possible proxy features</h3>" + (
        "<table><thead><tr><th>Feature</th><th>Sensitive attribute</th><th>Cramer's V</th>"
        "</tr></thead><tbody>" + "".join(
            f"<tr><td>{_e(p['feature'])}</td><td>{_e(p['sensitive'])}</td>"
            f"<td>{_num(p['strength'])}</td></tr>" for p in proxies) + "</tbody></table>"
        if proxies else "<p>No feature is strongly associated (Cramer's V of 0.5 or more) "
                        "with a sensitive attribute.</p>")

    limitations = "".join(f"<li>{_e(x)}</li>" for x in report["limitations"])
    recs = "".join(f"<li>{_pill(r['priority'])} {_e(r['text'])}</li>"
                   for r in report["recommendations"])
    note = f"<p class='note'>{_e(m['note'])}</p>" if m.get("note") else ""
    toc = "".join(f"<li>{_e(x)}</li>" for x in report["sections"])

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(report['title'])}</title>
<style>
 :root {{ --navy:#14213d; --muted:#5b6678; --line:#dfe4ec; }}
 body {{ font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; color:#1c2433;
        max-width: 960px; margin: 40px auto; padding: 0 24px; line-height: 1.55; font-size: 14px; }}
 h1 {{ font-size: 26px; color: var(--navy); border-top: 6px solid var(--navy); padding-top: 14px; margin-bottom: 4px; }}
 h2 {{ font-size: 19px; color: var(--navy); margin-top: 34px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }}
 h3 {{ font-size: 15px; color: var(--navy); margin: 18px 0 6px; }}
 table {{ border-collapse: collapse; width: 100%; margin: 8px 0 14px; font-size: 12.5px; }}
 th, td {{ border: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }}
 thead th {{ background: #f3f6fb; }}
 .scope th {{ width: 34%; background: #f8fafc; }}
 .muted {{ color: var(--muted); font-size: 12px; }}
 .pill {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
 .pill.ok {{ background: #e3f4ea; color: #1a7f4b; }} .pill.warn {{ background: #fdf0d5; color: #9a6200; }}
 .pill.bad {{ background: #fbe6e4; color: #b3261e; }}
 .summary {{ display: flex; gap: 24px; align-items: center; padding: 16px; border: 1px solid var(--line); border-radius: 10px; }}
 .score {{ font-size: 44px; font-weight: 800; color: var(--navy); line-height: 1; }}
 .note {{ background: #fdf0d5; color: #9a6200; padding: 8px 12px; border-radius: 8px; }}
 .case {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; margin: 10px 0; }}
 ul.recs li, ul.plain li {{ margin: 6px 0; }}
 .sign td {{ height: 34px; }}
 @media print {{ body {{ margin: 0; max-width: none; }} h2 {{ page-break-after: avoid; }}
   table, .case {{ page-break-inside: avoid; }} .noprint {{ display: none; }} }}
 @page {{ margin: 16mm; }}
</style></head><body>
<h1>{_e(report['title'])}</h1>
<p class="muted">Generated {_e(report['generated'])} by {_e(report['tool_version'])}. Bias auditing and explainability report.</p>
<p class="muted">Contents:</p><ol class="muted">{toc}</ol>

<h2>1. Executive summary</h2>
<div class="summary"><div><div class="score">{s['fairness_score']}<span class="muted">/100</span></div>
<div class="muted">Fairness score</div></div>
<div><p><strong>Verdict:</strong> <span class="pill {verdict_cls}">{_e(verdict)}</span></p>
<p><strong>Primary metric:</strong> {_e(m['name'])}. {_e(m['question'])}<br>
<span class="muted">Pass threshold: {_e(m['threshold'])}.</span></p></div></div>
{note}
<h3>Key findings</h3><ul class="plain">{findings}</ul>
<h3>Priority actions</h3><ul class="recs">{top_recs}</ul>

<h2>2. Scope: data and model</h2>
<table class="scope"><tbody>{scope}</tbody></table>

<h2>3. Method and metric selection</h2>
<p>Records are grouped by each sensitive attribute and by every combination of two or more
attributes (intersectional groups). Each group is compared with a reference group: the
best-off group with at least 100 records. The primary metric, <strong>{_e(m['name'])}</strong>,
decides pass or fail; the other metrics are reported for context.</p>
<table><thead><tr><th>Metric</th><th>Question answered</th><th>Pass threshold</th></tr></thead><tbody>
<tr><td>Demographic parity</td><td>Are groups selected at similar rates?</td><td>Selection-rate ratio of at least 0.80</td></tr>
<tr><td>Equal opportunity</td><td>Are truly qualified people approved at similar rates?</td><td>TPR ratio of at least 0.80</td></tr>
<tr><td>Equalized odds</td><td>Are misses and false alarms similar across groups?</td><td>TPR and FPR gaps of at most 0.10</td></tr>
<tr><td>Predictive parity</td><td>Does a positive decision mean the same for every group?</td><td>Precision ratio of at least 0.80</td></tr>
</tbody></table>
<p class="muted">Groups below 100 records are marked low confidence; if they fail they are marked
"Review" rather than "Fail". Selection rates include 95% Wilson confidence intervals. The
fairness score blends average and worst-group compliance with the primary metric, minus a
penalty per failing group, and is capped below 70 whenever any group fails.</p>

<h2>4. Fairness results: single attributes</h2>{single}
<h2>5. Fairness results: intersectional groups</h2>{inter}
<h2>6. Drift analysis</h2>{drift_html}
<h2>7. Explainability</h2>{explain_html}
<h2>8. Limitations</h2><ul class="plain">{limitations}</ul>
<h2>9. Recommendations</h2><ul class="recs">{recs}</ul>

<h2>10. Audit trail and sign-off</h2>
<table class="scope"><tbody>
<tr><th>Report generated</th><td>{_e(report['generated'])}</td></tr>
<tr><th>Tool</th><td>{_e(report['tool_version'])} (Python {_e(platform.python_version())})</td></tr>
<tr><th>Groups tested</th><td>{s['groups_tested']} ({s['small_sample_groups']} with fewer than 100 records)</td></tr>
</tbody></table>
<table class="sign"><thead><tr><th>Role</th><th>Name</th><th>Signature</th><th>Date</th></tr></thead><tbody>
<tr><td>Model owner</td><td></td><td></td><td></td></tr>
<tr><td>Independent reviewer</td><td></td><td></td><td></td></tr>
<tr><td>Compliance / risk</td><td></td><td></td><td></td></tr>
</tbody></table>
</body></html>"""
