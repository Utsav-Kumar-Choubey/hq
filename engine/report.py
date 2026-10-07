"""
report.py
---------
Turns the audit results into (a) a plain-language recommendations list and
(b) a self-contained HTML compliance report that can be downloaded or printed
to PDF from the browser.
"""

from __future__ import annotations

from datetime import date


def build_recommendations(fairness: dict, drift: dict) -> list[dict]:
    """Produce prioritised, human recommendations from the numbers."""
    recs = []
    s = fairness["summary"]

    # Flag the worst-performing group.
    worst_group = None
    for rows in list(fairness["single"].values()) + [fairness["intersectional"]]:
        for r in rows:
            if r["status"] == "fail":
                if worst_group is None or r["dp_ratio"] < worst_group["dp_ratio"]:
                    worst_group = r
    if worst_group:
        recs.append({
            "priority": "high",
            "text": (f"Review the selection gap for '{worst_group['group']}' "
                     f"(ratio {worst_group['dp_ratio']} vs the 0.80 threshold)."),
        })

    if drift["n_drifted"] > 0:
        top = drift["features"][0]
        recs.append({
            "priority": "medium",
            "text": (f"Retrain with recent data: '{top['feature']}' has drifted "
                     f"(PSI {top['psi']}, {top['band']})."),
        })

    if s["small_sample_groups"] > 0:
        recs.append({
            "priority": "medium",
            "text": (f"Collect more data for {s['small_sample_groups']} group(s) "
                     f"with fewer than 100 people; results there are low-confidence."),
        })

    recs.append({"priority": "low",
                 "text": "Schedule a fairness and drift re-check every quarter."})
    return recs


def build_report(model_name: str, dataset_info: dict, fairness: dict,
                 drift: dict, recommendations: list[dict]) -> dict:
    """Return a structured report (JSON) plus a rendered HTML string."""
    s = fairness["summary"]
    report = {
        "model_name": model_name,
        "generated": date.today().isoformat(),
        "dataset": dataset_info,
        "fairness_summary": s,
        "drift_summary": {"overall": drift["overall_band"],
                          "worst_psi": drift["worst_psi"]},
        "recommendations": recommendations,
    }
    report["html"] = _render_html(report, fairness, drift)
    return report


def _badge(priority: str) -> str:
    colors = {"high": "#b3261e", "medium": "#9a6200", "low": "#1a7f4b"}
    return (f'<span style="background:{colors.get(priority, "#555")};color:#fff;'
            f'padding:2px 8px;border-radius:999px;font-size:12px">{priority.upper()}</span>')


def _render_html(report: dict, fairness: dict, drift: dict) -> str:
    s = report["fairness_summary"]
    rows_html = ""
    for rows in list(fairness["single"].values()) + [fairness["intersectional"]]:
        for r in rows:
            rows_html += (
                f"<tr><td>{r['group']}</td><td>{r['size']}</td>"
                f"<td>{r['selection_rate']:.0%}</td><td>{r['dp_ratio']}</td>"
                f"<td>{r['status']}</td></tr>")
    drift_html = "".join(
        f"<tr><td>{d['feature']}</td><td>{d['psi']}</td><td>{d['band']}</td></tr>"
        for d in drift["features"])
    recs_html = "".join(
        f"<li>{_badge(r['priority'])} {r['text']}</li>"
        for r in report["recommendations"])

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Fairness Audit: {report['model_name']}</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:820px;margin:40px auto;color:#1c2433;line-height:1.6}}
 h1{{border-top:5px solid #14213d;padding-top:12px}}
 table{{border-collapse:collapse;width:100%;margin:12px 0}}
 th,td{{border:1px solid #dfe4ec;padding:8px 10px;text-align:left;font-size:14px}}
 th{{background:#f3f6fb}} li{{margin:8px 0;list-style:none}}
 .score{{font-size:42px;font-weight:700;color:#14213d}}
</style></head><body>
<h1>Model Fairness Audit: {report['model_name']}</h1>
<p>Generated {report['generated']} &middot; FairLens Bias Auditing &amp; Explainability Toolkit</p>
<h2>1. Executive summary</h2>
<p class="score">{s['fairness_score']}/100</p>
<p>Verdict: <strong>{s['verdict'].replace('_',' ')}</strong>.
{s['groups_tested']} groups tested, {s['groups_flagged']} flagged,
drift level {report['drift_summary']['overall']}.</p>
<h2>2. Fairness results (single &amp; intersectional)</h2>
<table><thead><tr><th>Group</th><th>Size</th><th>Selection rate</th>
<th>Ratio to top group</th><th>Status</th></tr></thead><tbody>{rows_html}</tbody></table>
<h2>3. Drift analysis</h2>
<table><thead><tr><th>Feature</th><th>PSI</th><th>Band</th></tr></thead>
<tbody>{drift_html}</tbody></table>
<h2>4. Limitations</h2>
<p>Fairness metrics can conflict; explanations describe model behaviour, not
ground-truth fairness. Groups under 100 people are low-confidence.</p>
<h2>5. Recommendations</h2>
<ul>{recs_html}</ul>
</body></html>"""
