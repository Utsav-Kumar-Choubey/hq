import json

from fastapi.testclient import TestClient

from engine.audit import run_full_audit
from server import app

client = TestClient(app)

SECTIONS = ["1. Executive summary", "2. Scope: data and model", "3. Method and metric selection",
            "4. Fairness results: single attributes", "5. Fairness results: intersectional groups",
            "6. Drift analysis", "7. Explainability", "8. Limitations", "9. Recommendations",
            "10. Audit trail and sign-off"]


def test_report_contains_every_section(loans):
    ref = loans.copy()
    ref["income"] = ref["income"] * 0.7
    result = run_full_audit(loans, ["gender", "age_band"], "predicted_approval", "loan_repaid",
                            reference=ref, use_case="lending", model_name="Loan model")
    page = result["report"]["html"]
    for section in SECTIONS:
        assert section in page
    assert "Equal opportunity" in page
    assert "Record #" in page            # example explanations
    assert "significant" in page.lower()  # drift table
    assert result["report"]["key_findings"]


def test_report_escapes_untrusted_values(loans):
    loans["gender"] = loans["gender"].replace({"Female": "<script>alert(1)</script>"})
    result = run_full_audit(loans, ["gender"], "predicted_approval", "loan_repaid",
                            model_name="<img src=x onerror=alert(1)>")
    page = result["report"]["html"]
    assert "<script>alert(1)</script>" not in page
    assert "<img src=x" not in page
    assert "&lt;script&gt;" in page


def test_report_without_outcome_or_reference(loans):
    result = run_full_audit(loans, ["gender"], "predicted_approval", None)
    page = result["report"]["html"]
    assert "Not tested" in page
    assert "not supplied" in page


def test_report_endpoints(loans):
    data = {"sensitive": '["gender","age_band"]', "prediction_col": "predicted_approval",
            "outcome_col": "loan_repaid"}
    files = {"dataset": ("loans.csv", loans.to_csv(index=False).encode(), "text/csv")}
    assert client.post("/audit", data=data, files=files).status_code == 200

    view = client.get("/report/view?print_dialog=true")
    assert view.status_code == 200 and "window.print()" in view.text
    assert "window.print()" not in client.get("/report/view").text

    download = client.get("/report/download")
    assert "attachment" in download.headers["content-disposition"]

    evidence = client.get("/report.json")
    assert evidence.status_code == 200
    body = json.loads(evidence.text)
    assert body["fairness"]["summary"]["groups_tested"] > 0
