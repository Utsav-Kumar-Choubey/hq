import io

import pytest
from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def _csv_bytes(df) -> bytes:
    return df.to_csv(index=False).encode()


def _audit(df, **fields):
    data = {"sensitive": '["gender","age_band"]',
            "prediction_col": "predicted_approval",
            "outcome_col": "loan_repaid", "intersectional": "true"}
    data.update(fields)
    return client.post("/audit", data=data,
                       files={"dataset": ("loans.csv", _csv_bytes(df), "text/csv")})


@pytest.mark.parametrize("path,needle", [("/", "FairLens"), ("/style.css", "--navy"),
                                         ("/app.js", "onRunAudit"), ("/health", "ok")])
def test_static_and_health(path, needle):
    r = client.get(path)
    assert r.status_code == 200 and needle in r.text


def test_upload_returns_columns(loans):
    r = client.post("/upload", files={"dataset": ("loans.csv", _csv_bytes(loans), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["rows"] == len(loans)
    assert body["suggestions"]["suggested_prediction"] == "predicted_approval"


def test_upload_rejects_non_csv():
    r = client.post("/upload", files={"dataset": ("model.pkl", b"\x80\x04", "application/octet-stream")})
    assert r.status_code == 400 and ".csv" in r.json()["detail"]


def test_upload_rejects_empty_csv():
    r = client.post("/upload", files={"dataset": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 400


def test_audit_end_to_end_and_report(loans):
    r = _audit(loans)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fairness"]["summary"]["groups_flagged"] >= 1
    assert "html" not in body["report"]
    rep = client.post("/report")
    assert rep.status_code == 200 and "<html" in rep.text.lower()


def test_audit_with_reference_detects_drift(loans):
    ref = loans.copy()
    ref["income"] = ref["income"] * 0.7
    r = client.post("/audit",
                    data={"sensitive": '["gender"]', "prediction_col": "predicted_approval",
                          "outcome_col": "loan_repaid"},
                    files={"dataset": ("loans.csv", _csv_bytes(loans), "text/csv"),
                           "reference": ("ref.csv", _csv_bytes(ref), "text/csv")})
    assert r.status_code == 200, r.text
    assert r.json()["drift"]["overall_band"] == "significant"


@pytest.mark.parametrize("fields,fragment", [
    ({"sensitive": '["nope"]'}, "not found"),
    ({"sensitive": "[]"}, "at least one"),
    ({"prediction_col": ""}, "decision"),
    ({"prediction_col": "income"}, "yes/no"),
    ({"sensitive": '["predicted_approval"]'}, "cannot also be"),
])
def test_audit_validation_errors(loans, fields, fragment):
    r = _audit(loans, **fields)
    assert r.status_code == 400
    assert fragment in r.json()["detail"]


def test_advisor_endpoint():
    assert client.get("/advisor/lending").json()["metric"] == "Equal opportunity"


def test_metrics_catalogue_endpoint():
    body = client.get("/metrics").json()
    assert {m["id"] for m in body["metrics"]} == {
        "demographic_parity", "equal_opportunity", "equalized_odds", "predictive_parity"}
    assert set(body["use_cases"]) == {"hiring", "lending", "triage", "scoring"}


def test_audit_honours_metric_field(loans):
    r = _audit(loans, metric="predictive_parity")
    assert r.status_code == 200
    assert r.json()["fairness"]["metric"]["id"] == "predictive_parity"


def test_explain_endpoint_after_audit(loans):
    assert _audit(loans).status_code == 200
    r = client.get("/explain/5")
    assert r.status_code == 200
    body = r.json()
    assert body["index"] == 5 and body["plain_language"]
    assert client.get(f"/explain/{len(loans) + 10}").status_code == 404


def test_audit_returns_explainability_summary(loans):
    body = _audit(loans).json()
    info = body["explainability"]
    assert info["available"] and info["global_importance"]
    assert "applicant_id" not in info["features"]
