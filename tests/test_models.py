import io
import pickle

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from engine import model_loader
from server import app

client = TestClient(app)
FEATURES = ["income", "loan_amount", "employment_years"]


def _pipeline(loans):
    X = loans[FEATURES + ["age_band"]]
    pipe = Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["age_band"]),
                                    ("num", StandardScaler(), FEATURES)])),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    return pipe.fit(X, loans["loan_repaid"])


def _bytes(obj) -> bytes:
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()


def _csv(df) -> bytes:
    return df.to_csv(index=False).encode()


def test_apply_model_adds_prediction_and_score(loans):
    out = model_loader.apply_model(_pipeline(loans), loans, [])
    assert set(out[model_loader.PRED_COL].unique()) <= {0, 1}
    assert out[model_loader.SCORE_COL].between(0, 1).all()


def test_model_missing_columns_is_reported(loans):
    with pytest.raises(model_loader.ModelError, match="expects columns"):
        model_loader.apply_model(_pipeline(loans), loans.drop(columns=["income"]), [])


def test_non_model_object_rejected():
    with pytest.raises(model_loader.ModelError, match="predict"):
        model_loader.load_model(pickle.dumps({"not": "a model"}), "x.pkl")


def test_audit_with_uploaded_model(loans):
    data = loans.drop(columns=["predicted_approval"])
    r = client.post("/audit",
                    data={"sensitive": '["gender"]', "outcome_col": "loan_repaid"},
                    files={"dataset": ("loans.csv", _csv(data), "text/csv"),
                           "model": ("model.joblib", _bytes(_pipeline(loans)),
                                     "application/octet-stream")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"]["prediction_column"] == "model_prediction"
    assert "model score" not in body["explainability"]["features"]


def test_upload_reports_model_applied(loans):
    data = loans.drop(columns=["predicted_approval"])
    r = client.post("/upload", files={
        "dataset": ("loans.csv", _csv(data), "text/csv"),
        "model": ("model.pkl", pickle.dumps(_pipeline(loans)), "application/octet-stream")})
    assert r.status_code == 200, r.text
    assert r.json()["model_applied"] is True
    assert r.json()["suggestions"]["suggested_prediction"] == "model_prediction"


def test_separate_predictions_csv_is_joined(loans):
    data = loans.drop(columns=["predicted_approval"])
    preds = loans[["predicted_approval"]]
    r = client.post("/audit",
                    data={"sensitive": '["gender"]', "prediction_col": "predicted_approval",
                          "outcome_col": "loan_repaid"},
                    files={"dataset": ("loans.csv", _csv(data), "text/csv"),
                           "model": ("preds.csv", _csv(preds), "text/csv")})
    assert r.status_code == 200, r.text


def test_predictions_row_mismatch_rejected(loans):
    r = client.post("/upload", files={
        "dataset": ("loans.csv", _csv(loans), "text/csv"),
        "model": ("preds.csv", _csv(pd.DataFrame({"p": [0, 1]})), "text/csv")})
    assert r.status_code == 400 and "line up" in r.json()["detail"]


def test_unsupported_model_extension_rejected(loans):
    r = client.post("/upload", files={
        "dataset": ("loans.csv", _csv(loans), "text/csv"),
        "model": ("model.h5", b"xx", "application/octet-stream")})
    assert r.status_code == 400 and ".joblib" in r.json()["detail"]
