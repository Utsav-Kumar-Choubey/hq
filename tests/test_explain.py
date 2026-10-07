import numpy as np
import pandas as pd
import pytest

from engine.explain import build_explainer, find_proxies

FEATURES = ["income", "loan_amount", "employment_years"]


def test_explanation_is_plain_language(loans):
    ex = build_explainer(loans, FEATURES, "predicted_approval")
    out = ex.explain(0)
    assert out["decision"] in {"approved", "rejected"}
    assert out["plain_language"].startswith("The model ")
    assert all("_" not in f["label"] for f in out["factors"])
    assert 0.8 <= out["fidelity"] <= 1.0


def test_categorical_features_are_folded():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"job": rng.choice(["clerk", "manager", "driver"], 800),
                       "score": rng.normal(0, 1, 800)})
    df["pred"] = ((df["job"] == "manager") | (df["score"] > 1)).astype(int)
    ex = build_explainer(df, ["job", "score"], "pred")
    features = {f["feature"] for f in ex.explain(0)["factors"]}
    assert features <= {"job", "score"}


def test_counterfactual_flips_decision(loans):
    ex = build_explainer(loans, FEATURES, "predicted_approval")
    idx = int(np.flatnonzero((ex.y == 0) & (ex.surrogate_pred == 0))[0])
    cfs = ex.explain(idx)["counterfactuals"]
    assert cfs, "expected at least one counterfactual"
    trial = ex.raw.iloc[[idx]].copy()
    trial[cfs[0]["feature"]] = float(cfs[0]["to"].replace(",", ""))
    assert int(ex.model.predict(ex._encode(trial))[0]) == 1


def test_explain_rejects_bad_index(loans):
    ex = build_explainer(loans, FEATURES, "predicted_approval")
    with pytest.raises(IndexError):
        ex.explain(len(loans))


def test_constant_predictions_raise():
    df = pd.DataFrame({"x": range(50), "pred": [1] * 50})
    with pytest.raises(ValueError):
        build_explainer(df, ["x"], "pred")


def test_proxy_detection():
    rng = np.random.default_rng(5)
    sex = rng.choice(["M", "F"], 2000)
    df = pd.DataFrame({"sex": sex,
                       "title": np.where(sex == "M", "Mr", "Ms"),
                       "noise": rng.choice(["a", "b"], 2000)})
    found = find_proxies(df, ["sex"], ["title", "noise"])
    assert [p["feature"] for p in found] == ["title"]
