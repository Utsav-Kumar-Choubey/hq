"""
model_loader.py
---------------
Lets users upload a trained model instead of (or as well as) a predictions
column. Supported formats:

  .pkl / .pickle / .joblib   scikit-learn compatible estimators or pipelines
  .onnx                      ONNX models (requires the optional onnxruntime)

The model is applied to the uploaded dataset to produce two new columns:
  model_prediction   the model's 0/1 decision
  model_score        the positive-class probability, when available

SECURITY: unpickling executes code embedded in the file. Only upload models
from a trusted source. FairLens is intended to run locally on your machine.
"""

from __future__ import annotations

import io
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd

from .data_loader import clean_features, feature_columns

MODEL_EXTENSIONS = (".pkl", ".pickle", ".joblib", ".onnx")
PRED_COL = "model_prediction"
SCORE_COL = "model_score"


class ModelError(ValueError):
    """Raised with a user-facing message when a model cannot be used."""


def is_model_file(filename: str) -> bool:
    return (filename or "").lower().endswith(MODEL_EXTENSIONS)


def load_model(raw: bytes, filename: str):
    name = (filename or "").lower()
    if name.endswith(".onnx"):
        try:
            import onnxruntime as ort  # optional dependency
        except ImportError:
            raise ModelError("ONNX models need the optional 'onnxruntime' package "
                             "(pip install onnxruntime).")
        try:
            return _OnnxModel(ort.InferenceSession(raw))
        except Exception as exc:
            raise ModelError(f"The ONNX model could not be loaded ({exc}).")
    try:
        import joblib
        model = joblib.load(io.BytesIO(raw))
    except Exception:
        try:
            model = pickle.loads(raw)
        except Exception as exc:
            raise ModelError("The model file could not be loaded. Save it with "
                             f"joblib.dump or pickle.dump ({exc.__class__.__name__}).")
    if not hasattr(model, "predict"):
        raise ModelError("The uploaded object has no predict() method, so it is "
                         "not a usable model.")
    return model


class _OnnxModel:
    """Minimal predict/predict_proba wrapper around an ONNX session."""

    def __init__(self, session):
        self.session = session
        self.inputs = session.get_inputs()

    @property
    def feature_names_in_(self):
        if len(self.inputs) > 1:
            return np.array([i.name for i in self.inputs])
        return None

    def _run(self, X: pd.DataFrame):
        if len(self.inputs) > 1:
            feeds = {i.name: X[[i.name]].to_numpy().astype(
                np.float32 if X[i.name].dtype.kind in "fiu" else object)
                for i in self.inputs}
        else:
            feeds = {self.inputs[0].name: X.to_numpy().astype(np.float32)}
        return self.session.run(None, feeds)

    def predict(self, X):
        return np.asarray(self._run(X)[0]).ravel()

    def predict_proba(self, X):
        out = self._run(X)
        if len(out) < 2:
            raise AttributeError("no probabilities")
        probs = out[1]
        if isinstance(probs, list) and probs and isinstance(probs[0], dict):
            keys = sorted(probs[0])
            return np.array([[p[k] for k in keys] for p in probs])
        return np.asarray(probs)


def _model_features(model, df: pd.DataFrame, exclude: List[Optional[str]]) -> List[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        names = [str(n) for n in names]
        missing = [n for n in names if n not in df.columns]
        if missing:
            raise ModelError(f"The model expects columns that are not in the dataset: "
                             f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
        return names
    return feature_columns(df, exclude)


def _positive_index(model) -> int:
    classes = list(getattr(model, "classes_", [0, 1]))
    for positive in (1, True, "1", "yes", "Yes", ">50K", "approved", "good"):
        if positive in classes:
            return classes.index(positive)
    return len(classes) - 1


def apply_model(model, df: pd.DataFrame, exclude: List[Optional[str]]) -> pd.DataFrame:
    """Return a copy of df with model_prediction (and model_score) added."""
    features = _model_features(model, df, exclude)
    X = clean_features(df[features])
    try:
        labels = model.predict(X)
    except Exception as exc:
        raise ModelError(f"The model failed to predict on this dataset ({exc}). Check "
                         "that the dataset has the same columns the model was trained on.")
    out = df.copy()
    score = None
    try:
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            score = proba[:, _positive_index(model)]
    except Exception:
        score = None
    if score is not None:
        out[SCORE_COL] = np.round(score, 4)
        out[PRED_COL] = (score >= 0.5).astype(int)
    else:
        labels = pd.Series(np.asarray(labels).ravel())
        classes = list(getattr(model, "classes_", sorted(labels.unique())))
        positive = classes[_positive_index(model)] if classes else 1
        out[PRED_COL] = (labels == positive).astype(int).to_numpy()
    return out


def merge_predictions(df: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Attach a separate predictions CSV to the dataset, row by row."""
    if len(preds) != len(df):
        raise ModelError(f"The predictions file has {len(preds):,} rows but the dataset "
                         f"has {len(df):,}. They must line up row by row.")
    new_cols = [c for c in preds.columns if c not in df.columns]
    if not new_cols:
        raise ModelError("The predictions file has no columns that are not already in "
                         "the dataset.")
    out = df.reset_index(drop=True).copy()
    for c in new_cols:
        out[c] = preds[c].to_numpy()
    return out
