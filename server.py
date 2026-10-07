"""
server.py
---------
FastAPI server that exposes the FairLens audit engine to the web page.

Endpoints:
  GET  /                -> the FairLens front end (index.html)
  GET  /health          -> liveness check
  POST /upload          -> read a CSV, return its columns and suggested roles
  POST /audit           -> run the full audit and return JSON results
  GET  /advisor/{case}  -> recommended fairness metric for a use case
  POST /report          -> the latest compliance report as an HTML file

Run:  python3 -m uvicorn server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from engine import data_loader
from engine.audit import advise_metric, run_full_audit

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="FairLens Bias Audit API", version="1.1.0")

# Latest generated report. Adequate for a single-user local tool; a multi-user
# deployment would key reports by session.
_LAST_REPORT_HTML = {"html": None}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _read_csv_upload(file: UploadFile, label: str = "dataset") -> pd.DataFrame:
    name = file.filename or label
    if not name.lower().endswith(".csv"):
        raise HTTPException(400, f"The {label} must be a .csv file (received '{name}').")
    raw = file.file.read()
    if len(raw) > data_loader.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"The {label} exceeds the 50 MB limit.")
    try:
        return data_loader.load_csv(io.BytesIO(raw))
    except ValueError as exc:
        raise HTTPException(400, f"Could not read '{name}': {exc}")


def _parse_list(value: str) -> List[str]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v).strip()]
    except (TypeError, ValueError):
        pass
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _validate_roles(df: pd.DataFrame, sensitive: List[str], prediction_col: str,
                    outcome_col: Optional[str]) -> None:
    if not prediction_col:
        raise HTTPException(400, "Select the column that holds the model decision.")
    if not sensitive:
        raise HTTPException(400, "Select at least one sensitive attribute to audit.")
    requested = sensitive + [prediction_col] + ([outcome_col] if outcome_col else [])
    missing = [c for c in requested if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found in the CSV: {missing}")
    if prediction_col in sensitive or (outcome_col and outcome_col in sensitive):
        raise HTTPException(400, "A sensitive attribute cannot also be the "
                                 "decision or outcome column.")
    if outcome_col and outcome_col == prediction_col:
        raise HTTPException(400, "The outcome and decision columns must differ.")
    if not data_loader.is_binary_like(df[prediction_col]):
        raise HTTPException(400, f"'{prediction_col}' must be a yes/no decision or a "
                                 "probability between 0 and 1.")
    if outcome_col and not data_loader.is_binary_like(df[outcome_col]):
        raise HTTPException(400, f"'{outcome_col}' must be a yes/no outcome.")


# --------------------------------------------------------------------------
# static front end
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse(BASE_DIR / "index.html", media_type="text/html")


@app.get("/style.css")
def style():
    return FileResponse(BASE_DIR / "style.css", media_type="text/css")


@app.get("/app.js")
def appjs():
    return FileResponse(BASE_DIR / "app.js", media_type="application/javascript")


@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.post("/upload")
async def upload(dataset: UploadFile = File(...)):
    """Read a CSV and return its columns plus suggested roles for the form."""
    df = _read_csv_upload(dataset)
    return JSONResponse({
        "filename": dataset.filename,
        "rows": int(len(df)),
        "columns": data_loader.describe_columns(df),
        "suggestions": data_loader.suggest_roles(df),
    })


@app.post("/audit")
async def audit(
    dataset: UploadFile = File(...),
    sensitive: str = Form(...),
    prediction_col: str = Form(""),
    outcome_col: str = Form(""),
    intersectional: bool = Form(True),
    model_name: str = Form("Uploaded model"),
    reference: Optional[UploadFile] = File(None),
):
    """Run the bias, drift and explainability audit and return JSON."""
    df = _read_csv_upload(dataset)
    sensitive_list = _parse_list(sensitive)
    outcome = outcome_col or None
    _validate_roles(df, sensitive_list, prediction_col, outcome)

    ref_df = None
    if reference is not None and reference.filename:
        ref_df = _read_csv_upload(reference, "reference dataset")

    result = run_full_audit(
        df, sensitive=sensitive_list, pred_col=prediction_col,
        outcome_col=outcome, intersectional=intersectional,
        reference=ref_df, model_name=model_name or "Uploaded model")

    _LAST_REPORT_HTML["html"] = result["report"].pop("html", None)
    return JSONResponse(result)


@app.get("/advisor/{usecase}")
def advisor(usecase: str):
    """Return the recommended fairness metric for a use case."""
    return JSONResponse(advise_metric(usecase))


@app.post("/report")
def report():
    """Return the latest report as a downloadable HTML file."""
    if not _LAST_REPORT_HTML["html"]:
        raise HTTPException(404, "No report yet. Run an audit first.")
    return Response(
        content=_LAST_REPORT_HTML["html"],
        media_type="text/html",
        headers={"Content-Disposition": "attachment; filename=fairlens_report.html"},
    )
