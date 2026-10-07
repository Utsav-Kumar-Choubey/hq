"""
server.py
---------
FastAPI server that exposes the audit engine to the web page.

Endpoints:
  GET  /                -> serves the FairLens front end (index.html)
  POST /upload          -> read a CSV, return its columns + suggested roles
  POST /audit           -> run the full audit, return JSON results
  GET  /advisor/{case}  -> return the recommended fairness metric for a use case
  POST /report          -> return the compliance report as a downloadable HTML file

Run:  uv run uvicorn server:app --reload --port 8000
Then open http://localhost:8000
"""

from __future__ import annotations

import io
import json

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from engine import data_loader
from engine.audit import run_full_audit, advise_metric

app = FastAPI(title="FairLens Bias Audit API")

# A tiny in-memory store so /audit can reuse the file uploaded to /upload.
# (Fine for a hackathon demo; a production app would use a session store.)
_LAST_REPORT_HTML = {"html": "<p>No report generated yet.</p>"}


def _read_upload(file: UploadFile) -> pd.DataFrame:
    try:
        raw = file.file.read()
        return data_loader.load_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not read '{file.filename}': {e}")


@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html") as f:
        return f.read()


@app.post("/upload")
async def upload(dataset: UploadFile = File(...)):
    """Read a CSV and return its columns + smart role suggestions for the form."""
    df = _read_upload(dataset)
    return JSONResponse({
        "filename": dataset.filename,
        "rows": int(len(df)),
        "columns": data_loader.describe_columns(df),
        "suggestions": data_loader.suggest_roles(df),
    })


@app.post("/audit")
async def audit(
    dataset: UploadFile = File(...),
    sensitive: str = Form(...),          # JSON list, e.g. '["gender","race"]'
    prediction_col: str = Form(...),
    outcome_col: str = Form(""),
    intersectional: bool = Form(True),
    model_name: str = Form("Uploaded model"),
    reference: UploadFile | None = File(None),
):
    """Run the full bias/drift/explainability audit and return JSON."""
    df = _read_upload(dataset)
    try:
        sensitive_list = json.loads(sensitive)
    except Exception:
        sensitive_list = [s.strip() for s in sensitive.split(",") if s.strip()]

    # Validate the chosen columns actually exist -> friendly error for the UI.
    missing = [c for c in sensitive_list + [prediction_col]
               if c and c not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found in CSV: {missing}")

    ref_df = _read_upload(reference) if reference is not None else None
    result = run_full_audit(
        df, sensitive=sensitive_list, pred_col=prediction_col,
        outcome_col=outcome_col or None, intersectional=intersectional,
        reference=ref_df, model_name=model_name)

    _LAST_REPORT_HTML["html"] = result["report"]["html"]
    # Drop the big HTML string from the JSON payload (fetched separately).
    result["report"].pop("html", None)
    return JSONResponse(result)


@app.get("/advisor/{usecase}")
def advisor(usecase: str):
    """Return the recommended fairness metric for a use case."""
    return JSONResponse(advise_metric(usecase))


@app.post("/report")
def report():
    """Return the last generated report as a downloadable HTML file."""
    return Response(
        content=_LAST_REPORT_HTML["html"],
        media_type="text/html",
        headers={"Content-Disposition":
                 "attachment; filename=fairlens_report.html"},
    )
