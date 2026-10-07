"""
test_server.py
--------------
End-to-end test of the FastAPI endpoints using Starlette's TestClient.
This exercises the exact code paths the browser hits, without needing a
long-running server. Run:  uv run python test_server.py
"""

from fastapi.testclient import TestClient
from server import app

client = TestClient(app)
CSV = "demo/adult_income_predictions.csv"


def main():
    print("=== GET / (home page) ===")
    r = client.get("/")
    assert r.status_code == 200 and "FairLens" in r.text
    print("  home page served OK")

    print("\n=== POST /upload ===")
    with open(CSV, "rb") as f:
        r = client.post("/upload", files={"dataset": ("adult.csv", f, "text/csv")})
    assert r.status_code == 200, r.text
    up = r.json()
    print("  rows:", up["rows"])
    print("  suggested sensitive:", up["suggestions"]["suggested_sensitive"])
    print("  suggested prediction:", up["suggestions"]["suggested_prediction"])

    print("\n=== POST /audit ===")
    with open(CSV, "rb") as f:
        r = client.post("/audit",
            files={"dataset": ("adult.csv", f, "text/csv")},
            data={"sensitive": '["sex","race"]',
                  "prediction_col": "predicted_high_income",
                  "outcome_col": "income_over_50k",
                  "intersectional": "true",
                  "model_name": "Adult Income Demo"})
    assert r.status_code == 200, r.text
    res = r.json()
    s = res["fairness"]["summary"]
    print("  score:", s["fairness_score"], "verdict:", s["verdict"],
          "flagged:", s["groups_flagged"])
    print("  explanation:", res["explanations"][0].get("plain_language", "n/a"))
    print("  top rec:", res["recommendations"][0]["text"])
    assert "html" not in res["report"], "HTML should be stripped from JSON"

    print("\n=== POST /audit with bad column (error handling) ===")
    with open(CSV, "rb") as f:
        r = client.post("/audit",
            files={"dataset": ("adult.csv", f, "text/csv")},
            data={"sensitive": '["does_not_exist"]',
                  "prediction_col": "predicted_high_income"})
    assert r.status_code == 400
    print("  correctly rejected missing column:", r.json()["detail"])

    print("\n=== GET /advisor/lending ===")
    r = client.get("/advisor/lending")
    assert r.status_code == 200
    print("  recommended:", r.json()["metric"])

    print("\n=== POST /report (download) ===")
    r = client.post("/report")
    assert r.status_code == 200 and "<html" in r.text.lower()
    print("  report bytes:", len(r.text),
          "| filename:", r.headers.get("content-disposition"))

    print("\n" + "=" * 55)
    print("ALL SERVER ENDPOINTS PASSED - front end is wired to engine.")
    print("=" * 55)


if __name__ == "__main__":
    main()
