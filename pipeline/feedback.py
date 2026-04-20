"""
Per-customer human-feedback store.

Each feedback record captures the user's corrections for a single report.
These records drive the adaptive learning loop in pipeline/adaptive.py.

File layout:
  data/customers/{customer_id}/feedback/{report_id}.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.customer import feedback_dir


# A correction record is a dict with schema:
#   { "field_type": "meddra" | "severity" | "drug",
#     "verbatim_term": "...",           # for meddra
#     "original":  {...},
#     "corrected": {...} }


def save_feedback(customer_id: str, report_id: str, corrections: list[dict]) -> None:
    """Persist corrections for a single report."""
    path = feedback_dir(customer_id) / f"{report_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "report_id": report_id,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "corrections": corrections,
    }
    path.write_text(json.dumps(payload, indent=2))


def get_feedback(customer_id: str, report_id: str) -> dict | None:
    path = feedback_dir(customer_id) / f"{report_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_feedback_history(customer_id: str) -> list[dict]:
    """Return all feedback records for a customer, oldest-first."""
    d = feedback_dir(customer_id)
    if not d.exists():
        return []
    out: list[dict] = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out
