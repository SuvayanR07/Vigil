"""
Per-customer report history.

Each classified report is persisted to:
  data/customers/{customer_id}/reports/{timestamp}_{hash}.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.customer import customer_dir, reports_dir


def _report_id(narrative: str) -> str:
    """Stable ID: YYYYMMDDHHMMSS_{8-char narrative hash}."""
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    h = hashlib.sha1(narrative.encode("utf-8")).hexdigest()[:8]
    return f"{ts}_{h}"


def save_report(customer_id: str, narrative: str, report: dict) -> str:
    """
    Persist a classified report for a customer.

    Returns the report_id so the caller can attach feedback later.
    """
    rid = _report_id(narrative)
    path = reports_dir(customer_id) / f"{rid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "report_id": rid,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "narrative": narrative,
        "report": report,
    }
    path.write_text(json.dumps(payload, indent=2))
    return rid


def get_report(customer_id: str, report_id: str) -> Optional[dict]:
    path = reports_dir(customer_id) / f"{report_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_report_history(customer_id: str) -> list[dict]:
    """Return all reports for a customer, sorted newest-first."""
    d = reports_dir(customer_id)
    if not d.exists():
        return []
    entries: list[dict] = []
    for p in sorted(d.glob("*.json"), reverse=True):
        try:
            entries.append(json.loads(p.read_text()))
        except Exception:
            continue
    return entries


def get_report_count(customer_id: str) -> int:
    d = reports_dir(customer_id)
    if not d.exists():
        return 0
    return len(list(d.glob("*.json")))
