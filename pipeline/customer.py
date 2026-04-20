"""
Per-customer profile management.

Each clinic/pharmacy gets an isolated directory:
  data/customers/{customer_id}/
    profile.json         — Customer metadata
    custom_terms.json    — Learned term → MedDRA mappings
    reports/             — Classified report history
    feedback/            — User corrections
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR


CUSTOMERS_ROOT = DATA_DIR / "customers"


class Customer(BaseModel):
    customer_id: str
    name: str
    created_at: str
    reports_processed: int = 0
    custom_mappings_count: int = 0


# --------------------------------------------------------------------------- #
# Path helpers                                                                 #
# --------------------------------------------------------------------------- #

def customer_dir(customer_id: str) -> Path:
    return CUSTOMERS_ROOT / customer_id


def profile_path(customer_id: str) -> Path:
    return customer_dir(customer_id) / "profile.json"


def custom_terms_path(customer_id: str) -> Path:
    return customer_dir(customer_id) / "custom_terms.json"


def reports_dir(customer_id: str) -> Path:
    return customer_dir(customer_id) / "reports"


def feedback_dir(customer_id: str) -> Path:
    return customer_dir(customer_id) / "feedback"


def _ensure_dirs(customer_id: str) -> None:
    customer_dir(customer_id).mkdir(parents=True, exist_ok=True)
    reports_dir(customer_id).mkdir(parents=True, exist_ok=True)
    feedback_dir(customer_id).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def create_customer(name: str) -> Customer:
    """Create a new customer profile and scaffold its data dirs."""
    cid = uuid.uuid4().hex[:12]
    _ensure_dirs(cid)

    customer = Customer(
        customer_id=cid,
        name=name.strip() or "Unnamed Organization",
        created_at=datetime.utcnow().isoformat(timespec="seconds"),
        reports_processed=0,
        custom_mappings_count=0,
    )
    _save_profile(customer)

    # Initialize empty custom_terms file so later reads don't error
    custom_terms_path(cid).write_text("{}")
    return customer


def load_customer(customer_id: str) -> Optional[Customer]:
    p = profile_path(customer_id)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return Customer(**data)
    except Exception:
        return None


def list_customers() -> list[Customer]:
    if not CUSTOMERS_ROOT.exists():
        return []
    out = []
    for child in sorted(CUSTOMERS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        c = load_customer(child.name)
        if c:
            out.append(c)
    return out


def increment_reports(customer_id: str) -> None:
    c = load_customer(customer_id)
    if not c:
        return
    c.reports_processed += 1
    _save_profile(c)


def update_mapping_count(customer_id: str, count: int) -> None:
    c = load_customer(customer_id)
    if not c:
        return
    c.custom_mappings_count = count
    _save_profile(c)


def _save_profile(customer: Customer) -> None:
    _ensure_dirs(customer.customer_id)
    profile_path(customer.customer_id).write_text(
        json.dumps(customer.model_dump(), indent=2)
    )
