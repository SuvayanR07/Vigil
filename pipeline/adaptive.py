"""
Adaptive per-customer learning.

Two mechanisms:

  1. CUSTOM TERM DICTIONARY (custom_terms.json) — a lookup table keyed
     on a lowercased verbatim phrase. When the same term has been
     corrected by the user ≥ 2 times, it becomes authoritative: the
     coder short-circuits ChromaDB and returns the learned mapping
     with confidence 0.95.

  2. CUSTOMER-SPECIFIC CHROMADB COLLECTION — every 50 reports
     augment_embeddings() pulls the latest corrections into a
     per-customer collection (chroma_db/customers/{id}). The RAG query
     merges the global + customer-specific results with a 0.1
     similarity boost for customer matches.

Both mechanisms stay local to one customer. Nothing is shared.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_DB_PATH
from pipeline.customer import custom_terms_path, update_mapping_count
from pipeline.feedback import get_feedback_history


LEARNING_FREQUENCY_THRESHOLD = 2  # ≥2 corrections -> trust it
CUSTOMER_SIMILARITY_BOOST = 0.1
AUGMENT_EVERY_N_REPORTS = 50


# --------------------------------------------------------------------------- #
# Custom-terms dictionary                                                      #
# --------------------------------------------------------------------------- #

def _load_custom_terms(customer_id: str) -> dict:
    path = custom_terms_path(customer_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_custom_terms(customer_id: str, terms: dict) -> None:
    path = custom_terms_path(customer_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(terms, indent=2))
    update_mapping_count(customer_id, len(terms))


def _normalize_term(term: str) -> str:
    return (term or "").strip().lower()


def lookup_custom_mapping(customer_id: str, verbatim_term: str) -> Optional[dict]:
    """
    Return the authoritative mapping for this verbatim term if the user has
    taught the system (≥ LEARNING_FREQUENCY_THRESHOLD corrections). Otherwise None.
    """
    if not customer_id or not verbatim_term:
        return None
    terms = _load_custom_terms(customer_id)
    entry = terms.get(_normalize_term(verbatim_term))
    if not entry:
        return None
    if entry.get("frequency", 0) < LEARNING_FREQUENCY_THRESHOLD:
        return None
    return entry


def record_correction(
    customer_id: str,
    verbatim_term: str,
    corrected: dict,
) -> dict:
    """
    Record a user's MedDRA correction. Increments frequency if the same
    (verbatim_term -> pt_code) correction has been seen before.

    Args:
        customer_id:    Customer to learn for.
        verbatim_term:  Raw reaction term the user corrected.
        corrected:      Dict with keys pt_name, pt_code, soc_name (and
                        optionally hlt_name).

    Returns:
        The updated entry (useful for UI feedback).
    """
    if not customer_id or not verbatim_term or not corrected.get("pt_code"):
        return {}

    terms = _load_custom_terms(customer_id)
    key = _normalize_term(verbatim_term)
    prev = terms.get(key, {})

    # If the correction changed (user picked a different PT), reset frequency
    same_target = prev.get("pt_code") == corrected.get("pt_code")
    new_freq = (prev.get("frequency", 0) + 1) if same_target else 1

    entry = {
        "pt_name": corrected.get("pt_name", ""),
        "pt_code": corrected.get("pt_code", ""),
        "soc_name": corrected.get("soc_name", ""),
        "hlt_name": corrected.get("hlt_name", ""),
        "frequency": new_freq,
        "last_updated": datetime.utcnow().isoformat(timespec="seconds"),
    }
    terms[key] = entry
    _save_custom_terms(customer_id, terms)
    return entry


def get_custom_terms(customer_id: str) -> dict:
    """Expose the raw dictionary for analytics/UI display."""
    return _load_custom_terms(customer_id)


# --------------------------------------------------------------------------- #
# Customer-specific ChromaDB collection                                        #
# --------------------------------------------------------------------------- #

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = embedding_functions.DefaultEmbeddingFunction()
    return _embedder


def _customer_collection_path(customer_id: str) -> str:
    p = Path(CHROMA_DB_PATH) / "customers" / customer_id
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _get_customer_collection(customer_id: str, create: bool = False):
    """Return the per-customer ChromaDB collection, or None if it doesn't exist."""
    client = chromadb.PersistentClient(path=_customer_collection_path(customer_id))
    name = f"customer_{customer_id}"
    try:
        return client.get_collection(name=name)
    except Exception:
        if not create:
            return None
        return client.create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )


def augment_embeddings(customer_id: str) -> dict:
    """
    Pull all MedDRA corrections from feedback history and add them to
    the customer's dedicated ChromaDB collection.

    Returns a small summary dict for logging.
    """
    if not customer_id:
        return {"added": 0, "reason": "no customer_id"}

    feedback_records = get_feedback_history(customer_id)
    pairs: list[tuple[str, dict]] = []  # (verbatim_term, corrected_dict)
    for rec in feedback_records:
        for corr in rec.get("corrections", []):
            if corr.get("field_type") != "meddra":
                continue
            verb = corr.get("verbatim_term", "").strip()
            corrected = corr.get("corrected") or {}
            if verb and corrected.get("pt_code"):
                pairs.append((verb, corrected))

    if not pairs:
        return {"added": 0, "reason": "no meddra corrections"}

    # Re-create the collection from scratch so it stays in sync with the
    # current state of custom_terms (no stale entries).
    client = chromadb.PersistentClient(path=_customer_collection_path(customer_id))
    name = f"customer_{customer_id}"
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        client.delete_collection(name)
    collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    embedder = _get_embedder()
    documents = [f"{c['pt_name']} | {v}" for v, c in pairs]
    embeddings = embedder(documents)
    ids = [f"cust_{i}" for i in range(len(documents))]
    metadatas = [
        {
            "pt_name": c["pt_name"],
            "pt_code": c["pt_code"],
            "soc_name": c.get("soc_name", ""),
            "hlt_name": c.get("hlt_name", ""),
            "source": "customer_feedback",
        }
        for _, c in pairs
    ]
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return {"added": len(pairs), "reason": "ok"}


def query_customer_collection(customer_id: str, verbatim_term: str, n_results: int = 3) -> list[dict]:
    """
    Query the per-customer collection. Returns candidates with a
    CUSTOMER_SIMILARITY_BOOST already applied. Empty list if no collection
    or no results.
    """
    if not customer_id or not verbatim_term:
        return []

    collection = _get_customer_collection(customer_id, create=False)
    if collection is None:
        return []

    try:
        embedder = _get_embedder()
        query_embedding = embedder([verbatim_term.strip()])
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["metadatas", "distances"],
        )
    except Exception:
        return []

    out: list[dict] = []
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    for meta, dist in zip(metadatas, distances):
        base_sim = round(1 - (dist / 2), 4)
        boosted = min(1.0, base_sim + CUSTOMER_SIMILARITY_BOOST)
        out.append({
            "pt_name": meta.get("pt_name", ""),
            "pt_code": meta.get("pt_code", ""),
            "soc_name": meta.get("soc_name", ""),
            "hlt_name": meta.get("hlt_name", ""),
            "similarity": round(boosted, 4),
            "source": "customer",
        })
    return out


def maybe_augment_after_report(customer_id: str, report_count: int) -> Optional[dict]:
    """
    Call after saving each report. Triggers augment_embeddings() every
    AUGMENT_EVERY_N_REPORTS. Returns the augment summary if it ran.
    """
    if not customer_id or report_count <= 0:
        return None
    if report_count % AUGMENT_EVERY_N_REPORTS != 0:
        return None
    return augment_embeddings(customer_id)
