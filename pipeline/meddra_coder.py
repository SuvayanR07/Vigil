"""
MedDRA coding pipeline using ChromaDB RAG + Ollama selection.

Two-step process:
  1. query_meddra()    - embed verbatim term, retrieve top-5 candidates from ChromaDB
  2. select_best_match() - send candidates to Ollama, pick the single best PT
"""

import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    TOP_K_CANDIDATES,
    CONFIDENCE_THRESHOLD,
)
from pipeline.ollama_client import generate

# --------------------------------------------------------------------------- #
# ChromaDB + embedding singletons                                              #
# Uses ChromaDB's built-in ONNX all-MiniLM-L6-v2 (no torch, no segfaults)      #
# --------------------------------------------------------------------------- #

_collection = None
_embedder = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    return _collection


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = embedding_functions.DefaultEmbeddingFunction()
    return _embedder


# --------------------------------------------------------------------------- #
# Step 1: retrieve top-K candidates                                            #
# --------------------------------------------------------------------------- #

def query_meddra(verbatim_term: str) -> list[dict]:
    """
    Embed a verbatim adverse event term and retrieve the top-K MedDRA
    Preferred Term candidates from ChromaDB.

    Args:
        verbatim_term: Raw text describing the adverse event
                       (e.g. "felt dizzy", "heart was racing").

    Returns:
        List of up to TOP_K_CANDIDATES dicts, each containing:
          - pt_name    : MedDRA Preferred Term name
          - pt_code    : MedDRA PT code
          - soc_name   : System Organ Class name
          - hlt_name   : High Level Term name
          - similarity : cosine similarity score (0–1, higher = closer)
          - rank       : 1-based rank
    """
    if not verbatim_term or not verbatim_term.strip():
        return []

    collection = _get_collection()
    embedder = _get_embedder()
    query_embedding = embedder([verbatim_term.strip()])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K_CANDIDATES,
        include=["metadatas", "distances"],
    )

    candidates = []
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]  # ChromaDB cosine: lower = closer (0–2 range)

    for rank, (meta, dist) in enumerate(zip(metadatas, distances), start=1):
        # Convert distance to similarity: cosine distance in [0,2] -> similarity in [0,1]
        similarity = round(1 - (dist / 2), 4)
        candidates.append({
            "pt_name": meta.get("pt_name", ""),
            "pt_code": meta.get("pt_code", ""),
            "soc_name": meta.get("soc_name", ""),
            "hlt_name": meta.get("hlt_name", ""),
            "similarity": similarity,
            "rank": rank,
        })

    return candidates


# --------------------------------------------------------------------------- #
# Step 2: Ollama selection from candidates                                     #
# --------------------------------------------------------------------------- #

SELECTION_SYSTEM_PROMPT = """You are a medical coding assistant specializing in MedDRA terminology.
Your task is to select the single best MedDRA Preferred Term (PT) for an adverse event.

CRITICAL RULES:
1. Prefer the MOST LITERAL and MOST GENERAL match. Do not assume extra clinical detail.
2. "headache" -> Headache (NOT Migraine, unless "throbbing/one-sided" is stated).
3. "tired/fatigue" -> Fatigue (NOT Somnolence, unless "sleepy/drowsy" is stated).
4. "couldn't sleep" -> Insomnia (NOT Sleep apnea, unless "stopped breathing" is stated).
5. "heart racing/pounding" -> Tachycardia or Palpitations (NOT Cardiac arrest).
6. When the top candidate already matches the verbatim term literally, PICK IT.
7. Only pick a more specific term if the verbatim text explicitly supports it.

Always respond in the exact format requested."""


def select_best_match(
    verbatim_term: str,
    candidates: list[dict],
    clinical_context: str = "",
) -> dict:
    """
    Ask Ollama to select the best MedDRA PT from the top-K candidates.

    Args:
        verbatim_term     : Original verbatim text (e.g. "felt dizzy").
        candidates        : List of dicts from query_meddra().
        clinical_context  : Optional surrounding narrative for disambiguation.

    Returns:
        Dict with keys:
          - verbatim_term  : original input
          - pt_name        : selected MedDRA Preferred Term
          - pt_code        : MedDRA PT code
          - soc_name       : System Organ Class
          - hlt_name       : High Level Term
          - confidence     : float 0–1 (blended from similarity + Ollama choice)
          - candidates     : full candidate list (for audit/display)
          - selection_method: "ollama" or "top1_fallback"
    """
    if not candidates:
        return _empty_result(verbatim_term)

    # If only one candidate, skip LLM
    if len(candidates) == 1:
        c = candidates[0]
        return {
            "verbatim_term": verbatim_term,
            "pt_name": c["pt_name"],
            "pt_code": c["pt_code"],
            "soc_name": c["soc_name"],
            "hlt_name": c["hlt_name"],
            "confidence": c["similarity"],
            "candidates": candidates,
            "selection_method": "top1_fallback",
        }

    # Shortcut: if top-1 is both high-confidence AND clearly dominant over top-2,
    # trust the RAG result. Gemma 2B tends to over-specify on obvious cases.
    top1_sim = candidates[0]["similarity"]
    top2_sim = candidates[1]["similarity"]
    if top1_sim >= 0.85 and (top1_sim - top2_sim) >= 0.05:
        c = candidates[0]
        return {
            "verbatim_term": verbatim_term,
            "pt_name": c["pt_name"],
            "pt_code": c["pt_code"],
            "soc_name": c["soc_name"],
            "hlt_name": c["hlt_name"],
            "confidence": round(min(1.0, c["similarity"] + 0.05), 4),
            "candidates": candidates,
            "selection_method": "rag_high_confidence",
        }

    # Build numbered candidate list for the prompt
    candidate_lines = []
    for c in candidates:
        candidate_lines.append(
            f"{c['rank']}. PT: {c['pt_name']} | Code: {c['pt_code']} "
            f"| SOC: {c['soc_name']} | Similarity: {c['similarity']:.3f}"
        )
    candidates_text = "\n".join(candidate_lines)

    context_line = f"\nClinical context: {clinical_context}" if clinical_context else ""

    prompt = f"""Adverse event verbatim term: "{verbatim_term}"{context_line}

MedDRA candidates (ranked by semantic similarity):
{candidates_text}

Select the single best MedDRA Preferred Term for this adverse event.
Respond on ONE line using EXACTLY this format:
SELECTION: <number>
REASON: <one sentence>

Where <number> is the candidate number (1-{len(candidates)})."""

    try:
        response = generate(prompt=prompt, system_prompt=SELECTION_SYSTEM_PROMPT)
        selected_idx = _parse_selection(response, len(candidates))
        selected = candidates[selected_idx]

        # Confidence: blend similarity with position (rank 1 = full weight)
        position_bonus = max(0, (len(candidates) - selected_idx) / len(candidates)) * 0.1
        confidence = round(min(1.0, selected["similarity"] + position_bonus), 4)

        return {
            "verbatim_term": verbatim_term,
            "pt_name": selected["pt_name"],
            "pt_code": selected["pt_code"],
            "soc_name": selected["soc_name"],
            "hlt_name": selected["hlt_name"],
            "confidence": confidence,
            "candidates": candidates,
            "selection_method": "ollama",
        }

    except Exception:
        # Fallback: return highest-similarity candidate
        top = candidates[0]
        return {
            "verbatim_term": verbatim_term,
            "pt_name": top["pt_name"],
            "pt_code": top["pt_code"],
            "soc_name": top["soc_name"],
            "hlt_name": top["hlt_name"],
            "confidence": round(top["similarity"] * 0.9, 4),  # slight penalty for fallback
            "candidates": candidates,
            "selection_method": "top1_fallback",
        }


# --------------------------------------------------------------------------- #
# Convenience: single-call full coding                                         #
# --------------------------------------------------------------------------- #

def code_reaction(
    verbatim_term: str,
    clinical_context: str = "",
    customer_id: str | None = None,
) -> dict:
    """
    Full pipeline: verbatim text in, best MedDRA match out.
    Combines query_meddra + select_best_match in one call.

    If `customer_id` is supplied, the adaptive learning layer runs first:
      1. Authoritative custom-term lookup (skips RAG entirely)
      2. Per-customer ChromaDB augmentation (merged into global candidates)
    """
    # 1. Authoritative lookup — if the user has taught the system this term,
    #    trust it and skip ChromaDB + LLM entirely.
    if customer_id:
        try:
            from pipeline.adaptive import lookup_custom_mapping
            learned = lookup_custom_mapping(customer_id, verbatim_term)
            if learned:
                return {
                    "verbatim_term": verbatim_term,
                    "pt_name": learned.get("pt_name", ""),
                    "pt_code": learned.get("pt_code", ""),
                    "soc_name": learned.get("soc_name", ""),
                    "hlt_name": learned.get("hlt_name", ""),
                    "confidence": 0.95,
                    "candidates": [learned | {"similarity": 0.95, "rank": 1}],
                    "selection_method": "custom_learned",
                }
        except Exception:
            pass  # fall through to RAG — never break the pipeline

    # 2. Global RAG candidates
    global_candidates = query_meddra(verbatim_term)

    # 3. Merge in customer-specific candidates (if the augmented collection exists)
    merged_candidates = global_candidates
    if customer_id:
        try:
            from pipeline.adaptive import query_customer_collection
            cust_cands = query_customer_collection(customer_id, verbatim_term, n_results=3)
            if cust_cands:
                # Drop any duplicates (by pt_code) and merge; boost already applied
                seen_codes = {c.get("pt_code") for c in global_candidates}
                new_cust = [c for c in cust_cands if c.get("pt_code") not in seen_codes]
                merged_candidates = sorted(
                    global_candidates + new_cust,
                    key=lambda c: c.get("similarity", 0),
                    reverse=True,
                )
                # Re-rank
                for i, c in enumerate(merged_candidates, start=1):
                    c["rank"] = i
        except Exception:
            pass

    return select_best_match(verbatim_term, merged_candidates, clinical_context)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _parse_selection(response: str, n_candidates: int) -> int:
    """
    Parse 'SELECTION: N' from Ollama's response.
    Returns 0-based index. Falls back to 0 on parse failure.
    """
    match = re.search(r"SELECTION\s*:\s*(\d+)", response, re.IGNORECASE)
    if match:
        number = int(match.group(1))
        idx = number - 1  # convert to 0-based
        if 0 <= idx < n_candidates:
            return idx
    return 0  # default to top candidate


def _empty_result(verbatim_term: str) -> dict:
    return {
        "verbatim_term": verbatim_term,
        "pt_name": "Unknown",
        "pt_code": "",
        "soc_name": "",
        "hlt_name": "",
        "confidence": 0.0,
        "candidates": [],
        "selection_method": "no_candidates",
    }
