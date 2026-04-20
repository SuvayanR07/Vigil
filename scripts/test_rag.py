"""
RAG pipeline test harness.
Tests 20 verbatim terms against known MedDRA Preferred Terms.
Prints top-1 accuracy, top-3 accuracy, and average confidence.

Run after embed_meddra.py has populated ChromaDB.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.meddra_coder import query_meddra, select_best_match

# 20 test cases: (verbatim_term, expected_pt_name, expected_pt_code)
TEST_CASES = [
    # Core 10 from the brief
    ("felt dizzy",             "Dizziness",          "10013573"),
    ("threw up",               "Vomiting",           "10047700"),
    ("couldn't sleep",         "Insomnia",           "10022437"),
    ("skin turned red",        "Rash",               "10037844"),
    ("heart was racing",       "Tachycardia",        "10043071"),
    ("bad headache",           "Headache",           "10019211"),
    ("feeling very tired",     "Fatigue",            "10016256"),
    ("stomach pain",           "Abdominal pain",     "10000081"),
    ("swollen ankles",         "Oedema peripheral",  "10030124"),
    ("short of breath",        "Dyspnoea",           "10013968"),
    # 10 additional common adverse events
    ("joint ache",             "Arthralgia",         "10003239"),
    ("muscle pain",            "Myalgia",            "10028411"),
    ("passed out",             "Syncope",            "10042772"),
    ("pins and needles",       "Paraesthesia",       "10033775"),
    ("low blood sugar",        "Hypoglycaemia",      "10020993"),
    ("nausea",                 "Nausea",             "10028813"),
    ("itching",                "Pruritus",           "10037087"),
    ("hair falling out",       "Alopecia",           "10001781"),
    ("severe allergic reaction","Anaphylaxis",       "10002198"),
    ("blood in urine",         "Hematuria",          "10019109"),
]


def run_tests(use_ollama: bool = True) -> None:
    print("=" * 70)
    print("VIGIL RAG PIPELINE — TEST HARNESS")
    print("=" * 70)
    print(f"Total test cases : {len(TEST_CASES)}")
    print(f"Ollama selection : {'ENABLED' if use_ollama else 'DISABLED (top-1 only)'}")
    print()

    top1_hits = 0
    top3_hits = 0
    total_confidence = 0.0
    results = []

    for i, (verbatim, expected_pt, expected_code) in enumerate(TEST_CASES, start=1):
        t0 = time.time()

        # Step 1: retrieve candidates
        candidates = query_meddra(verbatim)

        # Step 2: select best match
        if use_ollama:
            result = select_best_match(verbatim, candidates)
        else:
            # Top-1 only (no Ollama) — useful if Ollama isn't running
            if candidates:
                c = candidates[0]
                result = {
                    "verbatim_term": verbatim,
                    "pt_name": c["pt_name"],
                    "pt_code": c["pt_code"],
                    "soc_name": c["soc_name"],
                    "confidence": c["similarity"],
                    "candidates": candidates,
                    "selection_method": "top1_only",
                }
            else:
                result = {
                    "verbatim_term": verbatim,
                    "pt_name": "Unknown",
                    "pt_code": "",
                    "soc_name": "",
                    "confidence": 0.0,
                    "candidates": [],
                    "selection_method": "no_candidates",
                }

        elapsed = time.time() - t0

        # Evaluate
        predicted_name = result["pt_name"]
        predicted_code = result["pt_code"]
        top3_names = [c["pt_name"] for c in candidates[:3]]

        is_top1 = (
            predicted_name.lower() == expected_pt.lower()
            or predicted_code == expected_code
        )
        is_top3 = (
            expected_pt.lower() in [n.lower() for n in top3_names]
            or expected_code in [c["pt_code"] for c in candidates[:3]]
        )

        if is_top1:
            top1_hits += 1
        if is_top3:
            top3_hits += 1
        total_confidence += result["confidence"]

        status = "✓" if is_top1 else ("~" if is_top3 else "✗")
        results.append({
            "status": status,
            "verbatim": verbatim,
            "expected": expected_pt,
            "predicted": predicted_name,
            "code": predicted_code,
            "confidence": result["confidence"],
            "method": result["selection_method"],
            "elapsed": elapsed,
        })

        print(
            f"[{i:02d}] {status}  '{verbatim}'\n"
            f"      Expected : {expected_pt} ({expected_code})\n"
            f"      Got      : {predicted_name} ({predicted_code})"
            f"  conf={result['confidence']:.3f}  [{result['selection_method']}]"
            f"  {elapsed:.1f}s"
        )
        if not is_top1 and candidates:
            top3_str = " | ".join(
                f"{c['pt_name']} ({c['similarity']:.3f})" for c in candidates[:3]
            )
            print(f"      Top-3    : {top3_str}")
        print()

    # Summary
    n = len(TEST_CASES)
    top1_pct = top1_hits / n * 100
    top3_pct = top3_hits / n * 100
    avg_conf = total_confidence / n

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Top-1 accuracy  : {top1_hits}/{n}  ({top1_pct:.1f}%)")
    print(f"  Top-3 accuracy  : {top3_hits}/{n}  ({top3_pct:.1f}%)")
    print(f"  Avg confidence  : {avg_conf:.3f}")
    print()

    # Pass/Fail verdict
    if top1_pct >= 75:
        print("  ✓ PASS — Top-1 accuracy meets 75% target. Ready for Phase 3.")
    else:
        print("  ✗ BELOW TARGET — Top-1 < 75%.")
        print("    Fixes to try:")
        print("    1. Re-run embed_meddra.py (ChromaDB may be stale)")
        print("    2. Add more synonyms to failing terms in curate_meddra.py")
        print("    3. Check Ollama is running: ollama serve")

    if top3_pct >= 90:
        print("  ✓ Top-3 accuracy ≥ 90% — RAG retrieval is working well.")
    else:
        print("  ✗ Top-3 accuracy < 90% — synonym coverage may be insufficient.")

    print("=" * 70)


if __name__ == "__main__":
    # Pass --no-ollama to skip LLM selection and test RAG retrieval only
    use_ollama = "--no-ollama" not in sys.argv
    if not use_ollama:
        print("(Running in --no-ollama mode: using top-1 similarity only)\n")
    run_tests(use_ollama=use_ollama)
