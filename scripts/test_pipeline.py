"""
End-to-end pipeline test: 10 narratives covering a range of scenarios.

Runs each through classify_report() and reports:
  - pass/fail per narrative (basic sanity checks)
  - average latency
  - aggregate stats (serious count, total reactions, avg confidence)
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.classify import classify_report
from pipeline.schemas import ClassifiedReport


# --------------------------------------------------------------------------- #
# Test narratives                                                              #
# --------------------------------------------------------------------------- #

TEST_NARRATIVES = [
    {
        "id": 1,
        "name": "Metformin hospitalization (baseline)",
        "narrative": (
            "A 72-year-old female patient was prescribed Metformin 500mg twice "
            "daily for Type 2 Diabetes. She was also taking Lisinopril 10mg and "
            "Aspirin 81mg. Three days after starting Metformin, she experienced "
            "severe dizziness, nausea, and vomiting. She was hospitalized for "
            "2 days. Metformin was discontinued and symptoms resolved within "
            "48 hours."
        ),
        "expect_serious": True,
        "expect_min_reactions": 2,
    },
    {
        "id": 2,
        "name": "FAERS-style: fatal event",
        "narrative": (
            "65-year-old male with history of hypertension was started on "
            "Warfarin 5mg daily for atrial fibrillation. Four weeks later "
            "he presented with severe headache and was found to have a "
            "massive intracranial hemorrhage. The patient died within 24 hours "
            "of admission."
        ),
        "expect_serious": True,
        "expect_min_reactions": 1,
    },
    {
        "id": 3,
        "name": "FAERS-style: mild rash, non-serious",
        "narrative": (
            "Patient is a 34-year-old female who started Amoxicillin 500mg "
            "three times daily for sinusitis. On day 4 she noticed a mild "
            "itchy rash on her arms. Amoxicillin was stopped and the rash "
            "resolved over the next two days. No other treatment required."
        ),
        "expect_serious": False,
        "expect_min_reactions": 1,
    },
    {
        "id": 4,
        "name": "FAERS-style: chemo nausea",
        "narrative": (
            "A 58-year-old woman undergoing chemotherapy with Cisplatin "
            "experienced severe nausea and vomiting within hours of her "
            "second cycle. She was also receiving Ondansetron. She recovered "
            "after supportive care but required IV fluids in the emergency room."
        ),
        "expect_serious": True,
        "expect_min_reactions": 2,
    },
    {
        "id": 5,
        "name": "FAERS-style: anaphylaxis life-threatening",
        "narrative": (
            "A 42-year-old male received his first dose of intravenous "
            "Ceftriaxone 1g for pneumonia. Within 10 minutes he developed "
            "hives, difficulty breathing, and hypotension. This was "
            "life-threatening anaphylaxis. He was resuscitated with "
            "epinephrine and admitted to the ICU for 24 hours."
        ),
        "expect_serious": True,
        "expect_min_reactions": 2,
    },
    {
        "id": 6,
        "name": "Synthetic: multiple drugs, no serious event",
        "narrative": (
            "A 50-year-old man on Atorvastatin 20mg, Metoprolol 50mg twice "
            "daily, and Aspirin 81mg reports mild muscle aches and occasional "
            "headache since starting Atorvastatin two months ago. Symptoms "
            "are tolerable and have not affected his daily activities."
        ),
        "expect_serious": False,
        "expect_min_reactions": 1,
    },
    {
        "id": 7,
        "name": "Synthetic: messy patient language",
        "narrative": (
            "i took this new blood pressure med lisinopril like 2 weeks ago "
            "and ever since i been coughing nonstop and my throat feels "
            "scratchy. also feel kinda dizzy when i stand up fast. doc said "
            "switch meds."
        ),
        "expect_serious": False,
        "expect_min_reactions": 1,
    },
    {
        "id": 8,
        "name": "Synthetic: very short report",
        "narrative": "Patient took Ibuprofen and developed a skin rash.",
        "expect_serious": False,
        "expect_min_reactions": 1,
    },
    {
        "id": 9,
        "name": "Synthetic: negated hospitalization",
        "narrative": (
            "A 28-year-old female started Sertraline 50mg for depression. "
            "She experienced mild insomnia and nausea during the first week. "
            "She was NOT hospitalized and did not require any intervention. "
            "Symptoms resolved after dose reduction."
        ),
        "expect_serious": False,
        "expect_min_reactions": 1,
    },
    {
        "id": 10,
        "name": "Synthetic: surgery required",
        "narrative": (
            "A 60-year-old male on long-term Prednisone 10mg daily for "
            "rheumatoid arthritis developed a perforated gastric ulcer "
            "requiring emergency surgery. He recovered post-operatively "
            "after a 5-day hospital stay."
        ),
        "expect_serious": True,
        "expect_min_reactions": 1,
    },
]


# --------------------------------------------------------------------------- #
# Test runner                                                                  #
# --------------------------------------------------------------------------- #

def _check(report: ClassifiedReport, expected: dict) -> tuple[bool, list[str]]:
    """Return (passed, list of failure reasons)."""
    failures: list[str] = []

    if report.is_serious != expected["expect_serious"]:
        failures.append(
            f"is_serious={report.is_serious}, expected {expected['expect_serious']}"
        )

    n_reactions = len(report.coded_reactions)
    if n_reactions < expected["expect_min_reactions"]:
        failures.append(
            f"only {n_reactions} coded reactions, expected >= {expected['expect_min_reactions']}"
        )

    # Sanity: must have at least one drug extracted
    if not report.suspect_drugs and not report.concomitant_drugs:
        failures.append("no drugs extracted at all")

    return (len(failures) == 0, failures)


def _summarize_report(report: ClassifiedReport) -> str:
    """One-line summary of a ClassifiedReport for the test log."""
    drugs = [d.name for d in report.suspect_drugs] + [d.name for d in report.concomitant_drugs]
    reactions = [f"{m.verbatim_term}->{m.pt_name}({m.confidence:.2f})" for m in report.coded_reactions]
    serious_tag = "SERIOUS" if report.is_serious else "non-serious"
    criteria_hit = [k for k, v in report.seriousness_criteria.items() if v]
    criteria_str = f" [{','.join(criteria_hit)}]" if criteria_hit else ""
    return (
        f"    drugs:     {', '.join(drugs) or '(none)'}\n"
        f"    reactions: {'; '.join(reactions) or '(none)'}\n"
        f"    severity:  {serious_tag}{criteria_str} (conf={report.severity_confidence})\n"
        f"    flags:     {len(report.flags_for_review)}"
    )


def main():
    print("=" * 75)
    print("VIGIL FULL PIPELINE TEST")
    print("=" * 75)
    print(f"Running {len(TEST_NARRATIVES)} narratives through classify_report()...\n")

    results = []
    latencies: list[float] = []
    passed = 0

    for case in TEST_NARRATIVES:
        print(f"[{case['id']:2d}/10] {case['name']}")

        t0 = time.time()
        try:
            report = classify_report(case["narrative"])
            elapsed = time.time() - t0
            latencies.append(elapsed)

            ok, failures = _check(report, case)
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1

            print(f"    status:    {status}  ({elapsed:.1f}s)")
            print(_summarize_report(report))
            for f in failures:
                print(f"    !! {f}")
            print()

            results.append({
                "id": case["id"],
                "name": case["name"],
                "status": status,
                "latency": round(elapsed, 2),
                "failures": failures,
                "report": report.model_dump(),
            })

        except Exception as e:
            elapsed = time.time() - t0
            latencies.append(elapsed)
            print(f"    status:    ERROR  ({elapsed:.1f}s)")
            print(f"    !! Exception: {type(e).__name__}: {e}\n")
            results.append({
                "id": case["id"],
                "name": case["name"],
                "status": "ERROR",
                "latency": round(elapsed, 2),
                "error": f"{type(e).__name__}: {e}",
            })

    # --- Aggregate stats ---
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    total_reactions = sum(
        len(r.get("report", {}).get("coded_reactions", []))
        for r in results
        if r["status"] != "ERROR"
    )
    serious_count = sum(
        1 for r in results
        if r["status"] != "ERROR" and r.get("report", {}).get("is_serious")
    )
    all_confidences = [
        m["confidence"]
        for r in results if r["status"] != "ERROR"
        for m in r.get("report", {}).get("coded_reactions", [])
    ]
    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"  Passed:              {passed}/{len(TEST_NARRATIVES)}")
    print(f"  Avg latency:         {avg_latency:.1f}s  (max {max_latency:.1f}s)")
    print(f"  Total reactions:     {total_reactions}")
    print(f"  Serious events:      {serious_count}")
    print(f"  Avg MedDRA conf:     {avg_conf:.3f}")
    print("=" * 75)

    # Save results for later inspection
    out_path = Path(__file__).parent.parent / "data" / "pipeline_test_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "passed": passed,
            "total": len(TEST_NARRATIVES),
            "avg_latency": round(avg_latency, 2),
            "max_latency": round(max_latency, 2),
            "avg_confidence": round(avg_conf, 3),
            "results": results,
        }, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
