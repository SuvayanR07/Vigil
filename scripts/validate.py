"""
Formal accuracy validation against FAERS ground truth.

For each FAERS report:
  - Feed narrative to classify_report()
  - Compare predicted MedDRA PTs against ground_truth_reactions
  - Compare predicted is_serious against FAERS serious flag

Metrics:
  - Top-1 PT precision : of the PTs we predicted, what % matched ground truth
  - PT recall          : of ground truth PTs, what % did we recover
  - SOC accuracy       : of matched PTs, what % had correct System Organ Class
  - Severity accuracy  : of reports, what % had correct is_serious
  - Extraction rate    : % of reports where we got >= 1 reaction out
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAERS_SAMPLES_PATH, MEDDRA_TERMS_PATH
from pipeline.classify import classify_report


VALIDATION_SAMPLE_SIZE = 50
RESULTS_PATH = Path(__file__).parent.parent / "data" / "validation_results.json"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _load_pt_to_soc() -> dict[str, str]:
    """Build a lookup: lowercased pt_name -> soc_name from our MedDRA dictionary."""
    with open(MEDDRA_TERMS_PATH) as f:
        terms = json.load(f)
    return {t["pt_name"].lower(): t["soc_name"] for t in terms}


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _pt_match(predicted: str, ground_truth: str) -> bool:
    """
    Case-insensitive PT match. Accepts exact match or strong substring overlap
    (e.g. 'Nausea' vs 'Nausea and vomiting').
    """
    p, g = _normalize(predicted), _normalize(ground_truth)
    if not p or not g:
        return False
    if p == g:
        return True
    # Containment match — useful since FAERS sometimes uses compound PTs
    if p in g or g in p:
        # Avoid trivial false positives from tiny strings
        return min(len(p), len(g)) >= 4
    return False


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    with open(FAERS_SAMPLES_PATH) as f:
        all_reports = json.load(f)

    # Filter to reports with narrative + at least 1 ground truth PT
    usable = [
        r for r in all_reports
        if r.get("narrative") and r.get("ground_truth_reactions")
    ]
    sample = usable[:VALIDATION_SAMPLE_SIZE]
    pt_to_soc = _load_pt_to_soc()

    print("=" * 75)
    print("VIGIL VALIDATION: FAERS GROUND TRUTH")
    print("=" * 75)
    print(f"Running pipeline on {len(sample)} FAERS reports...\n")

    # Counters
    n_reports = len(sample)
    n_with_any_extraction = 0
    n_severity_correct = 0

    total_predicted_pts = 0
    total_gt_pts = 0
    pt_precision_hits = 0    # predictions that matched a GT PT
    pt_recall_hits = 0       # GT PTs that were captured by a prediction
    soc_checked = 0
    soc_correct = 0

    # FAERS seriousness distribution (from the 300-report fetch, all are '1')
    faers_serious_count = sum(1 for r in sample if r.get("serious") == "1")

    per_report_results = []
    latencies: list[float] = []

    for idx, report in enumerate(sample, start=1):
        narrative = report["narrative"]
        gt_pts = [gt["verbatim"] for gt in report["ground_truth_reactions"]]
        gt_serious = report.get("serious") == "1"

        t0 = time.time()
        try:
            classified = classify_report(narrative)
            elapsed = time.time() - t0
            latencies.append(elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            latencies.append(elapsed)
            print(f"[{idx:2d}/{n_reports}] ERROR: {type(e).__name__}: {e}")
            per_report_results.append({
                "report_id": report.get("report_id"),
                "error": f"{type(e).__name__}: {e}",
            })
            continue

        predicted_pts = [m.pt_name for m in classified.coded_reactions]
        if predicted_pts:
            n_with_any_extraction += 1

        # Severity check
        if classified.is_serious == gt_serious:
            n_severity_correct += 1

        # PT precision: each prediction that matches any GT PT
        report_precision_hits = 0
        for p in predicted_pts:
            if any(_pt_match(p, g) for g in gt_pts):
                report_precision_hits += 1
        pt_precision_hits += report_precision_hits
        total_predicted_pts += len(predicted_pts)

        # PT recall: each GT PT that was captured by any prediction
        report_recall_hits = 0
        for g in gt_pts:
            if any(_pt_match(p, g) for p in predicted_pts):
                report_recall_hits += 1
        pt_recall_hits += report_recall_hits
        total_gt_pts += len(gt_pts)

        # SOC accuracy: for each prediction that matched a GT PT, check SOC
        for match in classified.coded_reactions:
            for g in gt_pts:
                if _pt_match(match.pt_name, g):
                    gt_soc = pt_to_soc.get(g.lower())
                    if gt_soc:
                        soc_checked += 1
                        if _normalize(match.soc_name) == _normalize(gt_soc):
                            soc_correct += 1
                    break

        status = "OK" if predicted_pts else "NO_REACTIONS"
        print(
            f"[{idx:2d}/{n_reports}] {status:12s} "
            f"gt={len(gt_pts)} pred={len(predicted_pts)} "
            f"match={report_recall_hits}/{len(gt_pts)} "
            f"serious_pred={classified.is_serious} "
            f"({elapsed:.1f}s)"
        )

        per_report_results.append({
            "report_id": report.get("report_id"),
            "narrative_preview": narrative[:120],
            "ground_truth_pts": gt_pts,
            "predicted_pts": predicted_pts,
            "pt_precision_hits": report_precision_hits,
            "pt_recall_hits": report_recall_hits,
            "predicted_serious": classified.is_serious,
            "gt_serious": gt_serious,
            "latency": round(elapsed, 2),
        })

    # --- Aggregate metrics ---
    pt_precision = (pt_precision_hits / total_predicted_pts) if total_predicted_pts else 0
    pt_recall = (pt_recall_hits / total_gt_pts) if total_gt_pts else 0
    f1 = (
        2 * pt_precision * pt_recall / (pt_precision + pt_recall)
        if (pt_precision + pt_recall)
        else 0
    )
    soc_accuracy = (soc_correct / soc_checked) if soc_checked else 0
    severity_accuracy = n_severity_correct / n_reports if n_reports else 0
    extraction_rate = n_with_any_extraction / n_reports if n_reports else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    metrics = {
        "n_reports": n_reports,
        "extraction_rate": round(extraction_rate, 4),
        "total_predicted_pts": total_predicted_pts,
        "total_ground_truth_pts": total_gt_pts,
        "pt_top1_precision": round(pt_precision, 4),
        "pt_recall": round(pt_recall, 4),
        "pt_f1": round(f1, 4),
        "soc_accuracy": round(soc_accuracy, 4),
        "soc_checked": soc_checked,
        "severity_accuracy": round(severity_accuracy, 4),
        "faers_serious_fraction": round(faers_serious_count / n_reports, 4),
        "avg_latency_seconds": round(avg_latency, 2),
        "max_latency_seconds": round(max(latencies), 2) if latencies else 0,
    }

    # --- Print summary ---
    print("\n" + "=" * 75)
    print("VALIDATION RESULTS")
    print("=" * 75)
    print(f"  Reports tested:           {n_reports}")
    print(f"  Extraction rate:          {extraction_rate:.1%}  ({n_with_any_extraction}/{n_reports} reports yielded >=1 reaction)")
    print()
    print(f"  MedDRA PT precision:      {pt_precision:.1%}  ({pt_precision_hits}/{total_predicted_pts})")
    print(f"  MedDRA PT recall:         {pt_recall:.1%}  ({pt_recall_hits}/{total_gt_pts})")
    print(f"  MedDRA PT F1:             {f1:.3f}")
    print()
    print(f"  SOC accuracy (on match):  {soc_accuracy:.1%}  ({soc_correct}/{soc_checked})")
    print()
    print(f"  Severity accuracy:        {severity_accuracy:.1%}  ({n_severity_correct}/{n_reports})")
    print(f"  (FAERS sample is {faers_serious_count}/{n_reports} serious — this measures serious-event recall)")
    print()
    print(f"  Avg latency:              {avg_latency:.1f}s  (max {max(latencies):.1f}s)")
    print("=" * 75)

    # --- Save ---
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "metrics": metrics,
            "per_report": per_report_results,
        }, f, indent=2)
    print(f"\nFull results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
