"""
Learning-effectiveness analytics per customer.

Key metric: correction_rate over time. If adaptive learning is working,
the rate of manual corrections per report should trend downward as the
system accumulates custom mappings.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.adaptive import get_custom_terms
from pipeline.feedback import get_feedback_history
from pipeline.history import get_report_history


def get_learning_metrics(customer_id: str) -> dict:
    """Return a dict of learning metrics for dashboard display."""
    reports = get_report_history(customer_id)
    feedback = get_feedback_history(customer_id)
    custom = get_custom_terms(customer_id)

    total_reports = len(reports)
    total_corrections = sum(len(f.get("corrections", [])) for f in feedback)
    reports_with_feedback = len(feedback)

    correction_rate = (
        reports_with_feedback / total_reports if total_reports else 0.0
    )

    # Top corrected verbatim terms
    term_counter: Counter[str] = Counter()
    for fb in feedback:
        for c in fb.get("corrections", []):
            if c.get("field_type") == "meddra":
                term = c.get("verbatim_term", "").strip().lower()
                if term:
                    term_counter[term] += 1
    top_corrected = [
        {"term": t, "count": n} for t, n in term_counter.most_common(10)
    ]

    # Improvement trend: compare first half vs second half of reports
    improvement = _estimate_improvement(reports, feedback)

    return {
        "total_reports_processed": total_reports,
        "total_corrections_made": total_corrections,
        "reports_with_any_correction": reports_with_feedback,
        "correction_rate": round(correction_rate, 4),
        "top_corrected_terms": top_corrected,
        "custom_terms_count": len(custom),
        "authoritative_terms_count": sum(
            1 for v in custom.values() if v.get("frequency", 0) >= 2
        ),
        "estimated_accuracy_improvement": improvement,
        "trend_points": _trend_points(reports, feedback),
    }


def _estimate_improvement(reports: list[dict], feedback: list[dict]) -> float:
    """
    Compare correction rate in the first half of reports vs the second half.
    Positive value = correction rate dropped = system is learning.
    Returns a value in roughly [-1.0, 1.0].
    """
    n = len(reports)
    if n < 4:
        return 0.0

    # Reports come from history newest-first; reverse for chronological order
    chrono_ids = [r["report_id"] for r in reversed(reports)]
    half = n // 2
    first_ids = set(chrono_ids[:half])
    second_ids = set(chrono_ids[half:])

    fb_ids = {f["report_id"] for f in feedback}
    first_rate = len(first_ids & fb_ids) / max(1, len(first_ids))
    second_rate = len(second_ids & fb_ids) / max(1, len(second_ids))
    return round(first_rate - second_rate, 4)


def _trend_points(reports: list[dict], feedback: list[dict]) -> list[dict]:
    """
    Rolling correction rate: for each report, compute the cumulative
    correction rate up to and including that report.

    Useful for a line chart in the Analytics tab.
    """
    if not reports:
        return []
    chrono = list(reversed(reports))  # oldest first
    fb_ids = {f["report_id"] for f in feedback}

    cumulative_corr = 0
    trend: list[dict] = []
    for i, r in enumerate(chrono, start=1):
        if r["report_id"] in fb_ids:
            cumulative_corr += 1
        trend.append({
            "report_index": i,
            "created_at": r.get("created_at"),
            "correction_rate": round(cumulative_corr / i, 4),
        })
    return trend
