"""
Pipeline orchestrator: raw narrative -> ClassifiedReport.

Chains together:
  1. extractor.extract_report()    — LLM extraction of entities
  2. meddra_coder.code_reaction()  — RAG coding for each verbatim reaction
  3. severity.classify_severity()  — FDA seriousness classification
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIDENCE_THRESHOLD
from pipeline.extractor import extract_report
from pipeline.meddra_coder import code_reaction
from pipeline.schemas import (
    ClassifiedReport,
    CodedReport,
    MedDRAMatch,
)
from pipeline.severity import classify_severity


def classify_report(
    narrative: str,
    customer_id: str | None = None,
) -> ClassifiedReport:
    """
    Full pipeline: raw narrative in, fully classified report out.

    Args:
        narrative:   Free-text adverse event narrative.
        customer_id: Optional customer UUID. When supplied, the adaptive
                     learning layer applies (custom-term lookups + per-customer
                     ChromaDB augmentation), the report is persisted to the
                     customer's history, and the running report count is bumped.

    Returns:
        ClassifiedReport with patient, drugs, coded reactions,
        severity assessment, and review flags.
    """
    # Step 1: extraction
    extracted = extract_report(narrative)

    # Step 2: MedDRA coding for each reaction
    clinical_context = narrative[:500] if narrative else ""
    coded_reactions: list[MedDRAMatch] = []
    for verbatim in extracted.reactions_verbatim:
        match_dict = code_reaction(
            verbatim,
            clinical_context=clinical_context,
            customer_id=customer_id,
        )
        coded_reactions.append(
            MedDRAMatch(
                verbatim_term=match_dict["verbatim_term"],
                pt_code=match_dict.get("pt_code", ""),
                pt_name=match_dict.get("pt_name", "Unknown"),
                soc_name=match_dict.get("soc_name", ""),
                confidence=match_dict.get("confidence", 0.0),
                candidates=match_dict.get("candidates", []),
            )
        )

    # Step 3: build CodedReport
    coded = CodedReport(
        **extracted.model_dump(),
        coded_reactions=coded_reactions,
    )

    # Step 4: severity classification (returns ClassifiedReport with flags)
    classified = classify_severity(coded)

    # Step 5: persist to customer history (never block/break on persistence errors)
    if customer_id:
        try:
            from pipeline.customer import increment_reports
            from pipeline.history import get_report_count, save_report
            from pipeline.adaptive import maybe_augment_after_report

            report_id = save_report(
                customer_id, narrative, classified.model_dump()
            )
            increment_reports(customer_id)
            # Attach the persistence handle so the UI can wire feedback back
            classified.flags_for_review = list(classified.flags_for_review)  # copy
            # Use a non-intrusive channel: Pydantic won't reject model attrs
            # but adding to model_extra keeps the schema clean. Easier: return
            # a separate attribute via session state on the UI side.
            classified.__dict__["_report_id"] = report_id  # attached (not serialized)

            total = get_report_count(customer_id)
            maybe_augment_after_report(customer_id, total)
        except Exception:
            pass

    return classified


# --------------------------------------------------------------------------- #
# CLI smoke test                                                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import json

    TEST_NARRATIVE = (
        "A 72-year-old female patient was prescribed Metformin 500mg twice "
        "daily for Type 2 Diabetes. She was also taking Lisinopril 10mg and "
        "Aspirin 81mg. Three days after starting Metformin, she experienced "
        "severe dizziness, nausea, and vomiting. She was hospitalized for "
        "2 days. Metformin was discontinued and symptoms resolved within "
        "48 hours."
    )

    print("=" * 70)
    print("CLASSIFY PIPELINE SMOKE TEST")
    print("=" * 70)
    print(f"\nNarrative ({len(TEST_NARRATIVE.split())} words):")
    print(TEST_NARRATIVE)
    print("\nRunning full pipeline...\n")

    result = classify_report(TEST_NARRATIVE)
    print(json.dumps(result.model_dump(), indent=2))
