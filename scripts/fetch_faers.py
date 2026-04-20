"""
Downloads 300 adverse event reports from OpenFDA FAERS API.
Saves raw report data to data/faers_samples.json.
"""

import json
import time
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAERS_API_URL, FAERS_SAMPLE_SIZE, FAERS_SAMPLES_PATH, DATA_DIR


def fetch_batch(skip: int, limit: int = 100) -> list[dict]:
    params = {
        "limit": limit,
        "skip": skip,
        "search": 'serious:1 AND _exists_:patient.reaction AND _exists_:patient.drug',
    }
    resp = requests.get(FAERS_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


def extract_narrative(report: dict) -> str:
    """Pull the most useful free-text narrative from a FAERS report."""
    patient = report.get("patient", {})

    reactions = patient.get("reaction", [])
    reaction_text = ", ".join(
        r.get("reactionmeddrapt", "") for r in reactions if r.get("reactionmeddrapt")
    )

    drugs = patient.get("drug", [])
    drug_texts = []
    for d in drugs:
        name = d.get("medicinalproduct", "unknown drug")
        role = d.get("drugcharacterization", "")
        dose = d.get("drugdosagetext", "")
        indication = d.get("drugindication", "")
        drug_texts.append(f"{name} (role={role}, dose={dose}, indication={indication})")

    age = patient.get("patientonsetage", "unknown")
    age_unit = patient.get("patientonsetageunit", "")
    sex_map = {"1": "male", "2": "female"}
    sex = sex_map.get(str(patient.get("patientsex", "")), "unknown")

    outcome_map = {
        "1": "recovered",
        "2": "recovering",
        "3": "not recovered",
        "4": "recovered with sequelae",
        "5": "fatal",
        "6": "unknown",
    }
    outcomes = [
        outcome_map.get(str(r.get("reactionoutcome", "")), "unknown")
        for r in reactions
    ]
    outcome_text = ", ".join(set(outcomes))

    # Build seriousness sentence from structured FAERS flags so that severity
    # keywords actually appear in the narrative (otherwise rules-based severity
    # has nothing to match on).
    seriousness_phrases = []
    if str(report.get("seriousnessdeath", "")) == "1":
        seriousness_phrases.append("The patient died.")
    if str(report.get("seriousnesslifethreatening", "")) == "1":
        seriousness_phrases.append("The event was life-threatening.")
    if str(report.get("seriousnesshospitalization", "")) == "1":
        seriousness_phrases.append("The patient was hospitalized.")
    if str(report.get("seriousnessdisabling", "")) == "1":
        seriousness_phrases.append("The event resulted in disability.")
    if str(report.get("seriousnesscongenitalanomali", "")) == "1":
        seriousness_phrases.append("A congenital anomaly was reported.")
    if str(report.get("seriousnessother", "")) == "1":
        seriousness_phrases.append("The event was medically significant and required intervention.")
    seriousness_text = " " + " ".join(seriousness_phrases) if seriousness_phrases else ""

    narrative = (
        f"Patient: {age} {age_unit} {sex}. "
        f"Drugs: {'; '.join(drug_texts)}. "
        f"Reactions: {reaction_text}. "
        f"Outcome: {outcome_text}.{seriousness_text}"
    )
    return narrative.strip()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    batch_size = 100
    batches_needed = (FAERS_SAMPLE_SIZE + batch_size - 1) // batch_size

    print(f"Fetching {FAERS_SAMPLE_SIZE} FAERS reports in {batches_needed} batches...")

    for i in range(batches_needed):
        skip = i * batch_size
        limit = min(batch_size, FAERS_SAMPLE_SIZE - skip)
        print(f"  Batch {i + 1}/{batches_needed} (skip={skip}, limit={limit})...", end=" ")
        try:
            batch = fetch_batch(skip=skip, limit=limit)
            for report in batch:
                reports.append({
                    "report_id": report.get("safetyreportid", f"UNKNOWN_{len(reports)}"),
                    "serious": report.get("serious", ""),
                    "seriousness_criteria": {
                        "death": report.get("seriousnessdeath", ""),
                        "life_threatening": report.get("seriousnesslifethreatening", ""),
                        "hospitalization": report.get("seriousnesshospitalization", ""),
                        "disability": report.get("seriousnessdisabling", ""),
                        "congenital_anomaly": report.get("seriousnesscongenitalanomali", ""),
                        "other": report.get("seriousnessother", ""),
                    },
                    "narrative": extract_narrative(report),
                    "ground_truth_reactions": [
                        {
                            "verbatim": r.get("reactionmeddrapt", ""),
                            "outcome": r.get("reactionoutcome", ""),
                        }
                        for r in report.get("patient", {}).get("reaction", [])
                    ],
                    "raw": report,
                })
            print(f"got {len(batch)} reports.")
        except requests.HTTPError as e:
            print(f"HTTP error: {e}. Skipping batch.")
        except Exception as e:
            print(f"Error: {e}. Skipping batch.")

        if i < batches_needed - 1:
            time.sleep(0.5)

    with open(FAERS_SAMPLES_PATH, "w") as f:
        json.dump(reports, f, indent=2)

    print(f"\nSaved {len(reports)} reports to {FAERS_SAMPLES_PATH}")


if __name__ == "__main__":
    main()
