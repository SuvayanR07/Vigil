"""
Extraction engine: raw narrative -> ExtractedReport.

Key design: Gemma 2B cannot reliably produce valid JSON. Instead we ask for
a DELIMITED natural-language format (one field per line, pipe-separated
sub-fields) and parse it ourselves with regex.

This is much more robust than JSON parsing on a 2B-parameter model.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAX_NARRATIVE_WORDS
from pipeline.ollama_client import generate
from pipeline.schemas import (
    PatientInfo,
    DrugInfo,
    ExtractedReport,
)


# --------------------------------------------------------------------------- #
# Prompt                                                                       #
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a pharmacovigilance extraction assistant. Extract structured data from adverse drug event narratives.

OUTPUT RULES:
- Output ONE field per line using the exact tags: PATIENT, SUSPECT_DRUG, CONCOMITANT_DRUG, REACTION, ONSET, DECHALLENGE, OUTCOME, REPORTER.
- Separate sub-fields with " | " (space-pipe-space).
- Write "unknown" for fields not mentioned in the narrative.
- Use ONE REACTION: tag per reaction. Never put multiple reactions on one line.
- If the narrative lists symptoms with "and" or commas (e.g. "nausea and vomiting", "headache, fever, rash"), emit EACH symptom as its own separate REACTION: line. Do not collapse or skip any.
- Repeat SUSPECT_DRUG: and CONCOMITANT_DRUG: tags for each drug.
- Output ONLY the tagged lines. No JSON. No prose. No commentary.
- Extract only what is stated in THIS narrative. Do not invent or copy from examples.

PATIENT format: age | sex | weight         (e.g. "72 years | female | unknown")
DRUG format:    name | dose+freq | route | indication
                (route = oral/IV/IM/topical/inhaled/unknown. Do NOT put frequency here.)
REACTION format: one short verbatim phrase
OUTCOME format: include recovery status, hospitalization, and death if mentioned.

EXAMPLE (format only — do not copy these values):
PATIENT: 45 years | male | unknown
SUSPECT_DRUG: DrugX | 20mg daily | oral | hypertension
CONCOMITANT_DRUG: DrugY | 5mg daily | unknown | unknown
REACTION: headache
REACTION: rash
REACTION: fatigue
ONSET: 2 weeks after starting DrugX
DECHALLENGE: symptoms resolved after stopping DrugX
OUTCOME: recovered; no hospitalization
REPORTER: physician
"""

USER_TEMPLATE = """Extract structured fields from this adverse event narrative.
Follow the EXACT format specified. Output only the tagged lines.

NARRATIVE:
{narrative}

OUTPUT:"""


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def extract_report(narrative: str) -> ExtractedReport:
    """
    Extract structured adverse event fields from a free-text narrative.

    Args:
        narrative: Raw adverse event report text.

    Returns:
        ExtractedReport with parsed fields. Missing fields are None or [].
    """
    if not narrative or not narrative.strip():
        return ExtractedReport(narrative="", narrative_truncated=False)

    # Truncate to MAX_NARRATIVE_WORDS (Gemma 2B has an 8K context window)
    words = narrative.strip().split()
    truncated = len(words) > MAX_NARRATIVE_WORDS
    narrative_for_llm = " ".join(words[:MAX_NARRATIVE_WORDS]) if truncated else narrative.strip()

    # Call Ollama
    user_prompt = USER_TEMPLATE.format(narrative=narrative_for_llm)
    raw_output = generate(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)

    # Parse delimited output into structured fields
    parsed = _parse_delimited(raw_output)

    return ExtractedReport(
        patient=parsed["patient"],
        suspect_drugs=parsed["suspect_drugs"],
        concomitant_drugs=parsed["concomitant_drugs"],
        reactions_verbatim=parsed["reactions_verbatim"],
        onset_timeline=parsed["onset_timeline"],
        dechallenge=parsed["dechallenge"],
        outcome=parsed["outcome"],
        reporter_type=parsed["reporter_type"],
        narrative=narrative,
        narrative_truncated=truncated,
    )


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #

_UNKNOWN_TOKENS = {"", "unknown", "none", "n/a", "na", "not specified", "not stated"}


def _clean(value: str) -> str | None:
    """Return None for unknown-ish values, else the cleaned string."""
    v = value.strip().strip("[]").strip()
    if v.lower() in _UNKNOWN_TOKENS:
        return None
    return v


def _split_pipes(value: str) -> list[str]:
    """Split on '|' and strip each part."""
    return [p.strip() for p in value.split("|")]


def _parse_delimited(raw: str) -> dict:
    """
    Parse Gemma's delimited output into a dict of fields.

    Robust to:
      - Extra whitespace, blank lines
      - Markdown formatting (```, **, etc.)
      - Mixed case tags
      - Partial / missing fields
    """
    result = {
        "patient": PatientInfo(),
        "suspect_drugs": [],
        "concomitant_drugs": [],
        "reactions_verbatim": [],
        "onset_timeline": None,
        "dechallenge": None,
        "outcome": None,
        "reporter_type": None,
    }

    # Strip code fences / markdown
    cleaned = re.sub(r"```[a-z]*\n?", "", raw)
    cleaned = cleaned.replace("```", "").replace("**", "")

    for line in cleaned.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue

        tag, _, value = line.partition(":")
        tag = tag.strip().upper()
        value = value.strip()

        # Normalize common variants
        if tag in ("PATIENT", "PATIENT INFO", "DEMOGRAPHICS"):
            result["patient"] = _parse_patient(value)

        elif tag in ("SUSPECT_DRUG", "SUSPECT DRUG", "SUSPECT"):
            drug = _parse_drug(value, include_route_indication=True)
            if drug:
                result["suspect_drugs"].append(drug)

        elif tag in ("CONCOMITANT_DRUG", "CONCOMITANT DRUG", "CONCOMITANT"):
            drug = _parse_drug(value, include_route_indication=False)
            if drug:
                result["concomitant_drugs"].append(drug)

        elif tag in ("REACTION", "ADVERSE EVENT", "AE"):
            v = _clean(value)
            if v:
                # Split comma-, pipe-, semicolon-, or "and"-joined reactions
                for part in re.split(r"\s*\|\s*|,|;| and ", v):
                    p = part.strip()
                    if p and p.lower() not in _UNKNOWN_TOKENS:
                        result["reactions_verbatim"].append(p)

        elif tag in ("ONSET", "ONSET_TIMELINE", "TIMELINE"):
            result["onset_timeline"] = _clean(value)

        elif tag == "DECHALLENGE":
            result["dechallenge"] = _clean(value)

        elif tag in ("OUTCOME", "RESULT"):
            result["outcome"] = _clean(value)

        elif tag in ("REPORTER", "REPORTER_TYPE"):
            result["reporter_type"] = _clean(value)

    return result


def _parse_patient(value: str) -> PatientInfo:
    """Parse 'age | sex | weight' into PatientInfo. Robust to field bleeding."""
    parts = _split_pipes(value)
    age = _clean(parts[0]) if len(parts) > 0 else None
    sex = _clean(parts[1]) if len(parts) > 1 else None
    weight = _clean(parts[2]) if len(parts) > 2 else None

    # Field-bleed recovery: extract sex from age if Gemma mashed them together
    if age:
        age_lower = age.lower()
        if not sex:
            if "female" in age_lower or "woman" in age_lower:
                sex = "female"
            elif re.search(r"\bmale\b", age_lower) or "man" in age_lower:
                sex = "male"
        # Clean age: strip sex words, keep only "NN years" or "NN-year-old"
        age_match = re.search(r"(\d+)\s*[- ]?\s*(?:year|yr|y)[- ]?old?|(\d+)\s*(?:years?|yrs?|y)", age, re.I)
        if age_match:
            age_num = age_match.group(1) or age_match.group(2)
            age = f"{age_num} years"

    # Normalize sex
    if sex:
        sex_lower = sex.lower()
        if "female" in sex_lower or sex_lower in ("f", "woman"):
            sex = "female"
        elif re.search(r"\bmale\b", sex_lower) or sex_lower in ("m", "man"):
            sex = "male"

    return PatientInfo(age=age, sex=sex, weight=weight)


_VALID_ROUTES = {
    "oral", "po", "iv", "intravenous", "im", "intramuscular", "sc",
    "subcutaneous", "topical", "inhaled", "inhalation", "nasal",
    "rectal", "sublingual", "transdermal",
}
_FREQUENCY_WORDS = re.compile(
    r"\b(daily|twice|thrice|weekly|monthly|bid|tid|qid|qd|prn|hourly|per day|a day|per week)\b",
    re.I,
)


def _parse_drug(value: str, include_route_indication: bool) -> DrugInfo | None:
    """Parse 'name | dose | route | indication' into DrugInfo.
    Handles field-bleed where frequency lands in the route column."""
    parts = _split_pipes(value)
    name = _clean(parts[0]) if parts else None
    if not name:
        return None

    dose = _clean(parts[1]) if len(parts) > 1 else None

    route = None
    indication = None
    if include_route_indication:
        route = _clean(parts[2]) if len(parts) > 2 else None
        indication = _clean(parts[3]) if len(parts) > 3 else None

        # If "route" is actually a frequency (e.g. "twice daily"), merge into dose
        if route and _FREQUENCY_WORDS.search(route) and route.lower() not in _VALID_ROUTES:
            dose = f"{dose} {route}".strip() if dose else route
            route = None

        # If "route" isn't a recognized route, drop it
        if route and route.lower() not in _VALID_ROUTES:
            # Keep it only if it looks like a real route keyword
            if not any(r in route.lower() for r in _VALID_ROUTES):
                route = None

    return DrugInfo(name=name, dose=dose, route=route, indication=indication)


# --------------------------------------------------------------------------- #
# CLI smoke test                                                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    TEST_NARRATIVE = (
        "A 72-year-old female patient was prescribed Metformin 500mg twice "
        "daily for Type 2 Diabetes. She was also taking Lisinopril 10mg and "
        "Aspirin 81mg. Three days after starting Metformin, she experienced "
        "severe dizziness, nausea, and vomiting. She was hospitalized for "
        "2 days. Metformin was discontinued and symptoms resolved within "
        "48 hours."
    )

    print("=" * 70)
    print("EXTRACTOR SMOKE TEST")
    print("=" * 70)
    print(f"\nNarrative ({len(TEST_NARRATIVE.split())} words):")
    print(TEST_NARRATIVE)
    print("\nExtracting...\n")

    result = extract_report(TEST_NARRATIVE)

    import json
    print(json.dumps(result.model_dump(), indent=2))
