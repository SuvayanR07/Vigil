"""
Severity classifier: CodedReport -> ClassifiedReport.

Uses a rules engine to scan for FDA seriousness keywords, with Ollama
fallback for ambiguous (negated) contexts.

FDA's 6 seriousness criteria (21 CFR 314.80):
  1. death
  2. life-threatening
  3. hospitalization (initial or prolonged)
  4. disability / permanent damage
  5. congenital anomaly / birth defect
  6. other medically important (required intervention)
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIDENCE_THRESHOLD
from pipeline.ollama_client import generate
from pipeline.schemas import ClassifiedReport, CodedReport


# --------------------------------------------------------------------------- #
# Keyword rules                                                                #
# --------------------------------------------------------------------------- #

# Each criterion maps to a list of lowercase substring keywords.
# Word-boundary regex is used so "admitted" matches but "readmitted" also matches.
SERIOUSNESS_KEYWORDS: dict[str, list[str]] = {
    "death": [
        "died", "death", "fatal", "deceased", "passed away", "expired",
    ],
    "life_threatening": [
        "life-threatening", "life threatening", "near death", "near-death",
        "resuscitated", "resuscitation", "cardiac arrest", "code blue",
    ],
    "hospitalization": [
        "hospitalized", "hospitalised", "hospitalization", "hospitalisation",
        "admitted to hospital", "admitted to the hospital", "inpatient",
        "emergency room", "er visit", "emergency department",
        "admission", "icu", "intensive care",
    ],
    "disability": [
        "disability", "disabled", "incapacitated", "permanent damage",
        "permanent impairment", "permanently impaired", "permanent disability",
    ],
    "congenital_anomaly": [
        "birth defect", "congenital anomaly", "congenital abnormality",
        "congenital malformation", "teratogenic",
    ],
    "required_intervention": [
        "surgery", "surgical intervention", "operation", "emergency procedure",
        "medical intervention required", "required intervention",
    ],
}

# Negation patterns: if a keyword is preceded by these within ~5 words, treat as ambiguous.
_NEGATION_RE = re.compile(
    r"\b(no|not|without|denies|denied|never|absence of|ruled out|excluded)\b"
    r"(?:\s+\w+){0,4}\s+$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def classify_severity(report: CodedReport) -> ClassifiedReport:
    """
    Scan the narrative + extracted fields for FDA seriousness criteria.

    Args:
        report: A CodedReport (extraction + MedDRA coding complete).

    Returns:
        ClassifiedReport with is_serious, seriousness_criteria, confidence,
        and review flags populated.
    """
    # Build the full text to scan (narrative + outcome + onset + reactions)
    search_corpus = _build_search_corpus(report)

    criteria: dict[str, bool] = {k: False for k in SERIOUSNESS_KEYWORDS}
    ambiguous: list[tuple[str, str]] = []  # (criterion, keyword) pairs needing LLM check

    for criterion, keywords in SERIOUSNESS_KEYWORDS.items():
        for kw in keywords:
            match = _find_keyword(search_corpus, kw)
            if not match:
                continue

            # Check for negation in the window BEFORE the match
            window_start = max(0, match.start() - 80)
            preceding = search_corpus[window_start:match.start()]

            if _NEGATION_RE.search(preceding):
                # Ambiguous — defer to LLM
                ambiguous.append((criterion, kw))
            else:
                criteria[criterion] = True
                break  # This criterion already confirmed; move on

    # Ollama disambiguation for ambiguous hits (only if not already confirmed)
    for criterion, keyword in ambiguous:
        if criteria[criterion]:
            continue  # Already confirmed by another keyword
        if _ollama_confirm(search_corpus, criterion, keyword):
            criteria[criterion] = True

    # Overall seriousness
    is_serious = any(criteria.values())

    # Confidence: high when we have clear keyword hits, lower when LLM-disambiguated
    if is_serious:
        n_confirmed = sum(criteria.values())
        severity_confidence = round(min(1.0, 0.7 + 0.1 * n_confirmed), 4)
    else:
        severity_confidence = 0.85  # Confident "not serious" when no keywords found

    # Build flags for review
    flags: list[str] = []
    for match in report.coded_reactions:
        if match.confidence < CONFIDENCE_THRESHOLD:
            flags.append(
                f"Low-confidence MedDRA match: '{match.verbatim_term}' -> "
                f"{match.pt_name} ({match.confidence:.2f})"
            )
    if report.narrative_truncated:
        flags.append("Narrative was truncated to fit model context window.")
    if ambiguous and not is_serious:
        flags.append("Ambiguous seriousness language (negated context) — human review advised.")

    return ClassifiedReport(
        **report.model_dump(),
        is_serious=is_serious,
        seriousness_criteria=criteria,
        severity_confidence=severity_confidence,
        flags_for_review=flags,
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _build_search_corpus(report: CodedReport) -> str:
    """Concatenate all free-text fields into one lowercase string for scanning."""
    parts: list[str] = []
    if report.narrative:
        parts.append(report.narrative)
    if report.outcome:
        parts.append(report.outcome)
    if report.onset_timeline:
        parts.append(report.onset_timeline)
    if report.dechallenge:
        parts.append(report.dechallenge)
    parts.extend(report.reactions_verbatim)
    return " ".join(parts).lower()


def _find_keyword(corpus: str, keyword: str) -> re.Match | None:
    """Find keyword with word boundaries (handles multi-word phrases)."""
    # Escape regex special chars, then add word boundaries around the whole phrase
    escaped = re.escape(keyword.lower())
    pattern = rf"\b{escaped}\b"
    return re.search(pattern, corpus)


_OLLAMA_SYSTEM = """You are a pharmacovigilance severity classifier.
Answer with a single word: YES or NO. No explanation."""


def _ollama_confirm(corpus: str, criterion: str, keyword: str) -> bool:
    """Ask Ollama whether a negated-context keyword actually indicates the criterion."""
    # Extract a local context window around the keyword (±150 chars)
    match = _find_keyword(corpus, keyword)
    if not match:
        return False

    start = max(0, match.start() - 150)
    end = min(len(corpus), match.end() + 150)
    snippet = corpus[start:end]

    criterion_readable = criterion.replace("_", " ")
    prompt = (
        f"Narrative excerpt: \"{snippet}\"\n\n"
        f"Does this excerpt indicate that the patient actually experienced "
        f"{criterion_readable}? Answer YES or NO only."
    )

    try:
        response = generate(prompt=prompt, system_prompt=_OLLAMA_SYSTEM)
        return response.strip().upper().startswith("YES")
    except Exception:
        # If Ollama fails, default to False (conservative — don't invent seriousness)
        return False


# --------------------------------------------------------------------------- #
# CLI smoke test                                                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import json

    from pipeline.schemas import PatientInfo

    # Build a minimal CodedReport for testing
    test = CodedReport(
        patient=PatientInfo(age="72 years", sex="female"),
        reactions_verbatim=["severe dizziness", "nausea", "vomiting"],
        outcome="recovered; hospitalized for 2 days",
        narrative=(
            "A 72-year-old female was hospitalized for 2 days after severe "
            "dizziness, nausea, and vomiting."
        ),
    )

    result = classify_severity(test)
    print(json.dumps(result.model_dump(), indent=2))
