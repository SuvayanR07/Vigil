"""
Pydantic data models for the VIGIL pipeline.

Data flow:
  raw text  -->  ExtractedReport  -->  CodedReport  -->  ClassifiedReport
              extractor.py      meddra_coder.py    severity.py
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Core entities                                                                #
# --------------------------------------------------------------------------- #

class PatientInfo(BaseModel):
    """Basic patient demographics pulled from the narrative."""
    age: Optional[str] = None         # "72-year-old", "72 years", "unknown"
    sex: Optional[str] = None         # "male", "female", "unknown"
    weight: Optional[str] = None      # "65kg", "150 lbs", None


class DrugInfo(BaseModel):
    """A drug mentioned in the report (suspect or concomitant)."""
    name: str                          # "Metformin"
    dose: Optional[str] = None         # "500mg twice daily"
    route: Optional[str] = None        # "oral", "IV"
    indication: Optional[str] = None   # "Type 2 Diabetes"


class MedDRAMatch(BaseModel):
    """A single verbatim reaction with its best MedDRA coding."""
    verbatim_term: str
    pt_code: str
    pt_name: str
    soc_name: str
    confidence: float
    candidates: list[dict] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Pipeline stages                                                              #
# --------------------------------------------------------------------------- #

class ExtractedReport(BaseModel):
    """
    Output of pipeline/extractor.py.
    Raw entities pulled from the free-text narrative, not yet coded.
    """
    patient: PatientInfo = Field(default_factory=PatientInfo)
    suspect_drugs: list[DrugInfo] = Field(default_factory=list)
    concomitant_drugs: list[DrugInfo] = Field(default_factory=list)
    reactions_verbatim: list[str] = Field(default_factory=list)
    onset_timeline: Optional[str] = None
    dechallenge: Optional[str] = None
    outcome: Optional[str] = None
    reporter_type: Optional[str] = None
    narrative: Optional[str] = None
    narrative_truncated: bool = False


class CodedReport(ExtractedReport):
    """ExtractedReport + MedDRA-coded reactions."""
    coded_reactions: list[MedDRAMatch] = Field(default_factory=list)


class ClassifiedReport(CodedReport):
    """CodedReport + severity classification and review flags."""
    is_serious: bool = False
    seriousness_criteria: dict = Field(default_factory=dict)
    severity_confidence: float = 0.0
    flags_for_review: list[str] = Field(default_factory=list)
