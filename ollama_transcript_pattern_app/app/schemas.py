from typing import List
from pydantic import BaseModel, Field


class PatternDefinition(BaseModel):
    """Diese Klasse speichert den Namen und die Beschreibung eines Suchmusters."""

    name: str
    description: str


class Match(BaseModel):
    """Diese Klasse speichert einen Treffer mit Textbeleg und Confidence-Wert."""

    pattern: str
    evidence: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class AnalysisResponse(BaseModel):
    """Diese Klasse speichert das vollständige Ergebnis der Interviewanalyse."""

    transcript: str
    matches: List[Match]
    model: str
