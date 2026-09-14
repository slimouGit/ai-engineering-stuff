from typing import List, Optional
from pydantic import BaseModel, Field


class PatternDefinition(BaseModel):
    name: str
    description: str


class TextAnalysisRequest(BaseModel):
    transcript: str = Field(min_length=1)
    patterns: Optional[List[PatternDefinition]] = None


class Match(BaseModel):
    pattern: str
    evidence: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class AnalysisResponse(BaseModel):
    transcript: str
    matches: List[Match]
    model: str
