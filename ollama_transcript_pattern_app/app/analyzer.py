import json
from typing import Iterable, List

from .ollama_client import OLLAMA_MODEL, chat_json
from .schemas import AnalysisResponse, Match, PatternDefinition


DEFAULT_PATTERNS = [
    PatternDefinition(
        name="beschwerden_symptome",
        description="Aussagen über Beschwerden, Symptome, Schmerzen oder körperliche Probleme.",
    ),
    PatternDefinition(
        name="zeitangabe_verlauf",
        description="Zeitliche Angaben: seit wann, Dauer, Beginn, Häufigkeit oder Verlauf.",
    ),
    PatternDefinition(
        name="medikamente_behandlung",
        description="Medikamente, bisherige Behandlung, Therapie oder Dosierung.",
    ),
    PatternDefinition(
        name="vorgeschichte_risikofaktoren",
        description="Vorerkrankungen, frühere Ereignisse, Operationen oder relevante Risikofaktoren.",
    ),
    PatternDefinition(
        name="verneinung_ausschluss",
        description="Explizite Verneinungen oder Ausschlüsse, zum Beispiel 'nein', 'nicht', 'keine'.",
    ),
    PatternDefinition(
        name="frage_antwort_struktur",
        description="Erkennbare Frage-Antwort-Paare oder Interviewer-/Interviewten-Wechsel.",
    ),
]


SYSTEM_PROMPT = """Du analysierst deutsche Interviewtranskripte.
Deine Aufgabe ist ausschließlich, Textstellen den vorgegebenen Mustern zuzuordnen.
Erfinde keine Informationen und verwende nur Text, der im Transkript tatsächlich vorkommt.

Antworte ausschließlich als JSON in exakt dieser Struktur:
{
  "matches": [
    {
      "pattern": "name_des_musters",
      "evidence": "kurzes wörtliches oder sehr nahes Textfragment aus dem Transkript",
      "explanation": "kurze Begründung",
      "confidence": 0.0
    }
  ]
}

confidence liegt zwischen 0 und 1.
Wenn kein Muster gefunden wird, gib {"matches": []} zurück.
"""


def analyze_transcript(
    transcript: str,
    patterns: Iterable[PatternDefinition] | None = None,
) -> AnalysisResponse:
    selected_patterns: List[PatternDefinition] = list(patterns or DEFAULT_PATTERNS)

    pattern_text = "\n".join(
        f"- {p.name}: {p.description}" for p in selected_patterns
    )

    user_prompt = f"""MUSTER:
{pattern_text}

TRANSKRIPT:
{transcript}

Finde alle relevanten Treffer. Mehrere Treffer pro Muster sind erlaubt.
Gib höchstens 3 der wichtigsten Treffer pro Muster zurück.
Halte evidence und explanation jeweils kurz.
"""

    raw = chat_json(SYSTEM_PROMPT, user_prompt)
    matches = [Match.model_validate(item) for item in raw.get("matches", [])]

    allowed = {p.name for p in selected_patterns}
    matches = [m for m in matches if m.pattern in allowed]

    return AnalysisResponse(
        transcript=transcript,
        matches=matches,
        model=OLLAMA_MODEL,
    )
