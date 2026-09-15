import json

from .ollama_client import OLLAMA_MODEL, chat_json
from .schemas import AnalysisResponse, Match, PatternDefinition


DEFAULT_PATTERNS = [
    # Diese Muster werden an Ollama übergeben und im Interview gesucht.
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


def analyze_transcript(transcript: str) -> AnalysisResponse:
    # Baut den Prompt, ruft Ollama auf und filtert unbekannte Muster heraus.
    pattern_text = "\n".join(
        f"- {pattern.name}: {pattern.description}" for pattern in DEFAULT_PATTERNS
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

    allowed = {pattern.name for pattern in DEFAULT_PATTERNS}
    matches = [m for m in matches if m.pattern in allowed]

    return AnalysisResponse(
        transcript=transcript,
        matches=matches,
        model=OLLAMA_MODEL,
    )
