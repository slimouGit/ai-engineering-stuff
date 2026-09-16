import re

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


CHUNK_SIZE = 6000
CHUNK_OVERLAP_SENTENCES = 2
ABBREVIATION_PATTERN = re.compile(
    r"\b(?:Dr|Prof|Herr|Frau|z\.\s*B|d\.\s*h|bzw|usw|etc|vgl)\.",
    re.IGNORECASE,
)
PERIOD_MARKER = "__PERIOD__"


def split_sentences(text: str) -> list[str]:
    protected_text = ABBREVIATION_PATTERN.sub(
        lambda match: match.group().replace(".", PERIOD_MARKER),
        text.strip(),
    )
    sentences = re.split(
        r'(?<=[.!?])\s+|\n+',
        protected_text,
    )

    return [
        sentence.replace(PERIOD_MARKER, ".").strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    text: str,
    max_chars: int = CHUNK_SIZE,
    overlap_sentences: int = CHUNK_OVERLAP_SENTENCES,
) -> list[str]:

    sentences = split_sentences(text)

    chunks = []
    current_sentences = []

    for sentence in sentences:
        candidate = " ".join(
            current_sentences + [sentence]
        )

        if current_sentences and len(candidate) > max_chars:
            chunks.append(" ".join(current_sentences))

            current_sentences = (
                current_sentences[-overlap_sentences:]
                if overlap_sentences > 0
                else []
            )

        current_sentences.append(sentence)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def analyze_chunk(chunk: str) -> list[Match]:
    pattern_text = "\n".join(
        f"- {pattern.name}: {pattern.description}"
        for pattern in DEFAULT_PATTERNS
    )

    user_prompt = f"""MUSTER:
{pattern_text}

TRANSKRIPT:
{chunk}

Finde alle relevanten Treffer.
Mehrere Treffer pro Muster sind erlaubt.
Gib höchstens 3 der wichtigsten Treffer pro Muster zurück.
Halte evidence und explanation jeweils kurz.
"""

    raw = chat_json(SYSTEM_PROMPT, user_prompt)

    matches = [
        Match.model_validate(item)
        for item in raw.get("matches", [])
    ]

    allowed = {
        pattern.name
        for pattern in DEFAULT_PATTERNS
    }

    return [
        match
        for match in matches
        if match.pattern in allowed
    ]


def remove_duplicate_matches(
    matches: list[Match]
) -> list[Match]:

    unique_matches = []
    seen = set()

    for match in matches:
        key = (
            match.pattern,
            match.evidence.strip().lower(),
        )

        if key not in seen:
            seen.add(key)
            unique_matches.append(match)

    return unique_matches


def analyze_transcript(transcript: str) -> AnalysisResponse:
    chunks = chunk_text(transcript)

    all_matches = []

    for chunk in chunks:
        all_matches.extend(
            analyze_chunk(chunk)
        )

    all_matches = remove_duplicate_matches(
        all_matches
    )

    return AnalysisResponse(
        transcript=transcript,
        matches=all_matches,
        model=OLLAMA_MODEL,
    )