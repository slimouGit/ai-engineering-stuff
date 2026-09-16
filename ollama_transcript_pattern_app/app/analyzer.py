import re

from .config import ABBREVIATION_PATTERN, CHUNK_OVERLAP_SENTENCES, CHUNK_SIZE, PERIOD_MARKER
from .ollama_client import OLLAMA_MODEL, chat_json
from .patterns import DEFAULT_PATTERNS
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schemas import AnalysisResponse, Match


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
    user_prompt = build_user_prompt(chunk)

    raw = chat_json(SYSTEM_PROMPT, user_prompt)

    matches = []
    for item in raw.get("matches", []):
        if not isinstance(item, dict):
            continue
        item = dict(item)
        try:
            matches.append(Match.model_validate(item))
        except ValueError:
            continue

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
    seen_evidence = set()

    for match in matches:
        key = re.sub(r"\s+", " ", match.evidence.strip().lower())

        if not key or key in seen_evidence:
            continue

        if any(
            key in re.sub(r"\s+", " ", existing.evidence.strip().lower())
            or re.sub(r"\s+", " ", existing.evidence.strip().lower()) in key
            for existing in unique_matches
        ):
            continue

        seen_evidence.add(key)
        unique_matches.append(match)

    return unique_matches


def analyze_transcript(transcript: str) -> AnalysisResponse:
    chunks = chunk_text(transcript)

    all_matches = []

    for chunk in chunks:
        all_matches.extend(analyze_chunk(chunk))

    all_matches = remove_duplicate_matches(
        all_matches
    )

    return AnalysisResponse(
        transcript=transcript,
        matches=all_matches,
        model=OLLAMA_MODEL,
    )