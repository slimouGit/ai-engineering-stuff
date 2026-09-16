from .patterns import DEFAULT_PATTERNS


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


def build_user_prompt(chunk: str) -> str:
    pattern_text = "\n".join(
        f"- {pattern.name}: {pattern.description}"
        for pattern in DEFAULT_PATTERNS
    )

    return f"""MUSTER:
{pattern_text}

TRANSKRIPT:
{chunk}

Finde alle relevanten Treffer.
Mehrere Treffer pro Muster sind erlaubt.
Gib höchstens 3 der wichtigsten Treffer pro Muster zurück.
Halte evidence und explanation jeweils kurz.
"""
