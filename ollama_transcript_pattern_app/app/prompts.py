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
confidence ist Pflicht. Wenn du keine belastbare Confidence angeben kannst, gib den Treffer nicht aus.
Vergib hohe Werte nur bei einer eindeutigen fachlichen Zuordnung.
Ordne eine Textstelle nur einem einzigen Muster zu. Wenn mehrere Muster möglich erscheinen,
wähle das fachlich passendste und gib keine zusätzlichen Treffer für dieselbe Textstelle aus.
Für ein Muster darfst du mehrere unterschiedliche Textstellen zurückgeben.
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

Finde nur fachlich eindeutige Treffer.
Gib höchstens 2 der wichtigsten Treffer pro Muster zurück.
Verwende als evidence nur den kürzesten relevanten Originalausschnitt, nicht einen ganzen Dialog.
Ordne Fragen, Namen von Ärzten und allgemeine Gesprächsanteile nicht automatisch Behandlungsmustern zu.
Ordne Wörter wie "heute", "jetzt" oder "direkt" nur dann dem Verlauf zu, wenn sie den
Beginn, die Dauer oder die Veränderung eines Symptoms beschreiben.
Ordne "nicht" nur dann einer Verneinung zu, wenn tatsächlich ein medizinischer Sachverhalt
verneint oder ausgeschlossen wird.
Halte explanation kurz und begründe die konkrete Zuordnung.
"""
