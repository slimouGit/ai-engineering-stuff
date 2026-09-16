from .schemas import PatternDefinition


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
