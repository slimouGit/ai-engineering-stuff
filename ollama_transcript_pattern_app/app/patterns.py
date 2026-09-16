from .schemas import PatternDefinition


DEFAULT_PATTERNS = [
    PatternDefinition(
        name="beschwerden_symptome",
        description="Konkrete aktuelle Beschwerden oder Symptome der Patientin, zum Beispiel Schmerzen, Atemnot, Übelkeit oder Ausstrahlung. Keine bloßen Fragen oder allgemeinen Aussagen.",
    ),
    PatternDefinition(
        name="zeitangabe_verlauf",
        description="Konkrete Angaben zu Beginn, Dauer, Häufigkeit oder Veränderung eines aktuellen Symptoms. Keine isolierten Wörter wie heute, jetzt oder früher.",
    ),
    PatternDefinition(
        name="medikamente_behandlung",
        description="Konkrete Einnahme oder Anwendung eines Medikaments sowie eine konkrete Therapie oder Behandlung der Patientin. Ein Arztname oder eine allgemeine Frage zählt nicht.",
    ),
    PatternDefinition(
        name="vorgeschichte_risikofaktoren",
        description="Relevante eigene Vorerkrankungen, eigene Operationen oder persönliche Risikofaktoren der Patientin. Familienmitglieder zählen nur bei ausdrücklich relevantem familiärem Risiko.",
    ),
    PatternDefinition(
        name="verneinung_ausschluss",
        description="Medizinisch relevante Verneinung oder Ausschluss eines Symptoms, einer Allergie, eines Medikaments oder einer Vorerkrankung. Ein allgemeines nein reicht nicht.",
    ),
    PatternDefinition(
        name="frage_antwort_struktur",
        description="Ein vollständiges, inhaltlich erkennbares Frage-Antwort-Paar. Einzelne Fragen ohne Antwort nicht markieren.",
    ),
]
