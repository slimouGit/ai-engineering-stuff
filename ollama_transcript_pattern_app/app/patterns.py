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
        description="Relevante eigene Vorerkrankungen, eigene Operationen, persönliche Risikofaktoren oder ausdrücklich genannte familiäre Vorerkrankungen mit möglicher medizinischer Bedeutung, zum Beispiel Schlaganfall, Hirnblutung, Herzinfarkt oder Krebs in der Familie. Irrelevante Familieninformationen nicht markieren.",
    ),
    PatternDefinition(
        name="verneinung_ausschluss",
        description="Medizinisch relevante Verneinung oder Ausschluss eines Symptoms, einer Allergie, eines Medikaments oder einer Vorerkrankung. Beziehung, Familienstand oder soziale Angaben wie 'nicht verheiratet' sind niemals Treffer. Ein allgemeines nein reicht nicht.",
    ),
    PatternDefinition(
        name="frage_antwort_struktur",
        description="Ein explizites Frage-Antwort-Paar mit einer Frage und einer direkt anschließenden inhaltlichen Antwort. Implizite Fragen, einzelne Antworten und Gesprächswechsel ohne klar erkennbare Frage nicht markieren.",
    ),
]
