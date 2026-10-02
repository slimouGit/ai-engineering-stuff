# Embeddings – projektnahe Lern-App

Diese kleine Konsolen-App zeigt einen für textorientierte ML-/AI-Projekte typischen Embedding-Workflow.

## Was wird demonstriert?

```text
Referenztexte
    ↓
Ollama Embeddings
    ↓
Vektoren
    ├── Cosine Similarity → ähnlichste Textstellen
    └── K-Means → semantische Cluster
```

Zusätzlich besitzen die Referenztexte synthetische Ground-Truth-Labels. Dadurch kann man sehen, wie semantische Ähnlichkeit und bekannte annotierte Beispiele zusammenspielen können.

## Warum ist das projektrelevant?

Bei Texten können zwei Aussagen inhaltlich ähnlich sein, obwohl sie unterschiedliche Wörter verwenden.

Beispiel:

```text
"Ich übermittelte Nachrichten zwischen Mitgliedern."
```

und

```text
"Ich war für die Weitergabe von Informationen verantwortlich."
```

TF-IDF sieht vor allem Wörter. Embeddings versuchen stärker, die semantische Bedeutung abzubilden.

## Voraussetzungen

Python-Pakete:

```bash
pip install -r requirements.txt
```

Ollama muss lokal laufen.

Embedding-Modell installieren:

```bash
ollama pull nomic-embed-text
```

## Start

```bash
python app.py
```

## Begriffe

### Embedding

Ein Text wird in einen numerischen Vektor umgewandelt.

```text
"Ich gebe Informationen weiter"
        ↓
[0.12, -0.08, 0.42, ...]
```

Semantisch ähnliche Texte sollen ähnliche Vektoren erhalten.

### Cosine Similarity

Vergleicht die Richtung zweier Vektoren.

Grob:

```text
nahe 1 → sehr ähnlich
nahe 0 → wenig ähnlich
```

### K-Means

Gruppiert Embeddings ohne bekannte Zielklasse.

Das ist Unsupervised Learning.

### Ground Truth

Die bekannte fachlich richtige Klasse eines Referenzfalls.

In dieser App sind die Labels absichtlich synthetisch und nur für Lernzwecke gedacht.

## Wichtiger Unterschied zur Klassifikation

Die semantische Suche sagt:

> Welche bekannten Texte sind diesem Text ähnlich?

Ein trainierter Klassifikator sagt:

> Welche Klasse soll dieser Text bekommen?

Embeddings können Features für einen späteren Klassifikator sein, ersetzen die Modellvalidierung aber nicht.

## Sinnvolle nächste Erweiterung

Ein nächster Schritt wäre:

```text
Text
→ Embedding
→ Logistic Regression / Linear SVC
→ Cross-Validation
→ Precision / Recall / F1
→ Error Analysis
```

Damit würdest du TF-IDF und Embeddings auf derselben Klassifikationsaufgabe direkt vergleichen.
