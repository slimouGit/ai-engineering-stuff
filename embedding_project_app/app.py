"""
Projektnahe Lern-App: Embeddings + semantische Ähnlichkeit
==========================================================

Ziel:
    Ein neuer Textabschnitt wird nicht über einzelne Schlüsselwörter,
    sondern über seine semantische Bedeutung mit bekannten Beispielen verglichen.

Beispielhafte Pipeline:
    Text
      -> Embedding
      -> Vektor
      -> Cosine Similarity
      -> ähnlichste bekannte Textstellen
      -> optionale Clusteranalyse

Wichtig:
    Die Daten sind rein synthetische Lernbeispiele.
    Die App trifft keine fachlichen oder behördlichen Entscheidungen.

Voraussetzung:
    Ollama läuft lokal und das Embedding-Modell ist vorhanden:

        ollama pull nomic-embed-text

    Danach:
        python workflow-demo-app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. KONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"

CACHE_FILE = Path(__file__).parent / "embeddings_cache.json"


# ============================================================
# 2. SYNTHETISCHE REFERENZDATEN
# ============================================================

# Ground Truth:
# Die Labels sind hier absichtlich bereits bekannt.
# In einem realen Projekt würden sie z. B. durch fachliche Annotation entstehen.
#
# Die App nutzt die Labels NICHT zum Erzeugen der Embeddings.
# Die Labels helfen uns nur dabei, die gefundenen ähnlichen Beispiele zu interpretieren.

DATA = [
    {
        "id": "T001",
        "text": "Ich möchte meinen Termin verschieben, weil ich an diesem Tag nicht kommen kann.",
        "label": "nicht_relevant",
    },
    {
        "id": "T002",
        "text": "Meine Adresse hat sich geändert und ich möchte die neuen Kontaktdaten mitteilen.",
        "label": "nicht_relevant",
    },
    {
        "id": "T003",
        "text": "Ich benötige eine Kopie meiner Unterlagen für meine persönlichen Dokumente.",
        "label": "nicht_relevant",
    },
    {
        "id": "T004",
        "text": "Ich möchte wissen, wie der aktuelle Bearbeitungsstand meines Verfahrens ist.",
        "label": "nicht_relevant",
    },
    {
        "id": "T005",
        "text": "Ich hatte wiederholt Kontakt zu Mitgliedern einer bewaffneten Gruppe.",
        "label": "relevant",
    },
    {
        "id": "T006",
        "text": "Ich nahm mehrfach an Treffen einer militanten Organisation teil.",
        "label": "relevant",
    },
    {
        "id": "T007",
        "text": "Ich sollte Material und Nachrichten für Mitglieder einer bewaffneten Einheit transportieren.",
        "label": "relevant",
    },
    {
        "id": "T008",
        "text": "Ich habe Geld gesammelt und an Mitglieder einer gewaltbereiten Gruppe weitergegeben.",
        "label": "relevant",
    },
    {
        "id": "T009",
        "text": "Ich kannte einige Personen aus der Organisation, hatte aber keinen weiteren Kontakt.",
        "label": "nicht_relevant",
    },
    {
        "id": "T010",
        "text": "Mir wurde aufgetragen, Dokumente an Mitglieder der Gruppe zu überbringen.",
        "label": "relevant",
    },
    {
        "id": "T011",
        "text": "Ich habe den Namen der Organisation nur in den Nachrichten gehört.",
        "label": "nicht_relevant",
    },
    {
        "id": "T012",
        "text": "Ich war für die Weitergabe von Informationen zwischen mehreren Mitgliedern verantwortlich.",
        "label": "relevant",
    },
]


# ============================================================
# 3. EMBEDDING-FUNKTION
# ============================================================

def create_embeddings(texts):
    """
    Wandelt mehrere Texte über Ollama in Embedding-Vektoren um.

    Ein Embedding ist vereinfacht:
        Text -> Liste von Zahlen

    Semantisch ähnliche Texte sollen im Vektorraum näher beieinander liegen.
    """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        },
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    # Ollama /api/embed liefert:
    # {
    #   "embeddings": [
    #       [0.123, ...],
    #       [...]
    #   ]
    # }
    return np.array(result["embeddings"], dtype=float)


# ============================================================
# 4. CACHE
# ============================================================

def load_or_create_reference_embeddings(df):
    """
    Referenztexte müssen nicht bei jedem Programmstart neu eingebettet werden.

    Wenn sich Modell oder Texte ändern, wird der Cache automatisch neu erstellt.
    """

    signature = {
        "model": EMBEDDING_MODEL,
        "texts": df["text"].tolist(),
    }

    if CACHE_FILE.exists():
        cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

        if cached.get("signature") == signature:
            print("Embeddings aus Cache geladen.")
            return np.array(cached["embeddings"], dtype=float)

    print("Referenztexte werden eingebettet ...")
    embeddings = create_embeddings(df["text"].tolist())

    CACHE_FILE.write_text(
        json.dumps(
            {
                "signature": signature,
                "embeddings": embeddings.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return embeddings


# ============================================================
# 5. SEMANTISCHE SUCHE
# ============================================================

def find_similar(query, df, reference_embeddings, top_k=5):
    print("query ", query)
    print("refs ", reference_embeddings)
    """
    1. Query wird eingebettet.
    2. Cosine Similarity vergleicht den Query-Vektor mit allen Referenzvektoren.
    3. Die ähnlichsten Texte werden zurückgegeben.
    """

    query_embedding = create_embeddings([query])

    similarities = cosine_similarity(
        query_embedding,
        reference_embeddings,
    )[0]

    result = df.copy()
    result["similarity"] = similarities

    return result.sort_values(
        "similarity",
        ascending=False,
    ).head(top_k)


# ============================================================
# 6. EINFACHE LABEL-ORIENTIERUNG
# ============================================================

def similarity_based_label(similar_rows):
    """
    Lernbeispiel:
    Wir bilden aus den ähnlichsten bekannten Beispielen eine gewichtete Orientierung.

    Das ist KEIN trainierter Klassifikator.
    Es demonstriert nur, wie Ground-Truth-Beispiele zusammen mit
    semantischer Ähnlichkeit genutzt werden könnten.
    """

    scores = {
        "relevant": 0.0,
        "nicht_relevant": 0.0,
    }

    for _, row in similar_rows.iterrows():
        scores[row["label"]] += float(row["similarity"])

    suggestion = max(scores, key=scores.get)

    return suggestion, scores


# ============================================================
# 7. CLUSTERING DER EMBEDDINGS
# ============================================================

def cluster_reference_data(df, embeddings):
    """
    K-Means arbeitet direkt auf den Embedding-Vektoren.

    Es gibt hier kein y für das Clustering.
    Die bereits vorhandenen Labels werden nur danach angezeigt,
    damit wir die Cluster fachlich interpretieren können.
    """

    n_clusters = 3

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    cluster_ids = model.fit_predict(embeddings)

    clustered = df.copy()
    clustered["cluster"] = cluster_ids

    score = silhouette_score(
        embeddings,
        cluster_ids,
        metric="cosine",
    )

    return clustered, score


# ============================================================
# 8. HAUPTPROGRAMM
# ============================================================

def main():
    print("\n============================================")
    print("EMBEDDINGS – SEMANTISCHE TEXTANALYSE")
    print("============================================")

    df = pd.DataFrame(DATA)

    print("\nReferenzdaten:")
    print(df[["id", "label", "text"]].to_string(index=False))

    try:
        reference_embeddings = load_or_create_reference_embeddings(df)
    except requests.RequestException as exc:
        print("\nFEHLER:")
        print("Ollama konnte nicht erreicht werden.")
        print("Prüfe, ob Ollama läuft und das Modell installiert ist:")
        print(f"  ollama pull {EMBEDDING_MODEL}")
        print("\nTechnischer Fehler:")
        print(exc)
        return

    # --------------------------------------------------------
    # Clustering
    # --------------------------------------------------------

    clustered, silhouette = cluster_reference_data(
        df,
        reference_embeddings,
    )

    print("\n--- Cluster der Referenztexte ---")
    print(
        clustered[
            ["id", "label", "cluster", "text"]
        ].sort_values("cluster").to_string(index=False)
    )

    print("\nSilhouette Score:", round(silhouette, 3))

    # --------------------------------------------------------
    # Eigene semantische Suche
    # --------------------------------------------------------

    print("\n--- Neuen Text untersuchen ---")

    query = input(
        "Text eingeben [Enter = Beispiel]: "
    ).strip()

    if not query:
        query = (
            "Ich musste Informationen zwischen verschiedenen "
            "Mitgliedern der Organisation weitergeben."
        )

    similar = find_similar(
        query=query,
        df=df,
        reference_embeddings=reference_embeddings,
        top_k=5,
    )

    print("\nEingabe:")
    print(query)

    print("\n--- Ähnlichste bekannte Texte ---")

    for _, row in similar.iterrows():
        print()
        print(
            f'{row["id"]} | '
            f'Label: {row["label"]} | '
            f'Similarity: {row["similarity"]:.3f}'
        )
        print(row["text"])

    # --------------------------------------------------------
    # Ground-Truth-basierte Orientierung
    # --------------------------------------------------------

    suggestion, scores = similarity_based_label(similar)

    print("\n--- Einfache labelbasierte Orientierung ---")
    print("Gewicht relevant:      ", round(scores["relevant"], 3))
    print("Gewicht nicht_relevant:", round(scores["nicht_relevant"], 3))
    print("Orientierung:", suggestion)

    print("""
============================================
WAS DU HIER LERNST
============================================

1. TF-IDF:
   vergleicht stark über Wörter und Wortkombinationen.

2. Embeddings:
   bilden die Bedeutung eines ganzen Textes als Vektor ab.

3. Cosine Similarity:
   misst, wie ähnlich zwei Embedding-Vektoren sind.

   grob:
       nahe 1 -> sehr ähnlich
       nahe 0 -> wenig ähnlich

4. K-Means:
   kann Embeddings ohne Zielvariable in Gruppen clustern.

5. Ground Truth:
   kann anschließend genutzt werden, um gefundene ähnliche
   Beispiele fachlich einzuordnen.

6. Wichtig:
   Embedding Similarity ist noch keine fertige Klassifikation.
   Für ein echtes System müssten Thresholds, Evaluation,
   Ground Truth, Fehleranalyse und fachliche Regeln geprüft werden.
""")


if __name__ == "__main__":
    main()
