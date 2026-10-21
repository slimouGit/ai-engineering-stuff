# AI Engineering Stuff – App-Übersicht

Diese README fasst die **ausführbaren Lern-, Demo- und Projekt-Apps** im Repository zusammen. Sie beschreibt jeweils **Ort**, **Einstiegspunkt**, **Kurzfunktion** und **Intention**.

> Hinweis: Einige Ordner enthalten nur Dokumentation, Modellreste oder IDE-Dateien und sind aktuell keine eigenständig ausführbaren Apps. Diese sind am Ende separat aufgeführt.

## Schnellübersicht

| Bereich | App / Ordner | Einstiegspunkt | Kurzbeschreibung |
|---|---|---|---|
| API / Ollama | `FastAPI/` | `server.py`, `client.py` | Kleine Streaming-API vor Ollama |
| LoRA / Fine-Tuning | `bamf_lora_clean/` | `app.py` | Lokaler LoRA-Trainingsworkflow bis GGUF/Ollama |
| ML-Grundlagen | `bamf_ml_preparation/` | `train.py`, `app/main.py` | Textklassifikation + FastAPI |
| Bias | `bias_demo_app/` | `app.py` | Einfaches Bias-/Threshold-Beispiel |
| Embeddings | `embedding_project_app/` | `app.py` | Embeddings, Cosine Similarity, K-Means |
| Encoder | `encoder_example/` | `encoder_example.py` | DistilBERT-Encoder für Textklassifikation |
| IBM ICA | `ica_dummies/` | `cli.py` | Kleine ICA-API-/Agent-Demos |
| MCP | `mcp-demo/` | `app.py`, `mcp_server.py` | Ollama + MCP + Ticket-Tools |
| MLflow | `mlflow_demo/` | mehrere Skripte | Tracking, Traces, Prompts, Thresholds |
| Multi-Agent | `multi-agent-support-demo/` | `app.py` | Mehrere Agenten + RAG + MCP + Ollama |
| NLP / Sequenzen | `next-word-prediction/` | `next_word_prediction.py` | LSTM Next-Word-Prediction |
| NumPy | `numpy/` | `numpy-app.py` | NumPy-Grundlagen |
| Transcript-Analyse | `ollama_transcript_pattern_app/` | `app/main.py` | Lokale Transcript-Musteranalyse mit Ollama |
| Pandas | `pandas-sandbox/` | `pandas-app.py` | DataFrame-Grundlagen und Datenprüfung |
| RAG | `rag-demo/` | `src/app.py` | Kleine modulare RAG-Konsole |
| RAG | `rag-demo-py/` | `app.py`, `api.py` | Größere RAG-App mit Flask/FastAPI/SQLite/Ollama |
| scikit-learn | `scikit-learn-Lernpfad-sandbox/` | mehrere Skripte | ML-Lernpfad mit mehreren Teil-Apps |
| Attention | `self-attention/` | `app.py`, `bert-based.py` | Self-Attention einmal didaktisch, einmal mit BERT |
| Mini-LLM | `simple-chat-bot/` | `train.py`, `ask.py` | Kleines DistilGPT2-Fine-Tuning und Q&A |
| ML-Modelle | `sklearn_methoden_apps/` | `01_...py` bis `11_...py` | Kleine Demos zu klassischen ML-Verfahren |
| Tokenisierung | `sliding-window/` | `SimpleSubwordTokenizer.py` | Sliding Windows für Next-Token-Daten |
| Klassifikation | `sms-spam-classifier/` | `app.py` | SMS Spam/Ham inklusive Training und Persistenz |
| Klassifikation/API | `ticket-classification/` | `train.py`, `app.py` | Support-Ticket-Klassifikation als FastAPI |
| Speech-to-Text | `transcripter/` | `main.py` | Whisper-Transkription mit CUDA-Erkennung |

---

# 1. API, LLM und lokale Modelle

## `FastAPI/`

**Ort:** `FastAPI/`  
**Einstieg:** `server.py` und `client.py`

**Was macht die App?**  
`server.py` stellt eine kleine FastAPI vor einen lokalen Ollama-Server. Chat-Anfragen werden an Ollama weitergereicht und als Stream zurückgegeben. `client.py` ist ein einfacher Konsolenclient mit konfigurierbaren LLM-Parametern wie `temperature`, `top_p`, `top_k`, `num_predict` und `num_ctx`.

**Intention:**  
Verstehen, wie eine eigene Python-API zwischen Client und lokalem LLM sitzt und wie Streaming sowie Modellparameter über HTTP übertragen werden.

---

## `encoder_example/`

**Ort:** `encoder_example/encoder_example.py`  
**Einstieg:** `encoder_example.py`

**Was macht die App?**  
Verwendet einen DistilBERT-Encoder über Hugging Face `pipeline()` und klassifiziert eingegebenen Text als positives oder negatives Sentiment.

**Intention:**  
Einfacher Einstieg in den Unterschied zwischen **Encoder-Modellen zur Klassifikation** und generativen LLMs.

---

## `ica_dummies/`

**Ort:** `ica_dummies/`  
**Wichtige Dateien:** `cli.py`, `ica-connector.py`, `ica-models.py`, `ica-completions.py`

**Was macht die App?**  
Mehrere kleine Testskripte für die Kommunikation mit IBM-ICA-Endpunkten. Unter anderem werden Agents/Modelle abgefragt und Completion-Aufrufe vorbereitet.

**Intention:**  
Die externe ICA-Schnittstelle isoliert testen, bevor sie in eine größere Anwendung integriert wird.

---

# 2. Fine-Tuning und LoRA

## `bamf_lora_clean/`

**Ort:** `bamf_lora_clean/`  
**Einstieg:** `app.py`  
**Konfiguration:** `config.py`

**Was macht die App?**  
Vollständiger lokaler Lernworkflow:

```text
synthetische Trainingsdaten
→ lokales Transformers-Modell
→ LoRA-Training
→ Adapter speichern
→ Adapter + Basismodell mergen
→ GGUF erzeugen
→ als Ollama-Modell importieren
→ Vorher/Nachher vergleichen
```

**Intention:**  
Den kompletten Weg vom Basismodell bis zu einer lokal nutzbaren, angepassten Ollama-Variante nachvollziehen. Die Daten sind ausdrücklich synthetisch und dienen nur dem Lernen.

---

## `bamf_lora_demo/`

**Ort:** `bamf_lora_demo/`  
**Status:** Dokumentation und Modell-/Adapterreste vorhanden; im aktuellen Repository-Snapshot fehlt die eigentliche `app.py`.

**Was beschreibt der Ordner?**  
Eine ältere/alternative LoRA-Lernvariante mit PEFT, lokalem Transformers-Modell und anschließendem Ollama-Import.

**Intention:**  
LoRA-Grundidee verstehen: Basismodell + gelabelte Beispiele → Adapter → zusammengeführtes Modell.

---

## `bamf_lora_pycharm/` und `bamf_lora_simple_auto/`

**Status:** Keine vollständige eigenständige App im aktuellen Snapshot.  
`bamf_lora_pycharm/` enthält im Wesentlichen Dokumentation, `bamf_lora_simple_auto/` vor allem Modellordner.

**Intention:**  
Frühere bzw. vorbereitende Varianten des LoRA-Lernprojekts.

---

# 3. Klassisches Machine Learning und Textklassifikation

## `bamf_ml_preparation/`

**Ort:** `bamf_ml_preparation/`  
**Training:** `train.py`  
**API:** `app/main.py`

**Was macht die App?**  
Trainiert eine Textklassifikation mit `TF-IDF + LogisticRegression`, gibt Klassifikationsmetriken aus und speichert das Modell als `model.joblib`. Die FastAPI lädt das Modell und bietet `/health` und `/predict` an.

**Intention:**  
Kompakter End-to-End-Workflow für klassische Textklassifikation: Daten → Split → Training → Evaluation → Modell speichern → API.

---

## `ticket-classification/`

**Ort:** `ticket-classification/`  
**Training:** `train.py`  
**API:** `app.py`  
**Modellwrapper:** `model.py`

**Was macht die App?**  
Klassifiziert Support-Tickets mit `TF-IDF + LogisticRegression` in Kategorien wie Technik, Zugang, Abrechnung oder Allgemein. Das trainierte Modell wird mit `joblib` gespeichert und über FastAPI bereitgestellt.

**Intention:**  
Praxisnahes Beispiel für den vollständigen Lebenszyklus eines kleinen Klassifikationsmodells inklusive REST-API und Tests.

---

## `sms-spam-classifier/`

**Ort:** `sms-spam-classifier/`  
**Datendownload:** `data/download_data.py`  
**Training:** `train/train_model.py`  
**Vorhersage:** `predict/predict.py`  
**CLI:** `app.py`

**Was macht die App?**  
Lädt einen SMS-Datensatz, trainiert ein Spam/Ham-Modell mit TF-IDF und Logistic Regression, speichert das Modell und ermöglicht danach interaktive Vorhersagen inklusive Spam-Wahrscheinlichkeit.

**Intention:**  
Saubere Trennung von Datenbeschaffung, Training, Modellpersistenz und Inferenz lernen.

---

# 4. scikit-learn Lernpfad

Der Ordner `scikit-learn-Lernpfad-sandbox/` enthält mehrere bewusst kleine Lern-Apps.

## `TextklassifikationOhneLLM.py`

**Ort:** `scikit-learn-Lernpfad-sandbox/TextklassifikationOhneLLM.py`

**Kurz:** Klassische Textklassifikation mit TF-IDF und Logistic Regression auf gelabelten Beispielen.

**Intention:** Verstehen, dass Textklassifikation nicht zwingend ein LLM benötigt und wie aus Text numerische Features werden.

---

## `phase1_car_ml.py`

**Ort:** `scikit-learn-Lernpfad-sandbox/phase1_car_ml.py`

**Kurz:** Klassifiziert Autos anhand kategorialer Merkmale. Verwendet One-Hot-Encoding, Train/Test-Split und Logistic Regression.

**Intention:** Grundbegriffe wie `X`, `y`, Features, Target, Split, Encoding und Evaluation praktisch lernen.

---

## `car.py`

**Ort:** `scikit-learn-Lernpfad-sandbox/car.py`

**Kurz:** Ältere/kompaktere Variante des Auto-Klassifikationsbeispiels mit OneHotEncoder, Logistic Regression und eigener Vorhersage.

**Intention:** Einfacher Einstieg in kategoriale Daten und Modellvorhersagen.

---

## `scikit-learn-app.py`

**Ort:** `scikit-learn-Lernpfad-sandbox/scikit-learn-app.py`

**Kurz:** Arbeitet mit Support-Falldaten, behandelt fehlende Werte, trainiert Logistic Regression und wertet `classification_report` aus.

**Intention:** Typischen tabellarischen ML-Workflow mit Pandas und scikit-learn verstehen.

---

## `seeds_cross_validation.py`

**Ort:** `scikit-learn-Lernpfad-sandbox/seeds_cross_validation.py`

**Kurz:** Vergleicht unterschiedliche Random Seeds mit Cross-Validation und F1.

**Intention:** Verstehen, warum ein einzelner Train/Test-Split schwanken kann und warum Cross-Validation stabilere Aussagen ermöglicht.

---

## `GroundTruthErrorAnalysis/`

**Ort:** `scikit-learn-Lernpfad-sandbox/GroundTruthErrorAnalysis/ground_truth_error_analysis.py`

**Kurz:** Vergleicht Vorhersagen mit Ground Truth, kennzeichnet TP/TN/FP/FN und zeigt False Positives bzw. False Negatives gezielt an.

**Intention:** Nicht nur Metriken betrachten, sondern konkrete Modellfehler systematisch analysieren.

---

## `class_imbalance/`

**Ort:** `scikit-learn-Lernpfad-sandbox/class_imbalance/class_imbalance.py`

**Kurz:** Vergleicht Logistic Regression mit und ohne `class_weight='balanced'` auf einem absichtlich unausgeglichenen Datensatz.

**Intention:** Class Imbalance, Stratifikation sowie Auswirkungen auf Precision/Recall/F1 verstehen.

---

## `phase3_threshold_console/`

**Ort:** `scikit-learn-Lernpfad-sandbox/phase3_threshold_console/phase3_threshold_console.py`

**Kurz:** Nutzt `predict_proba()` und einen veränderbaren Threshold, um zu zeigen, wie sich Precision, Recall und F1 verändern.

**Intention:** Zusammenhang zwischen Confidence, Schwellenwert, False Positives und False Negatives verstehen.

---

## `retraining/`

**Ort:** `scikit-learn-Lernpfad-sandbox/retraining/retraining.py`

**Kurz:** Trainiert eine Modellversion V1, ergänzt neue Ground-Truth-Beispiele und trainiert V2 auf demselben Testset.

**Intention:** Retraining als kontrollierte Verbesserung mit neuen fachlich geprüften Beispielen nachvollziehen.

---

## `workflow-demo-app/`

**Ort:** `scikit-learn-Lernpfad-sandbox/workflow-demo-app/workflow-demo-app.py`

**Kurz:** Vollständigerer ML-Workflow mit TF-IDF, mehreren Klassifikatoren, `GridSearchCV`, Cross-Validation, Hyperparametervergleich und finaler Testset-Evaluation.

**Intention:** Den Übergang von einer einzelnen Modellidee zu einem systematischen Modellvergleich verstehen.

---

# 5. Einzelne scikit-learn Verfahren

## `sklearn_methoden_apps/`

**Ort:** `sklearn_methoden_apps/`

| Datei | Verfahren | Intention |
|---|---|---|
| `01_logistic_regression.py` | Logistic Regression | Klassifikation und Wahrscheinlichkeiten verstehen |
| `02_linear_svc.py` | Linear SVC | Margin-basierte Klassifikation kennenlernen |
| `03_decision_tree.py` | Decision Tree | Entscheidungsregeln und Baumlogik verstehen |
| `04_random_forest.py` | Random Forest | Ensemble aus vielen Bäumen kennenlernen |
| `05_knn.py` | K-Nearest Neighbors | Vorhersage über ähnliche Datenpunkte verstehen |
| `06_naive_bayes.py` | Naive Bayes | Wahrscheinlichkeitsbasierte Klassifikation kennenlernen |
| `07_gradient_boosting.py` | Gradient Boosting | Schrittweises Ensemble-Lernen verstehen |
| `08_linear_regression.py` | Linear Regression | Numerische Zielwerte vorhersagen |
| `09_kmeans.py` | K-Means | Unüberwachtes Clustering verstehen |
| `10_pca.py` | PCA | Dimensionsreduktion kennenlernen |
| `11_multiple_linear_regression.py` | Multiple Linear Regression | Regression mit mehreren Einflussgrößen |

**Gesamtintention:**  
Kleine isolierte Beispiele, um Modelltypen und typische Einsatzgebiete unabhängig voneinander kennenzulernen.

---

# 6. Thresholds, Metriken und MLflow

## `mlflow_demo/training_demo.py`

**Ort:** `mlflow_demo/training_demo.py`

**Was macht die App?**  
Trainiert Logistic Regression auf dem Iris-Datensatz, loggt Parameter und Metriken in MLflow und speichert Modell-/Grafik-Artefakte.

**Intention:**  
MLflow-Grundprinzip verstehen: Experiment → Run → Parameter → Metriken → Artefakte → Modell.

---

## `mlflow_demo/mlflow_trace_demo.py`

**Ort:** `mlflow_demo/mlflow_trace_demo.py`

**Was macht die App?**  
Verbindet ein klassisches Trainingsbeispiel mit MLflow Tracing und vergleicht mehrere `C`-Werte einer Logistic Regression.

**Intention:**  
Unterschied zwischen normalen MLflow-Runs und Traces praktisch sehen.

---

## `mlflow_demo/prompt_vergleich.py`

**Ort:** `mlflow_demo/prompt_vergleich.py`

**Was macht die App?**  
Sendet Prompts an ein OpenAI-Modell, aktiviert MLflow-Autologging und speichert Prompt, Modell und Antwort pro Run.

**Intention:**  
LLM-/Prompt-Experimente reproduzierbar vergleichen und dokumentieren.

---

## `mlflow_demo/mlflow_threshold_demo.py`

**Ort:** `mlflow_demo/mlflow_threshold_demo.py`

**Was macht die App?**  
Trainiert eine binäre Logistic Regression, testet viele Thresholds und berechnet Precision, Recall, F1 und F2. Zusätzlich wird eine gemeinsame Threshold-Grafik als Artefakt gespeichert.

**Intention:**  
Verstehen, wie sich die Wahl des Schwellenwerts auf die Klassifikationsmetriken auswirkt und wie man diese Versuche in MLflow dokumentiert.

---

## `mlflow_demo/threshold_metrics_app/`

**Ort:** `mlflow_demo/threshold_metrics_app/app.py`

**Was macht die App?**  
Anschauliche Konsolen-App mit Kundenservice-Nachrichten (`dringend` / `nicht_dringend`). Sie trainiert ein Textmodell, speichert es als `saved_model.joblib`, berechnet Precision/Recall/F1/F2 über viele Thresholds, erzeugt eine gemeinsame Grafik und erlaubt später einen **individuellen Threshold** mit den vorherigen Werten zu vergleichen.

**Intention:**  
Thresholds und Metriken nicht nur theoretisch, sondern an einer verständlichen Datengrundlage ausprobieren. Gleichzeitig wird Modellpersistenz und MLflow-Logging gezeigt.

---

## `mlflow_demo/app.py`

**Ort:** `mlflow_demo/app.py`

**Was macht die App?**  
Kleine MLflow-GenAI-Demo mit aktivem Modell, Modellparametern und Tracing einer `predict()`-Funktion.

**Intention:**  
MLflow-Funktionen für GenAI-Anwendungen und Traces kennenlernen.

---

# 7. Bias und Datenverständnis

## `bias_demo_app/`

**Ort:** `bias_demo_app/app.py`

**Was macht die App?**  
Erzeugt zwei synthetische Gruppen mit gleicher Qualifikationsverteilung, zieht bei Gruppe B systematisch Score-Punkte ab und zeigt die Auswirkungen eines festen Thresholds grafisch.

**Intention:**  
Bias sehr einfach sichtbar machen: gleiche Ausgangslage, aber systematisch unterschiedliche Modell-Scores und dadurch unterschiedliche Entscheidungen.

> Hinweis: Die vorhandene README spricht von Streamlit; die aktuelle `app.py` ist im Snapshot jedoch eine normale Python-/Matplotlib-Demo.

---

## `pandas-sandbox/`

**Ort:** `pandas-sandbox/pandas-app.py`

**Was macht die App?**  
Lädt `support_cases.csv` und demonstriert `head`, `info`, `describe`, Spaltenauswahl, fehlende Werte, Duplikate, Filtern und `value_counts`.

**Intention:**  
Pandas-Grundlagen für spätere Datenanalyse- und ML-Aufgaben auf einer greifbaren Support-Datengrundlage lernen.

---

## `numpy/`

**Ort:** `numpy/numpy-app.py`

**Was macht die App?**  
Zeigt Arrays, Matrizen, Shape, Slicing, Boolean Indexing, Vektorrechnung, Aggregationen, Reshape und Broadcasting.

**Intention:**  
NumPy-Grundlagen verstehen, die unter vielen ML-/Deep-Learning-Bibliotheken liegen.

---

# 8. Embeddings, RAG und semantische Suche

## `embedding_project_app/`

**Ort:** `embedding_project_app/app.py`

**Was macht die App?**  
Erzeugt Embeddings über Ollama, vergleicht Texte mit Cosine Similarity und gruppiert sie mit K-Means. Synthetische Ground-Truth-Labels helfen bei der Interpretation.

**Intention:**  
Semantische Textrepräsentationen, Similarity-Suche und Clustering praktisch verstehen und von klassischer Klassifikation abgrenzen.

---

## `rag-demo/`

**Ort:** `rag-demo/src/app.py`

**Was macht die App?**  
Kleine modulare RAG-Konsolenanwendung. Dokumente werden geladen, in Chunks zerlegt, über einen TF-IDF-Retriever gesucht und anschließend an ein konfigurierbares LLM (`mock`, Ollama oder LM Studio) übergeben.

**Intention:**  
RAG bewusst ohne große Infrastruktur verstehen: Dokument → Chunking → Retrieval → Kontext → LLM-Antwort.

---

## `rag-demo-py/`

**Ort:** `rag-demo-py/`  
**Web-UI:** `app.py`  
**API:** `api.py`

**Was macht die App?**  
Umfangreichere lokale RAG-Demo mit TXT/PDF/URL-Ingest, Chunking, Ollama-Embeddings, SQLite-Speicherung, Cosine Similarity, Flask-Weboberfläche und FastAPI-Endpunkten.

**Intention:**  
Einen realistischeren RAG-Prototypen mit persistierten Dokumenten, Embeddings, Modellwahl und Datenbankinspektion aufbauen.

---

# 9. MCP und Agenten

## `mcp-demo/`

**Ort:** `mcp-demo/`  
**Client:** `app.py`  
**MCP-Server:** `mcp_server.py`

**Was macht die App?**  
Ein lokales Ollama-Modell kann über MCP Ticket-Tools aufrufen, die Daten aus `tickets.json` lesen oder verändern.

**Intention:**  
Den Unterschied zwischen reinem Modellwissen und externen Tools verstehen: Das Modell entscheidet, wann ein Tool benötigt wird; MCP standardisiert den Zugriff.

---

## `multi-agent-support-demo/`

**Ort:** `multi-agent-support-demo/app.py`

**Was macht die App?**  
Support-Demo mit drei spezialisierten Agenten:

```text
TicketAgent
→ KnowledgeAgent + RAG
→ ActionAgent + MCP
```

Ollama dient als lokales LLM, `nomic-embed-text` für RAG und ein MCP-Server für Ticket-Aktionen.

**Intention:**  
Zeigen, wie eine komplexere Aufgabe in spezialisierte Agenten aufgeteilt werden kann und wie RAG und Tools zusammenspielen.

---

# 10. Transcript- und Musteranalyse

## `ollama_transcript_pattern_app/`

**Ort:** `ollama_transcript_pattern_app/`  
**API-Einstieg:** `app/main.py`

**Was macht die App?**  
Lädt Interview-/Transcript-Texte, analysiert sie mit einem lokalen Ollama-Modell und gibt definierte Muster mit Textbeleg, Erklärung und Confidence aus. Zusätzlich existieren Ground-Truth- und Evaluationskomponenten.

**Wichtige Dateien:**

- `app/main.py` – API-Einstieg
- `app/analyzer.py` – Analyseablauf
- `app/ollama_client.py` – Ollama-Zugriff
- `app/patterns.py` – Musterdefinitionen
- `app/prompts.py` – Prompting
- `app/evaluation.py` – Evaluation
- `data/ground_truth.json` – Ground Truth

**Intention:**  
Projektnahe, vollständig lokale Textanalyse mit strukturierter Ausgabe und Möglichkeit zur fachlichen Evaluation.

---

# 11. Deep Learning, Tokenisierung und Sprachmodelle

## `next-word-prediction/`

**Ort:** `next-word-prediction/next_word_prediction.py`

**Was macht die App?**  
Trainiert ein kleines LSTM auf deutschem Beispieltext und gibt für eine Eingabe die wahrscheinlichsten nächsten Wörter aus bzw. generiert mehrere Wörter automatisch.

**Intention:**  
Next-Token-/Next-Word-Prediction und sequenzielles Lernen vor Transformer-Modellen praktisch verstehen.

---

## `sliding-window/`

**Ort:** `sliding-window/SimpleSubwordTokenizer.py`

**Was macht die App?**  
Tokenisiert einen Text, erzeugt Token-IDs und baut daraus über ein Sliding Window Input-/Target-Paare für Next-Token-Prediction.

**Intention:**  
Verstehen, wie aus einem langen Text viele kleine Trainingsbeispiele für Sprachmodelle entstehen.

---

## `self-attention/`

### `app.py`

Berechnet eine vereinfachte Attention-Matrix mit kleinen künstlichen Tokenvektoren.

**Intention:** Self-Attention mathematisch und visuell nachvollziehen, ohne ein großes Modell zu benötigen.

### `bert-based.py`

Lädt `bert-base-uncased`, liest die echten Attention-Werte aus dem letzten BERT-Layer und gibt eine Attention-Matrix aus.

**Intention:** Den Übergang vom vereinfachten Attention-Prinzip zu einem realen Transformer-Modell sehen.

---

## `simple-chat-bot/`

**Ort:** `simple-chat-bot/`  
**Training:** `train.py`  
**Nutzung:** `ask.py`

**Was macht die App?**  
Fine-tuned `distilgpt2` auf einer kleinen lokalen Textdatei und speichert das Ergebnis in `trained_model/`. `ask.py` lädt dieses Modell und erzeugt Antworten auf Fragen.

**Intention:**  
Ein sehr kleines generatives Fine-Tuning mit Hugging Face Trainer und anschließender lokaler Inferenz verstehen.

---

# 12. Speech-to-Text

## `transcripter/`

**Ort:** `transcripter/main.py`  
**GPU-Test:** `transcripter/cudatest.py`

**Was macht die App?**  
Verwendet Whisper, um `voice.mp3` lokal zu transkribieren. CUDA wird verwendet, falls verfügbar. Zusätzlich werden Zeitsegmente ausgegeben und das Transkript als Datei gespeichert.

**Intention:**  
Lokale Audio-zu-Text-Verarbeitung und GPU-Nutzung mit einem fertigen Deep-Learning-Modell kennenlernen.

---

# 13. Ordner ohne vollständige eigenständige App

## `rag-demo-java/`

Im aktuellen Repository-Snapshot sind nur IDE-Dateien vorhanden. Es gibt keine Java-Quellcodes bzw. keine ausführbare RAG-Anwendung.

## `bamf_lora_demo/`

README und Modell-/Adapterdaten sind vorhanden, die im README beschriebene Haupt-App fehlt im aktuellen Snapshot.

## `bamf_lora_pycharm/`

Im Wesentlichen Dokumentation; keine vollständige App im Snapshot.

## `bamf_lora_simple_auto/`

Enthält überwiegend Modellordner; keine vollständige App im Snapshot.

## `.idea/` und `bin/`

Keine Projekt-Apps. `.idea/` enthält PyCharm-Projektmetadaten. `bin/` enthält IDE-/Tool-Binaries und sollte nicht als Lern-App betrachtet werden.

---

# 14. Lernunterlagen im Repository

Zusätzlich liegen mehrere Lernunterlagen im Root bzw. in Unterordnern:

- `ML.pdf`
- `ML_Projektvorbereitung_Lernfragen.pdf`
- `ML_Projektvorbereitung_Lernfragen.docx`
- `LLM_Lernskript_ASM_LoRA.pdf`
- `numpy/NumPy_Lerneinheit.pdf`
- `sklearn_methoden_apps/ML_Modelle_Anwendungsfaelle_Lernuebersicht.pdf`
- `sklearn_methoden_apps/ML_Modelle_Anwendungsfaelle_Lernuebersicht.docx`

Diese sind **Dokumentation/Lernmaterial**, keine ausführbaren Apps.

---

# 15. Empfohlene Lernreihenfolge

Für einen übersichtlichen Lernpfad durch das Repository bietet sich folgende Reihenfolge an:

```text
1. numpy/
2. pandas-sandbox/
3. scikit-learn-Lernpfad-sandbox/
4. sklearn_methoden_apps/
5. ticket-classification/ bzw. bamf_ml_preparation/
6. mlflow_demo/
7. embedding_project_app/
8. rag-demo/ → rag-demo-py/
9. encoder_example/
10. sliding-window/ → self-attention/ → next-word-prediction/
11. simple-chat-bot/
12. bamf_lora_clean/
13. mcp-demo/
14. multi-agent-support-demo/
15. ollama_transcript_pattern_app/
```

Damit geht der Lernpfad grob von **Datenverarbeitung → klassischem ML → Evaluation → Embeddings/RAG → Transformer/LLM → Fine-Tuning → Tools/Agenten**.

---

# 16. Repository-Schwerpunkte auf einen Blick

Das Repository deckt inzwischen vier große Lernbereiche ab:

1. **Data Science / klassisches ML**  
   NumPy, Pandas, scikit-learn, Ground Truth, Error Analysis, Class Imbalance, Cross-Validation, Thresholds, Retraining und Modellvergleich.

2. **NLP / Deep Learning / LLMs**  
   Tokenisierung, Sliding Windows, Attention, Encoder, LSTM, generative Modelle und LoRA.

3. **AI Engineering**  
   FastAPI, Modellpersistenz, MLflow, Ollama, lokale Inferenz und Modellintegration.

4. **Moderne AI-Architekturen**  
   Embeddings, RAG, MCP, Multi-Agent-Systeme und transcriptbasierte Musteranalyse.
