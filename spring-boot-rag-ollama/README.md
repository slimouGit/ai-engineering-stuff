# Spring Boot RAG Ollama

Lokale RAG-Anwendung mit:

- Spring Boot
- Thymeleaf
- H2
- JPA
- PDFBox
- Ollama

## Voraussetzungen

- Java 21
- Maven
- Ollama

## Ollama-Modelle installieren

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Alternativ kann auch ein anderes Embedding-Modell genutzt werden, wenn der Modellname `embed`, `embedding`, `minilm` oder `bge` enthält.

## Start

```bash
ollama serve
mvn spring-boot:run
```

Danach:

```text
http://localhost:8080
```

## H2-Konsole

```text
http://localhost:8080/h2-console
```

JDBC URL:

```text
jdbc:h2:file:./data/ragdb
```

Benutzer:

```text
sa
```

Passwort leer lassen.
