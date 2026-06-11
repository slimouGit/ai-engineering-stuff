package com.example.rag.service;

import com.example.rag.dto.OllamaDtos.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.util.Comparator;
import java.util.List;
import java.util.Locale;

@Service
public class OllamaService {

    private final RestClient restClient;

    public OllamaService(@Value("${ollama.base-url}") String ollamaBaseUrl) {
        this.restClient = RestClient.builder()
                .baseUrl(ollamaBaseUrl)
                .build();
    }

    public List<ModelInfo> getAvailableModels() {
        TagsResponse response = restClient.get()
                .uri("/api/tags")
                .retrieve()
                .body(TagsResponse.class);

        if (response == null || response.models() == null) {
            return List.of();
        }

        return response.models();
    }

    public String detectChatModelName() {
        return getAvailableModels().stream()
                .filter(model -> !isEmbeddingModel(modelName(model)))
                .sorted(Comparator.comparing(ModelInfo::modified_at, Comparator.nullsLast(Comparator.reverseOrder())))
                .map(this::modelName)
                .filter(name -> name != null && !name.isBlank())
                .findFirst()
                .orElseThrow(() -> new IllegalStateException("Kein Chat-Modell in Ollama gefunden."));
    }

    public String detectEmbeddingModelName() {
        return getAvailableModels().stream()
                .filter(model -> isEmbeddingModel(modelName(model)))
                .map(this::modelName)
                .filter(name -> name != null && !name.isBlank())
                .findFirst()
                .orElseThrow(() -> new IllegalStateException(
                        "Kein Embedding-Modell in Ollama gefunden. Bitte z. B. `ollama pull nomic-embed-text` oder `ollama pull embeddinggemma` ausführen."
                ));
    }

    public List<Double> embed(String text) {
        String embeddingModel = detectEmbeddingModelName();
        EmbedRequest request = new EmbedRequest(embeddingModel, text);

        EmbedResponse response = restClient.post()
                .uri("/api/embed")
                .body(request)
                .retrieve()
                .body(EmbedResponse.class);

        if (response == null || response.embeddings() == null || response.embeddings().isEmpty()) {
            throw new IllegalStateException("Ollama hat keine Embeddings zurückgegeben.");
        }

        return response.embeddings().get(0);
    }

    public String generateAnswer(String context, String question) {
        String chatModel = detectChatModelName();

        String prompt = """
                Du bist ein lokaler RAG-Assistent.
                Beantworte die Frage ausschließlich anhand des bereitgestellten Kontexts.
                Antworte auf Deutsch.
                Wenn die Antwort nicht im Kontext steht, antworte exakt:
                Diese Information steht nicht im Dokument.

                KONTEXT:
                %s

                FRAGE:
                %s

                ANTWORT:
                """.formatted(context, question);

        GenerateRequest request = new GenerateRequest(chatModel, prompt, false);

        GenerateResponse response = restClient.post()
                .uri("/api/generate")
                .body(request)
                .retrieve()
                .body(GenerateResponse.class);

        if (response == null || response.response() == null) {
            throw new IllegalStateException("Ollama hat keine Antwort zurückgegeben.");
        }

        return response.response().trim();
    }

    private boolean isEmbeddingModel(String name) {
        if (name == null) {
            return false;
        }

        String lower = name.toLowerCase(Locale.ROOT);
        return lower.contains("embed")
                || lower.contains("embedding")
                || lower.contains("minilm")
                || lower.contains("bge");
    }

    private String modelName(ModelInfo model) {
        if (model == null) {
            return null;
        }
        if (model.name() != null && !model.name().isBlank()) {
            return model.name();
        }
        return model.model();
    }
}
