package com.example.rag.dto;

import java.util.List;

public final class OllamaDtos {

    private OllamaDtos() {
    }

    public record TagsResponse(List<ModelInfo> models) {
    }

    public record ModelInfo(
            String name,
            String model,
            String modified_at,
            Long size
    ) {
    }

    public record GenerateRequest(
            String model,
            String prompt,
            boolean stream
    ) {
    }

    public record GenerateResponse(
            String model,
            String response,
            boolean done
    ) {
    }

    public record EmbedRequest(
            String model,
            String input
    ) {
    }

    public record EmbedResponse(
            String model,
            List<List<Double>> embeddings
    ) {
    }
}
