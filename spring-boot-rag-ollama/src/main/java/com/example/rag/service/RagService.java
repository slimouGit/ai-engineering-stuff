package com.example.rag.service;

import com.example.rag.entity.DocumentChunk;
import com.example.rag.repository.DocumentChunkRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.util.Comparator;
import java.util.List;

@Service
public class RagService {

    private static final int TOP_K = 4;

    private final DocumentTextExtractor documentTextExtractor;
    private final TextChunker textChunker;
    private final OllamaService ollamaService;
    private final DocumentChunkRepository documentChunkRepository;
    private final ObjectMapper objectMapper;
    private final VectorMath vectorMath;

    public RagService(
            DocumentTextExtractor documentTextExtractor,
            TextChunker textChunker,
            OllamaService ollamaService,
            DocumentChunkRepository documentChunkRepository,
            ObjectMapper objectMapper,
            VectorMath vectorMath
    ) {
        this.documentTextExtractor = documentTextExtractor;
        this.textChunker = textChunker;
        this.ollamaService = ollamaService;
        this.documentChunkRepository = documentChunkRepository;
        this.objectMapper = objectMapper;
        this.vectorMath = vectorMath;
    }

    @Transactional
    public IngestResult ingest(MultipartFile file) {
        String documentName = sanitizeDocumentName(file.getOriginalFilename());
        String documentText = documentTextExtractor.extract(file);
        List<String> chunks = textChunker.split(documentText);

        documentChunkRepository.deleteByDocumentName(documentName);

        for (int i = 0; i < chunks.size(); i++) {
            String chunk = chunks.get(i);
            List<Double> embedding = ollamaService.embed(chunk);

            DocumentChunk entity = new DocumentChunk();
            entity.setDocumentName(documentName);
            entity.setChunkIndex(i);
            entity.setContent(chunk);
            entity.setEmbeddingJson(toJson(embedding));

            documentChunkRepository.save(entity);
        }

        return new IngestResult(documentName, chunks.size());
    }

    @Transactional(readOnly = true)
    public AskResult ask(String documentName, String question) {
        if (documentName == null || documentName.isBlank()) {
            throw new IllegalArgumentException("Bitte ein Dokument auswählen.");
        }
        if (question == null || question.isBlank()) {
            throw new IllegalArgumentException("Bitte eine Frage eingeben.");
        }

        List<DocumentChunk> chunks = documentChunkRepository.findByDocumentNameOrderByChunkIndexAsc(documentName);
        if (chunks.isEmpty()) {
            throw new IllegalArgumentException("Das ausgewählte Dokument wurde nicht gefunden.");
        }

        List<Double> questionEmbedding = ollamaService.embed(question);

        List<ScoredChunk> topChunks = chunks.stream()
                .map(chunk -> new ScoredChunk(
                        chunk.getChunkIndex(),
                        chunk.getContent(),
                        vectorMath.cosineSimilarity(questionEmbedding, fromJson(chunk.getEmbeddingJson()))
                ))
                .sorted(Comparator.comparingDouble(ScoredChunk::score).reversed())
                .limit(TOP_K)
                .toList();

        String context = topChunks.stream()
                .map(chunk -> "[Chunk " + chunk.index() + ", Score " + String.format("%.4f", chunk.score()) + "]\n" + chunk.content())
                .reduce((a, b) -> a + "\n\n---\n\n" + b)
                .orElse("");

        String answer = ollamaService.generateAnswer(context, question);

        return new AskResult(answer, topChunks);
    }

    public List<String> documentNames() {
        return documentChunkRepository.findDistinctDocumentNames();
    }

    private String sanitizeDocumentName(String originalName) {
        if (originalName == null || originalName.isBlank()) {
            return "unbenanntes-dokument";
        }
        return originalName.replaceAll("[^a-zA-Z0-9äöüÄÖÜß._ -]", "_").trim();
    }

    private String toJson(List<Double> embedding) {
        try {
            return objectMapper.writeValueAsString(embedding);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Embedding konnte nicht gespeichert werden.", e);
        }
    }

    private List<Double> fromJson(String json) {
        try {
            return objectMapper.readValue(
                    json,
                    objectMapper.getTypeFactory().constructCollectionType(List.class, Double.class)
            );
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Embedding konnte nicht gelesen werden.", e);
        }
    }

    public record IngestResult(String documentName, int chunkCount) {
    }

    public record AskResult(String answer, List<ScoredChunk> chunks) {
    }

    public record ScoredChunk(int index, String content, double score) {
    }
}
