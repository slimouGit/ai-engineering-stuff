package com.example.rag.service;

import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
public class TextChunker {

    private static final int DEFAULT_CHUNK_SIZE = 1200;
    private static final int DEFAULT_OVERLAP = 200;

    public List<String> split(String text) {
        return split(text, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP);
    }

    public List<String> split(String text, int chunkSize, int overlap) {
        String cleaned = clean(text);

        if (cleaned.isBlank()) {
            throw new IllegalArgumentException("Das Dokument enthält keinen lesbaren Text.");
        }

        List<String> chunks = new ArrayList<>();
        int start = 0;

        while (start < cleaned.length()) {
            int end = Math.min(start + chunkSize, cleaned.length());
            int adjustedEnd = adjustEndToWordBoundary(cleaned, start, end);

            chunks.add(cleaned.substring(start, adjustedEnd).trim());

            if (adjustedEnd >= cleaned.length()) {
                break;
            }

            start = Math.max(0, adjustedEnd - overlap);
        }

        return chunks.stream()
                .filter(chunk -> !chunk.isBlank())
                .toList();
    }

    private String clean(String text) {
        if (text == null) {
            return "";
        }

        return text
                .replace("\r", "")
                .replaceAll("[ \t]+", " ")
                .replaceAll("\n{3,}", "\n\n")
                .trim();
    }

    private int adjustEndToWordBoundary(String text, int start, int proposedEnd) {
        if (proposedEnd >= text.length()) {
            return text.length();
        }

        int lastSpace = text.lastIndexOf(' ', proposedEnd);
        if (lastSpace <= start + 300) {
            return proposedEnd;
        }

        return lastSpace;
    }
}
