package com.example.rag.service;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.nio.charset.StandardCharsets;
import java.util.Locale;

@Service
public class DocumentTextExtractor {

    public String extract(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("Bitte eine TXT- oder PDF-Datei hochladen.");
        }

        String filename = file.getOriginalFilename();
        if (filename == null || filename.isBlank()) {
            throw new IllegalArgumentException("Der Dateiname fehlt.");
        }

        String lowerFilename = filename.toLowerCase(Locale.ROOT);

        try {
            if (lowerFilename.endsWith(".txt")) {
                return new String(file.getBytes(), StandardCharsets.UTF_8);
            }

            if (lowerFilename.endsWith(".pdf")) {
                try (var pdf = Loader.loadPDF(file.getBytes())) {
                    return new PDFTextStripper().getText(pdf);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Die Datei konnte nicht gelesen werden: " + e.getMessage(), e);
        }

        throw new IllegalArgumentException("Nicht unterstütztes Dateiformat. Erlaubt sind .txt und .pdf.");
    }
}
