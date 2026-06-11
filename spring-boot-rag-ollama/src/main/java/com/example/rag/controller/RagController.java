package com.example.rag.controller;

import com.example.rag.dto.OllamaDtos.ModelInfo;
import com.example.rag.service.OllamaService;
import com.example.rag.service.RagService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@Controller
public class RagController {

    private final RagService ragService;
    private final OllamaService ollamaService;

    public RagController(RagService ragService, OllamaService ollamaService) {
        this.ragService = ragService;
        this.ollamaService = ollamaService;
    }

    @GetMapping("/")
    public String index(Model model) {
        addCommonData(model);
        return "index";
    }

    @PostMapping("/upload")
    public String upload(@RequestParam("file") MultipartFile file, Model model) {
        try {
            RagService.IngestResult result = ragService.ingest(file);
            model.addAttribute("success", "Dokument wurde indexiert: " + result.documentName() + " (" + result.chunkCount() + " Chunks).");
            model.addAttribute("selectedDocument", result.documentName());
        } catch (Exception e) {
            model.addAttribute("error", e.getMessage());
        }

        addCommonData(model);
        return "index";
    }

    @PostMapping("/ask")
    public String ask(
            @RequestParam("documentName") String documentName,
            @RequestParam("question") String question,
            Model model
    ) {
        try {
            RagService.AskResult result = ragService.ask(documentName, question);
            model.addAttribute("answer", result.answer());
            model.addAttribute("usedChunks", result.chunks());
            model.addAttribute("selectedDocument", documentName);
            model.addAttribute("question", question);
        } catch (Exception e) {
            model.addAttribute("error", e.getMessage());
        }

        addCommonData(model);
        return "index";
    }

    private void addCommonData(Model model) {
        model.addAttribute("documents", ragService.documentNames());

        try {
            List<ModelInfo> models = ollamaService.getAvailableModels();
            model.addAttribute("models", models);
            model.addAttribute("chatModel", ollamaService.detectChatModelName());
            model.addAttribute("embeddingModel", ollamaService.detectEmbeddingModelName());
        } catch (Exception e) {
            model.addAttribute("modelError", e.getMessage());
        }
    }
}
