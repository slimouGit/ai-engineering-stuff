package com.example.rag.repository;

import com.example.rag.entity.DocumentChunk;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface DocumentChunkRepository extends JpaRepository<DocumentChunk, Long> {

    List<DocumentChunk> findByDocumentNameOrderByChunkIndexAsc(String documentName);

    void deleteByDocumentName(String documentName);

    @Query("select distinct c.documentName from DocumentChunk c order by c.documentName")
    List<String> findDistinctDocumentNames();
}
