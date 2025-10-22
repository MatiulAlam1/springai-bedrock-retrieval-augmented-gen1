package com.example.rag.service;

import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

@Service
public class EmbeddingService {

    private static final Logger LOGGER = LoggerFactory.getLogger(EmbeddingService.class);

    private final EmbeddingModel embeddingModel;  // Spring AI auto-configured bean
    private final FreeEmbeddingService freeEmbeddingService;

    @Value("${rag.embedding.provider}")
    private String embeddingProvider;

    @Value("${rag.embedding.dimension:1536}")
    private int embeddingDimension;

    @Autowired
    public EmbeddingService(
            EmbeddingModel embeddingModel,
            FreeEmbeddingService freeEmbeddingService) {
        this.embeddingModel = embeddingModel;
        this.freeEmbeddingService = freeEmbeddingService;
    }

    public float[] generateEmbeddings(String inputText) {
        try {
            LOGGER.info("Generating embeddings using provider: {}", embeddingProvider);

            if ("huggingface".equalsIgnoreCase(embeddingProvider)) {
                float[] embeddings = freeEmbeddingService.generateEmbeddings(inputText);
                validateDimensions(embeddings);
                return embeddings;
            } else {
                // Use Spring AI's EmbeddingModel abstraction
                EmbeddingResponse response = embeddingModel.embedForResponse(List.of(inputText));

                // Extract the embedding from the response

                float[] embeddings = response.getResults().get(0).getOutput();

                validateDimensions(embeddings);
                LOGGER.info("Generated Bedrock Titan embedding with {} dimensions", embeddings.length);
                return embeddings;
            }

        } catch (Exception e) {
            LOGGER.error("Failed to generate embeddings: {}", e.getMessage(), e);

            // Fallback to free service if not already using it
            if (!"huggingface".equalsIgnoreCase(embeddingProvider)) {
                LOGGER.warn("Falling back to free embedding service due to error");
                try {
                    float[] fallbackEmbeddings = freeEmbeddingService.generateEmbeddings(inputText);
                    validateDimensions(fallbackEmbeddings);
                    return fallbackEmbeddings;
                } catch (Exception fallbackException) {
                    LOGGER.error("Fallback embedding also failed: {}", fallbackException.getMessage());
                    throw new RuntimeException("Failed to generate embeddings with both primary and fallback services", e);
                }
            }

            throw new RuntimeException("Failed to generate embeddings", e);
        }
    }

    private void validateDimensions(float[] embeddings) {
        if (embeddings.length != embeddingDimension) {
            LOGGER.warn("Expected {} dimensions but got {}. Consider updating rag.embedding.dimension property.",
                    embeddingDimension, embeddings.length);
        }
    }

    public int getEmbeddingDimension() {
        return embeddingDimension;
    }
}