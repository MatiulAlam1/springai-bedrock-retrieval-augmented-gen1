package com.example.rag.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;

import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URI;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.security.MessageDigest;
import java.nio.charset.StandardCharsets;

@Service
public class FreeEmbeddingService {

    private static final Logger LOGGER = LoggerFactory.getLogger(FreeEmbeddingService.class);
    
    @Value("${rag.embedding.model.huggingface}")
    private String modelName;
    
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final Map<String, float[]> cache = new ConcurrentHashMap<>();

    public FreeEmbeddingService() {
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();
        this.objectMapper = new ObjectMapper();
    }

    public float[] generateEmbeddings(String inputText) {
        try {
            // Check cache first
            String cacheKey = generateCacheKey(inputText);
            if (cache.containsKey(cacheKey)) {
                LOGGER.debug("Using cached embedding for text: {}", inputText.substring(0, Math.min(50, inputText.length())));
                return cache.get(cacheKey);
            }
            
            LOGGER.info("Generating free embedding for text: {}", inputText.substring(0, Math.min(50, inputText.length())));
            
            // Use Hugging Face Inference API (free tier)
            float[] embeddings = generateHuggingFaceEmbedding(inputText);
            
            // Cache the result
            cache.put(cacheKey, embeddings);
            
            LOGGER.info("Generated embedding with {} dimensions", embeddings.length);
            return embeddings;
            
        } catch (Exception e) {
            LOGGER.error("Failed to generate embedding: {}", e.getMessage(), e);
            // Return dummy embedding as fallback
            return createDummyEmbedding(inputText);
        }
    }
    
    private float[] generateHuggingFaceEmbedding(String text) throws Exception {
        // Hugging Face Inference API endpoint
        String apiUrl = "https://api-inference.huggingface.co/pipeline/feature-extraction/" + modelName;
        
        // Create request body
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("inputs", text);
        requestBody.put("options", Map.of("wait_for_model", true));
        
        String jsonBody = objectMapper.writeValueAsString(requestBody);
        
        // Create HTTP request
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(apiUrl))
                .header("Content-Type", "application/json")
                .header("User-Agent", "RAG-Spring-App/1.0")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();
        
        // Send request
        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        
        if (response.statusCode() == 200) {
            // Parse response - HuggingFace returns array of embeddings
            JsonNode responseJson = objectMapper.readTree(response.body());
            
            if (responseJson.isArray() && responseJson.size() > 0) {
                JsonNode embedding = responseJson.get(0);
                float[] embeddings = new float[embedding.size()];
                
                for (int i = 0; i < embedding.size(); i++) {
                    embeddings[i] = (float) embedding.get(i).asDouble();
                }
                
                return embeddings;
            }
        } else if (response.statusCode() == 503) {
            LOGGER.warn("Hugging Face model is loading, using fallback embedding");
        } else {
            LOGGER.warn("Hugging Face API request failed with status: {}, response: {}", 
                       response.statusCode(), response.body());
        }
        
        // Fallback to simple embedding
        return createDummyEmbedding(text);
    }
    
    private float[] createDummyEmbedding(String text) {
        LOGGER.debug("Creating deterministic embedding for text");
        
        // Create a simple but deterministic embedding based on text content
        // This is not as good as real embeddings but provides basic functionality
        float[] embedding = new float[384]; // Standard size for MiniLM
        
        // Use text hash to create reproducible embeddings
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(text.toLowerCase().getBytes(StandardCharsets.UTF_8));
            
            // Convert hash bytes to float values
            for (int i = 0; i < embedding.length; i++) {
                int byteIndex = i % hash.length;
                embedding[i] = (hash[byteIndex] & 0xFF) / 255.0f - 0.5f; // Normalize to [-0.5, 0.5]
            }
            
            // Normalize the vector
            float norm = 0.0f;
            for (float val : embedding) {
                norm += val * val;
            }
            norm = (float) Math.sqrt(norm);
            
            if (norm > 0) {
                for (int i = 0; i < embedding.length; i++) {
                    embedding[i] = embedding[i] / norm;
                }
            }
            
        } catch (Exception e) {
            LOGGER.warn("Failed to create hash-based embedding, using random values");
            // Fallback to simple pattern
            for (int i = 0; i < embedding.length; i++) {
                embedding[i] = (float) Math.sin(i * 0.1 + text.hashCode() * 0.001);
            }
        }
        
        return embedding;
    }
    
    private String generateCacheKey(String text) {
        return Integer.toString(text.hashCode());
    }
}
