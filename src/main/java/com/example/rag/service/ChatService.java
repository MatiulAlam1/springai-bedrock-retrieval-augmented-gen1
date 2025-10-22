package com.example.rag.service;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.apache.poi.xwpf.extractor.XWPFWordExtractor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

@Service
public class ChatService {

    private static final Logger LOGGER = LoggerFactory.getLogger(ChatService.class);

    private final EmbeddingService embeddingService;
    private final VectorStoreService vectorStoreService;
    private final ChatClient chatClient;

    @Autowired
    public ChatService(EmbeddingService embeddingService,
                       VectorStoreService vectorStoreService,
                       ChatModel chatModel) {
        this.embeddingService = embeddingService;
        this.vectorStoreService = vectorStoreService;
        this.chatClient = ChatClient.builder(chatModel).build();
    }

    public void indexDocument(String document) {
        LOGGER.debug("Indexing document with length: {}", document.length());

        // Generate embeddings for the document
        float[] embeddings = embeddingService.generateEmbeddings(document);

        // Store the document embeddings in the vector store with a unique ID
        String documentId = "doc_" + System.currentTimeMillis();
        List<Float> embeddingList = new ArrayList<>();
        for (float f : embeddings) {
            embeddingList.add(f);
        }

        // Store both embeddings and document content
        vectorStoreService.saveEmbedding(documentId, embeddingList, document);
        LOGGER.info("Successfully indexed document with ID: {}", documentId);
    }

    public void indexDocument(MultipartFile file) throws IOException {
        String content = extractTextFromFile(file);
        indexDocumentFromString(content);
    }

    private void indexDocumentFromString(String document) {
        // Generate embeddings for the document
        float[] embeddings = embeddingService.generateEmbeddings(document);

        // Store the document embeddings in the vector store with a unique ID
        String documentId = "doc_" + System.currentTimeMillis();
        List<Float> embeddingList = new ArrayList<>();
        for (float f : embeddings) {
            embeddingList.add(f);
        }

        // Store both embeddings and document content
        vectorStoreService.saveEmbedding(documentId, embeddingList, document);
    }

    private String extractTextFromFile(MultipartFile file) throws IOException {
        String filename = file.getOriginalFilename();
        if (filename == null || filename.isEmpty()) {
            throw new IllegalArgumentException("File name is required");
        }

        String extension = filename.substring(filename.lastIndexOf('.') + 1).toLowerCase();

        switch (extension) {
            case "txt":
                return new String(file.getBytes(), StandardCharsets.UTF_8);
            case "pdf":
                try (PDDocument document = PDDocument.load(file.getInputStream())) {
                    PDFTextStripper stripper = new PDFTextStripper();
                    return stripper.getText(document);
                } catch (IOException e) {
                    throw new IOException("Failed to extract text from PDF file", e);
                }
            case "docx":
                try (XWPFDocument document = new XWPFDocument(file.getInputStream());
                     XWPFWordExtractor extractor = new XWPFWordExtractor(document)) {
                    return extractor.getText();
                } catch (IOException e) {
                    throw new IOException("Failed to extract text from DOCX file", e);
                }
            case "doc":
                throw new UnsupportedOperationException("Legacy .doc format not supported. Please use .docx format.");
            default:
                throw new UnsupportedOperationException("Unsupported file type: " + extension);
        }
    }

    public String queryAndChat(String query) {
        LOGGER.info("Processing query: {}", query);

        try {
            // Generate embeddings for the query
            float[] queryEmbeddings = embeddingService.generateEmbeddings(query);

            // Query the vector store for similar documents
            List<String> similarDocuments = vectorStoreService.querySimilarEmbeddings(queryEmbeddings);

            // Use the retrieved context to generate a response
            String context = buildContextFromSimilarDocuments(similarDocuments);

            if (context.isEmpty()) {
                LOGGER.warn("No relevant context found for query");
                return "I don't have enough information to answer that question. Please upload relevant documents first.";
            }

            String enhancedPrompt = buildEnhancedPrompt(context, query);

            // Generate response using Spring AI ChatClient
            String response = chatClient.prompt()
                    .user(enhancedPrompt)
                    .call()
                    .content();

            LOGGER.info("Successfully generated response");
            return response;

        } catch (Exception e) {
            LOGGER.error("Error generating response: {}", e.getMessage(), e);
            return "I encountered an error while processing your question. Please try again.";
        }
    }

    private String buildContextFromSimilarDocuments(List<String> similarDocuments) {
        if (similarDocuments == null || similarDocuments.isEmpty()) {
            return "";
        }
        return String.join("\n\n---\n\n", similarDocuments);
    }

    private String buildEnhancedPrompt(String context, String query) {
        return String.format(
                "You are a helpful assistant. Use the following context to answer the user's question. " +
                        "If the answer cannot be found in the context, say so.\n\n" +
                        "Context:\n%s\n\n" +
                        "Question: %s\n\n" +
                        "Answer:",
                context, query
        );
    }
}