package com.example.rag.controller;

import com.example.rag.model.ChatRequest;
import com.example.rag.service.ChatService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RestController
@RequestMapping("/api/chat")
public class ChatController {

    private static final Logger LOGGER = LoggerFactory.getLogger(ChatController.class);
    private final ChatService chatService;

    @Autowired
    public ChatController(ChatService chatService) {
        this.chatService = chatService;
    }

//    @PostMapping
//    public ResponseEntity<String> sendMessage(@RequestBody ChatRequest chatRequest) {
//        LOGGER.info("Received chat message: {}", chatRequest.getMessage());
//        String response = chatService.processChatMessage(chatRequest);
//        LOGGER.info("Chat response: {}", response);
//        return ResponseEntity.ok(response);
//    }

    // Endpoint to index a custom document
    @PostMapping("/index")
    public ResponseEntity<String> indexDocument(@RequestParam("file") MultipartFile file) {
        try {
            LOGGER.info("Indexing document: {}", file.getOriginalFilename());
            chatService.indexDocument(file);
            LOGGER.info("Document indexed successfully: {}", file.getOriginalFilename());
            return ResponseEntity.ok("Document indexed successfully");
        } catch (Exception e) {
            LOGGER.error("Error indexing document: {}", e.getMessage(), e);
            return ResponseEntity.badRequest().body("Error indexing document: " + e.getMessage());
        }
    }

    // Endpoint to query the vector DB and send request to Bedrock
    @PostMapping("/query")
    public ResponseEntity<String> queryVectorDb(@RequestBody String query) {
        LOGGER.info("Received query request: {}", query);
        String response = chatService.queryAndChat(query);
        LOGGER.info("Generated response: {}", response);
        return ResponseEntity.ok(response);
    }
}