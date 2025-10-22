package com.example.rag.service;

import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.DataType;
import io.milvus.grpc.SearchResults;
import io.milvus.param.ConnectParam;
import io.milvus.param.IndexType;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.collection.*;
import io.milvus.param.dml.*;
import io.milvus.param.index.*;
import io.milvus.response.SearchResultsWrapper;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
public class VectorStoreService {

    private static final Logger LOGGER = LoggerFactory.getLogger(VectorStoreService.class);

    @Value("${spring.ai.vectorstore.milvus.host}")
    private String milvusHost;

    @Value("${spring.ai.vectorstore.milvus.port}")
    private Integer milvusPort;

    @Value("${spring.ai.vectorstore.milvus.database-name}")
    private String databaseName;

    @Value("${spring.ai.vectorstore.milvus.collection-name}")
    private String collectionName;

    @Value("${spring.ai.vectorstore.milvus.embedding-dimension}")
    private Integer embeddingDimension;

    private MilvusServiceClient milvusClient;

    @PostConstruct
    public void init() {
        try {
            LOGGER.info("Connecting to Milvus at {}:{}", milvusHost, milvusPort);
            milvusClient = new MilvusServiceClient(
                    ConnectParam.newBuilder()
                            .withHost(milvusHost)
                            .withPort(milvusPort)
                            .build()
            );

            createCollectionIfNotExists();
            LOGGER.info("Milvus initialized successfully. Collection '{}'", collectionName);
        } catch (Exception e) {
            LOGGER.error("Failed to initialize Milvus: {}", e.getMessage(), e);
        }
    }

    private void createCollectionIfNotExists() {
        R<Boolean> hasCollection = milvusClient.hasCollection(
                HasCollectionParam.newBuilder()
                        .withCollectionName(collectionName)
                        .build()
        );

        if (hasCollection.getData()) {
            LOGGER.info("Collection '{}' already exists", collectionName);
            return;
        }

        // Define fields
        FieldType idField = FieldType.newBuilder()
                .withName("id")
                .withDataType(DataType.VarChar)
                .withMaxLength(64)
                .withPrimaryKey(true)
                .build();

        FieldType vectorField = FieldType.newBuilder()
                .withName("embedding")
                .withDataType(DataType.FloatVector)
                .withDimension(embeddingDimension)
                .build();

        FieldType contentField = FieldType.newBuilder()
                .withName("content")
                .withDataType(DataType.VarChar)
                .withMaxLength(2048)
                .build();

        CreateCollectionParam createParam = CreateCollectionParam.newBuilder()
                .withCollectionName(collectionName)
                .addFieldType(idField)
                .addFieldType(vectorField)
                .addFieldType(contentField)
                .build();

        milvusClient.createCollection(createParam);

        // Create index
        CreateIndexParam indexParam = CreateIndexParam.newBuilder()
                .withCollectionName(collectionName)
                .withFieldName("embedding")
                .withIndexType(IndexType.IVF_FLAT)
                .withMetricType(MetricType.COSINE)
                .withSyncMode(true)
                .build();
        milvusClient.createIndex(indexParam);

        LOGGER.info("Created collection '{}' with IVF_FLAT index and COSINE metric.", collectionName);
    }

    public void saveEmbedding(String id, List<Float> embedding, String content) {
        try {
            List<InsertParam.Field> fields = List.of(
                    new InsertParam.Field("id", List.of(id)),
                    new InsertParam.Field("embedding", List.of(embedding)),
                    new InsertParam.Field("content", List.of(content))
            );

            InsertParam insertParam = InsertParam.newBuilder()
                    .withCollectionName(collectionName)
                    .withFields(fields)
                    .build();

            milvusClient.insert(insertParam);
            LOGGER.info("Inserted embedding for document '{}'", id);
        } catch (Exception e) {
            LOGGER.error("Failed to insert embedding: {}", e.getMessage(), e);
        }
    }

    public List<String> querySimilarEmbeddings(float[] queryEmbedding) {
        try {
            SearchParam searchParam = SearchParam.newBuilder()
                    .withCollectionName(collectionName)
                    .withMetricType(MetricType.COSINE)
                    .withTopK(5)
                    .withVectors(Collections.singletonList(queryEmbedding))
                    .addOutField("content")
                    .withParams("{\"nprobe\":10}")
                    .build();

            R<SearchResults> searchResults = milvusClient.search(searchParam);
            SearchResultsWrapper wrapper = new SearchResultsWrapper(searchResults.getData().getResults());

            List<SearchResultsWrapper.IDScore> scores = wrapper.getIDScore(0);
            List<String> results = new ArrayList<>();

            for (SearchResultsWrapper.IDScore score : scores) {
                results.add("ID: " + score.getStrID () + " (similarity: " +
                        String.format("%.3f", score.getScore()) + ")");
            }

            LOGGER.info("Found {} similar embeddings.", results.size());
            return results;

        } catch (Exception e) {
            LOGGER.error("Search failed: {}", e.getMessage(), e);
            return List.of("Milvus search error: " + e.getMessage());
        }
    }
}
