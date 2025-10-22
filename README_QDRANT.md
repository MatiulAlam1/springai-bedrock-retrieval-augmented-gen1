# RAG Spring AI Bedrock Application

A comprehensive Retrieval-Augmented Generation (RAG) application built with Spring Boot, AWS Bedrock, and Qdrant vector database.

## 🚀 Features

- **Document Processing**: Support for PDF, Word (.docx), and plain text files
- **Vector Storage**: Qdrant vector database integration for persistent storage
- **AWS Bedrock Integration**: Claude 3.5 Sonnet for responses, Amazon Titan for embeddings
- **RESTful APIs**: Upload documents for indexing and query the knowledge base
- **Similarity Search**: Cosine similarity-based document retrieval

## 🏗️ Architecture

```
Document Upload → Text Extraction → Embedding Generation → Vector Storage (Qdrant)
                                                                    ↓
User Query → Embedding Generation → Similarity Search → Context Retrieval → LLM Response
```

## 📋 Prerequisites

- Java 17+
- Maven 3.6+
- Docker & Docker Compose
- AWS Account with Bedrock access

## 🛠️ Setup Instructions

### 1. Start Qdrant Vector Database

```bash
# Start Qdrant container
docker-compose up -d

# Verify Qdrant is running
curl http://localhost:6333/health
```

### 2. Configure AWS Credentials

Set up your AWS credentials for Bedrock access:

```bash
# Option 1: Environment Variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1

# Option 2: AWS CLI
aws configure
```

### 3. Configure Application

Update `src/main/resources/application.properties`:

```properties
# AWS Bedrock Configuration
aws.bedrock.region=us-east-1
aws.bedrock.bearer-token=your_bearer_token_if_needed

# Model Configuration
aws.bedrock.claude.model=anthropic.claude-3-5-sonnet-20241022-v2:0
aws.bedrock.embedding.model=amazon.titan-embed-text-v2:0
```

### 4. Build and Run

```bash
# Build the application
mvn clean install

# Run the application
mvn spring-boot:run
```

The application will start on `http://localhost:8080`

## 📝 API Endpoints

### 1. Index Documents

Upload and index documents for vector search:

```bash
curl -X POST http://localhost:8080/api/chat/index \
  -F "file=@your_document.pdf" \
  -F "id=doc1"
```

### 2. Query Knowledge Base

Ask questions against your indexed documents:

```bash
curl -X POST http://localhost:8080/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the main topic discussed in the documents?"
  }'
```

### 3. Basic Chat

Direct chat with Claude (without RAG):

```bash
curl -X POST http://localhost:8080/api/chat/send \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?"
  }'
```

## 🔧 How Qdrant Integration Works

### Vector Storage Process

1. **Document Upload**: Files are uploaded via `/api/chat/index`
2. **Text Extraction**: Content is extracted based on file type
3. **Embedding Generation**: Amazon Titan creates 1536-dimensional embeddings
4. **Qdrant Storage**: Vectors stored with metadata in Qdrant collection

### Vector Search Process

1. **Query Embedding**: User query converted to vector using same Titan model
2. **Similarity Search**: Qdrant performs cosine similarity search
3. **Context Retrieval**: Top matching documents retrieved
4. **LLM Response**: Claude generates response using retrieved context

### Qdrant Collection Structure

```json
{
  "id": "uuid",
  "vector": [1536-dimensional array],
  "payload": {
    "document_id": "user_provided_id",
    "content": "extracted_text_content", 
    "timestamp": 1635724800000
  }
}
```

## 🔍 Vector Database Features

- **Persistent Storage**: Docker volume for data persistence
- **Cosine Similarity**: Optimized for text embeddings
- **Metadata Storage**: Document content and metadata searchable
- **HTTP API**: RESTful interface for vector operations
- **Scalable**: Production-ready vector database

## 🚨 Troubleshooting

### Qdrant Connection Issues

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# View Qdrant logs
docker logs qdrant-vector-db

# Restart Qdrant
docker-compose restart qdrant
```

### AWS Bedrock Access

- Ensure your AWS account has Bedrock model access enabled
- Check AWS credentials are properly configured
- Verify region settings match your Bedrock setup

### Application Logs

Logs are written to `logs/rag-application.log` for debugging.

## 📊 Performance Notes

- **Embedding Size**: Amazon Titan generates 1536-dimensional vectors
- **Search Limit**: Returns top 5 similar documents by default
- **Similarity Threshold**: Cosine similarity scores from 0.0 to 1.0
- **Collection Auto-Init**: Qdrant collection created automatically on startup

## 🔮 Future Enhancements

- [ ] Implement proper Qdrant Java client (currently using HTTP API)
- [ ] Add document chunking for large files
- [ ] Implement user session management
- [ ] Add support for more file formats
- [ ] Integrate with Spring Security for authentication
- [ ] Add monitoring and metrics collection
