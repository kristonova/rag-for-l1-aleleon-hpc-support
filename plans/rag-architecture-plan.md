# RAG Architecture Plan - Embedding Model Serving with Podman

## Executive Summary

This plan outlines the recommended architecture for serving the `intfloat/multilingual-e5-large` embedding model, persisting embeddings to ChromaDB, and orchestrating the RAG chain using LangChain—all running on Podman with AMD ROCm GPU support.

---

## Current Architecture Analysis

### Existing Setup (`compose.yml`)
- **vLLM Server**: `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` on AMD ROCm GPUs (port 8000)
- **RAG App**: Currently commented out, would use `HuggingFaceEmbeddings` locally
- **Vector Store**: In-memory ChromaDB (no persistence)
- **Orchestration**: Podman-compose with profiles

### Existing Setup (`rag_slurm_vllm.py`)
- **Embedding Model**: `intfloat/multilingual-e5-large` (runs locally via CPU/GPU)
- **LLM Model**: `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (via vLLM)
- **Vector Store**: ChromaDB (in-memory, data lost on restart)
- **RAG Chain**: LangChain classic with retrieval + generation

### Issues with Current Setup
1. **No embedding model serving**: Embeddings computed locally, not as a service
2. **No persistent storage**: ChromaDB in-memory, data lost on container restart
3. **Tight coupling**: All logic in single script, hard to maintain
4. **No API interface**: Cannot reuse embedding service for other applications

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Podman Container Network                            │
│                                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │  embedding-service   │  │    vllm-rocm         │  │    chromadb      │  │
│  │  (Port 8001)         │  │  (Port 8000)         │  │  (Port 8002)     │  │
│  │                      │  │                      │  │                  │  │
│  │  intfloat/multi-     │  │  Qwen/Qwen3.5-35B-   │  │  ChromaDB        │  │
│  │  lingual-e5-large    │  │  A3B-GPTQ-Int4       │  │  Persistent      │  │
│  │  FastAPI +           │  │  vLLM + ROCm         │  │  Storage         │  │
│  │  sentence-transform  │  │  GPU Inference       │  │                  │  │
│  │                      │  │                      │  │                  │  │
│  │  REST API:           │  │  REST API:           │  │  REST API:       │  │
│  │  POST /embed         │  │  POST /v1/chat       │  │  GET/POST /api   │  │
│  └──────────┬───────────┘  └──────────┬───────────┘  └────────┬─────────┘  │
│             │                         │                       │            │
│             │                         │                       │            │
│             ▼                         ▼                       ▼            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        rag-app                                       │  │
│  │                        (LangChain RAG Chain)                         │  │
│  │                                                                      │  │
│  │  - Calls embedding-service for text embeddings                     │  │
│  │  - Stores embeddings in chromadb via API                           │  │
│  │  - Retrieves context from chromadb                                 │  │
│  │  - Queries vllm-rocm for LLM generation                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Embedding Service (`embedding-service`)

**Purpose**: Serve `intfloat/multilingual-e5-large` as a REST API

**Technology Stack**:
- Base Image: `python:3.11-slim`
- Framework: `FastAPI` + `uvicorn`
- Embedding Model: `sentence-transformers` (intfloat/multilingual-e5-large)
- Port: `8001`

**API Endpoints**:
```python
POST /embed
  Request: {"texts": ["text1", "text2", ...]}
  Response: {"embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]}

GET /health
  Response: {"status": "ok"}
```

**Dockerfile.embedding**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install fastapi uvicorn sentence-transformers chromadb

# Copy application code
COPY embedding_api.py .

# Expose port
EXPOSE 8001

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "embedding_api:app", "--host", "0.0.0.0", "--port", "8001"]
```

**Environment Variables**:
- `MODEL_NAME`: intfloat/multilingual-e5-large (default)

**Volume Mounts**:
- `~/.cache/huggingface:/root/.cache/huggingface` (for model weights)

---

### 2. LLM Service (`vllm-rocm`)

**Purpose**: Serve Qwen LLM inference on AMD ROCm GPUs

**Technology Stack**:
- Base Image: `docker.io/rocm/vllm-dev:nightly`
- Framework: `vLLM` with ROCm support
- Model: `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`
- Port: `8000`

**API Endpoints**:
```python
POST /v1/chat/completions
  Request: {"model": "...", "messages": [...], "temperature": 0.6}
  Response: {"choices": [{"message": {"content": "..."}}]}

GET /health
  Response: {"status": "ok"}
```

**Existing Configuration** (from compose.yml):
- Devices: `/dev/kfd`, `/dev/dri` (ROCm GPU access)
- Environment: `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
- Healthcheck: `curl -f http://localhost:8000/health`
- Start period: 300s (model loading time)

---

### 3. ChromaDB Vector Store (`chromadb`)

**Purpose**: Persistent vector database for embeddings

**Technology Stack**:
- Base Image: `chromadb/chroma:latest`
- Port: `8002` (mapped to container port 8000)
- Storage: Persistent volume `chromadb-data`

**API Endpoints**:
```python
GET /api/v1/heartbeat
  Response: {"heartbeat": 1234567890}

POST /api/v1/collections
  Request: {"name": "wiki-embeddings", "embedding_function": "default"}
  Response: {"id": "...", "name": "...", "dimension": 1024}

POST /api/v1/collections/{collection_id}/add
  Request: {"embeddings": [[...], [...]], "documents": ["text1", "text2"]}
  Response: {"ids": ["id1", "id2"]}

GET /api/v1/collections/{collection_id}/get
  Response: {"ids": [...], "embeddings": [[...]], "documents": [...]}

POST /api/v1/collections/{collection_id}/query
  Request: {"query_embeddings": [[...]], "n_results": 3}
  Response: {"ids": [...], "embeddings": [[...]], "distances": [...]}
```

**Environment Variables**:
- `CHROMA_SERVER_AUTHN_CREDENTIALS`: Secret key for authentication
- `CHROMA_SERVER_AUTHN_PROVIDER`: Token auth provider

**Volume Mounts**:
- `chromadb-data:/chroma/data` (persistent storage)

---

### 4. RAG Application (`rag-app`)

**Purpose**: Orchestrate RAG chain using all services

**Technology Stack**:
- Base Image: Custom (built from `Dockerfile.rag-app`)
- Framework: `LangChain` (classic API via `langchain-classic`)
- Dependencies: `chromadb`, `requests`, `langchain-huggingface`

**Service Dependencies**:
- `embedding-service`: For text embeddings
- `vllm-rocm`: For LLM generation
- `chromadb`: For vector storage and retrieval

**Environment Variables**:
- `EMBEDDING_API_URL`: http://embedding-service:8001
- `LLM_API_URL`: http://vllm-rocm:8000/v1
- `CHROMADB_URL`: http://chromadb:8000

**Workflow**:
```
1. User asks question
   │
   ▼
2. Call embedding-service POST /embed
   │  Input: [question]
   │  Output: [1024D embedding vector]
   │
   ▼
3. Call chromadb POST /collections/{id}/query
   │  Input: {query_embeddings: [...], n_results: 3}
   │  Output: Top 3 relevant chunks
   │
   ▼
4. Call vllm-rocm POST /v1/chat/completions
   │  Input: {messages: [{role: "user", content: "context + question"}]}
   │  Output: Generated answer
   │
   ▼
5. Return answer to user
```

---

## Updated compose.yml Structure

```yaml
services:
  # ---- Embedding Service (intfloat/multilingual-e5-large) ----
  embedding-service:
    build:
      context: .
      dockerfile: Dockerfile.embedding
    container_name: embedding-service
    profiles:
      - embedding
    ports:
      - "8001:8001"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - MODEL_NAME=intfloat/multilingual-e5-large
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s

  # ---- LLM Service (vLLM on ROCm) ----
  vllm-rocm:
    image: docker.io/rocm/vllm-dev:nightly
    container_name: vllm-rocm
    profiles:
      - vllm-rocm
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - keep-groups
    cap_add:
      - SYS_PTRACE
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8000:8000"
    environment:
      - FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
      - TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
    command: >
      vllm serve
      --model Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
      --dtype float16
      --gpu-memory-utilization 0.80
      --max-model-len 32768
      --tensor-parallel-size 1
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --reasoning-parser qwen3
      --trust-remote-code
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 10
      start_period: 300s

  # ---- ChromaDB Vector Store ----
  chromadb:
    image: chromadb/chroma:latest
    container_name: chromadb
    profiles:
      - chromadb
    ports:
      - "8002:8000"
    volumes:
      - chromadb-data:/chroma/data
    environment:
      - CHROMA_SERVER_AUTHN_CREDENTIALS=your-secret-key
      - CHROMA_SERVER_AUTHN_PROVIDER=chromadb.server.auth.token.TokenAuthProvider
    restart: unless-stopped

  # ---- RAG Application ----
  rag-app:
    build:
      context: .
      dockerfile: Dockerfile.rag-app
    container_name: rag-app
    profiles:
      - rag-app
    environment:
      - EMBEDDING_API_URL=http://embedding-service:8001
      - LLM_API_URL=http://vllm-rocm:8000/v1
      - CHROMADB_URL=http://chromadb:8000
    depends_on:
      embedding-service:
        condition: service_healthy
      vllm-rocm:
        condition: service_healthy
      chromadb:
        condition: service_healthy
    stdin_open: true
    tty: true

volumes:
  chromadb-data:
```

---

## Implementation Plan

### Phase 1: Embedding Service

1. Create `Dockerfile.embedding`
2. Create `embedding_api.py` with FastAPI endpoints
3. Test embedding service locally
4. Add to `compose.yml` with profile `embedding`

### Phase 2: ChromaDB Integration

1. Create persistent volume `chromadb-data`
2. Configure ChromaDB with authentication
3. Create `chromadb_client.py` for API interactions
4. Add to `compose.yml` with profile `chromadb`

### Phase 3: RAG Application

1. Create `Dockerfile.rag-app`
2. Refactor `rag_slurm_vllm.py` into `rag_app.py`
3. Update to use API clients for all services
4. Add to `compose.yml` with profile `rag-app`

### Phase 4: Testing & Validation

1. Test embedding service health check
2. Test ChromaDB persistence across restarts
3. Test full RAG pipeline end-to-end
4. Document usage instructions

---

## Usage Instructions

### Start All Services
```bash
podman-compose --profile embedding --profile vllm-rocm --profile chromadb --profile rag-app up
```

### Start Only Embedding Service
```bash
podman-compose --profile embedding up embedding-service
```

### Start Only LLM Service
```bash
podman-compose --profile vllm-rocm up vllm-rocm
```

### Start Only ChromaDB
```bash
podman-compose --profile chromadb up chromadb
```

### Run RAG Application
```bash
podman-compose --profile rag-app run rag-app
```

### Test Embedding API
```bash
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Test embedding"]}'
```

### Test ChromaDB API
```bash
curl http://localhost:8002/api/v1/heartbeat
```

### Test vLLM API
```bash
curl http://localhost:8000/health
```

---

## Benefits of This Architecture

1. **Modularity**: Each service is independent and can be updated separately
2. **Scalability**: Each service can be scaled independently based on load
3. **Reusability**: Embedding API can be used by other applications
4. **Persistence**: ChromaDB stores data across container restarts
5. **Health Checks**: All services have health checks for reliability
6. **Dependency Management**: RAG app waits for all services to be ready
7. **Podman-native**: All services run as containers with proper networking
8. **Profile-based**: Start only needed services to save resources

---

## Next Steps

1. Review this architecture plan
2. Confirm if any adjustments are needed
3. If approved, I'll create the actual files in Code mode:
   - `Dockerfile.embedding`
   - `embedding_api.py`
   - `Dockerfile.rag-app`
   - `rag_app.py` (refactored from `rag_slurm_vllm.py`)
   - `chromadb_client.py`
   - Updated `compose.yml`