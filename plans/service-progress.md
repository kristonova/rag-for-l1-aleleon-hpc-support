# Service Progress Documentation

## Overview

This document tracks the current progress of the RAG (Retrieval-Augmented Generation) system services running on Podman with AMD ROCm GPU support.

---

## Service Status

### ✅ Completed Services

#### 1. Embedding Service (`embedding-service`)

**Status**: **DONE** - Fully operational

**Purpose**: Serve `intfloat/multilingual-e5-large` embedding model as a REST API

**Command to Run**:
```bash
podman-compose --podman-run-args="--replace" --profile embedding-service up -d
```

**Configuration**:
- **Image**: Built from `services/embedding/Dockerfile.embedding`
- **Port**: 8001
- **Model**: intfloat/multilingual-e5-large
- **Health Check**: `curl -f http://localhost:8001/health`
- **Profile**: `embedding-service` (also available as `model`)

**API Endpoints**:
- `POST /embed` - Generate embeddings for text
- `GET /health` - Health check endpoint

**Files**:
- [`services/embedding/Dockerfile.embedding`](services/embedding/Dockerfile.embedding)
- [`services/embedding/embedding_api.py`](services/embedding/embedding_api.py)

---

#### 2. vLLM Service (`vllm-rocm`)

**Status**: **DONE** - Fully operational

**Purpose**: Serve Qwen LLM inference on AMD ROCm GPUs

**Command to Run**:
```bash
podman-compose --podman-run-args="--replace" --profile vllm-rocm up -d
```

**Configuration**:
- **Image**: `docker.io/rocm/vllm-dev:nightly`
- **Port**: 8000
- **Model**: Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
- **GPU**: AMD ROCm (via `/dev/kfd`, `/dev/dri`)
- **Health Check**: `curl -f http://localhost:8000/health`
- **Profile**: `vllm-rocm` (also available as `model`)

**API Endpoints**:
- `POST /v1/chat/completions` - LLM inference via OpenAI-compatible API
- `GET /health` - Health check endpoint

**Configuration**:
- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`

---

### ✅ Implemented Services

#### 3. ChromaDB Service (`chromadb`)

**Status**: **IMPLEMENTED** - Ready to use

**Purpose**: Persistent vector database for storing and retrieving embeddings

**Configuration**:
- **Image**: `chromadb/chroma:0.5.18` (v2 API support)
- **Port**: 8002 (mapped to container port 8000)
- **Storage**: Persistent volume `chromadb-data`
- **Authentication**: Token-based auth enabled

**Command to Run**:
```bash
podman-compose --podman-run-args="--replace" --profile chromadb up -d
```

**API Endpoints** (ChromaDB v2):
- `GET /api/v2/heartbeat` - Server heartbeat
- `POST /api/v2/collections` - Create collection
- `POST /api/v2/collections/{name}/add` - Add documents
- `POST /api/v2/collections/{name}/query` - Query collection

**Profile**: `chromadb`

---

#### 4. RAG Application (`rag-app`)

**Status**: **IMPLEMENTED** - Ready to use

**Purpose**: Orchestrate RAG chain using all services

**Configuration**:
- **Build**: From `services/rag-app/Dockerfile.rag-app`
- **Environment Variables**:
  - `EMBEDDING_API_URL=http://embedding-service:8001`
  - `LLM_API_URL=http://vllm-rocm:8000/v1`
  - `CHROMADB_URL=http://chromadb:8000`
  - `CHROMADB_AUTH_TOKEN=your-secret-key`
- **Dependencies**: Waits for all services to be healthy

**Command to Run**:
```bash
podman-compose --profile rag-app run rag-app
```

**Profile**: `rag-app`

---

## Combined Service Command

### Run Both Embedding Service and vLLM

**Command**:
```bash
podman-compose --podman-run-args="--replace" --profile model up -d
```

**Description**: This command starts both the embedding-service and vllm-rocm services together using the `model` profile.

**Use Case**: When you need both services running for testing or development without ChromaDB and RAG app.

---

## Service Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Podman Container Network                            │
│                                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │  embedding-service   │  │    vllm-rocm         │  │    chromadb      │  │
│  │  (Port 8001)         │  │  (Port 8000)         │  │  (Port 8002)     │  │
│  │                      │  │                      │  │                  │  │
│  │  ✅ intfloat/multi-  │  │  ✅ Qwen/Qwen3.5-35B- │  │  🚧 ChromaDB     │  │
│  │  lingual-e5-large    │  │  A3B-GPTQ-Int4       │  │  (WIP)           │  │
│  │  FastAPI +           │  │  vLLM + ROCm         │  │                  │  │
│  │  sentence-transform  │  │  GPU Inference       │  │                  │  │
│  │                      │  │                      │  │                  │  │
│  │  ✅ REST API:        │  │  ✅ REST API:        │  │  🚧 REST API:    │  │
│  │  POST /embed         │  │  POST /v1/chat       │  │  GET/POST /api   │  │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        rag-app                                       │  │
│  │                        (LangChain RAG Chain)                         │  │
│  │                                                                      │  │
│  │  🚧 Not yet implemented                                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Usage Guide

### Start Individual Services

**Embedding Service Only**:
```bash
podman-compose --podman-run-args="--replace" --profile embedding-service up -d
```

**vLLM Service Only**:
```bash
podman-compose --podman-run-args="--replace" --profile vllm-rocm up -d
```

### Start Both Services Together

**Model Services (Embedding + vLLM)**:
```bash
podman-compose --podman-run-args="--replace" --profile model up -d
```

### Test Services

**Test Embedding Service**:
```bash
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Test embedding"]}'
```

**Test vLLM Service**:
```bash
curl http://localhost:8000/health
```

### Check Service Status

```bash
podman-compose ps
podman-compose logs embedding-service
podman-compose logs vllm-rocm
```

---

## Next Steps

1. **Implement ChromaDB Service**:
   - Create persistent volume configuration
   - Add authentication setup
   - Test vector storage and retrieval

2. **Implement RAG Application**:
   - Create `Dockerfile.rag-app`
   - Refactor existing RAG logic into service
   - Integrate with all three services

3. **End-to-End Testing**:
   - Test full RAG pipeline
   - Validate data persistence across restarts
   - Performance benchmarking

---

## References

- **Architecture Plan**: [`plans/rag-architecture-plan.md`](plans/rag-architecture-plan.md)
- **Main Configuration**: [`compose.yml`](compose.yml)
- **Embedding Service**: [`services/embedding/`](services/embedding/)
- **README**: [`README.md`](README.md)