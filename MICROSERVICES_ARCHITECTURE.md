# Arsitektur Microservices — RAG for L1 ALELEON HPC Support

Dokumen ini menjelaskan secara teknis dan lengkap seluruh bangunan microservices yang ada di project ini, termasuk hubungan antar service, teknologi yang digunakan, alur data, dan detail implementasi.

---

## 1. Diagram Arsitektur Utama

```mermaid
graph TB
    subgraph "User Interface"
        TG["👤 Telegram User"]
    end

    subgraph "Frontend Service"
        TB["telegram-bot<br/>python-telegram-bot<br/>Port: -"]
    end

    subgraph "API Gateway"
        API["rag-api<br/>FastAPI + Uvicorn<br/>Port: 8080"]
    end

    subgraph "Core Logic"
        RAG["rag_app.py<br/>RAG Chain Engine<br/>Hybrid Retrieval"]
    end

    subgraph "Infrastructure Services"
        EMB["embedding-service<br/>BAAI/bge-m3 (FlagEmbedding)<br/>Port: 8001"]
        VLLM["vllm-rocm<br/>Qwen3.5-35B-A3B-GPTQ-Int4<br/>Port: 8000"]
        QD["qdrant<br/>Vector Database<br/>Port: 6333/6334"]
    end

    subgraph "Observability"
        PROM["promtail<br/>Log Aggregator<br/>→ Loki"]
    end

    subgraph "Benchmark Services"
        BR["benchmark<br/>Retrieval Benchmark"]
        BT["benchmark-ttft<br/>Latency Benchmark"]
    end

    TG -->|"Telegram API<br/>Long Polling"| TB
    TB -->|"POST /ask<br/>POST /review-script"| API
    API --> RAG
    RAG -->|"POST /embed<br/>POST /embed/multi"| EMB
    RAG -->|"OpenAI-compatible<br/>chat.completions"| VLLM
    RAG -->|"query_points<br/>upsert"| QD
    BR -->|"embed + query"| EMB
    BR -->|"generate"| VLLM
    BR -->|"4 collections"| QD
    BT -->|"POST /ask"| API
    PROM -->|"scrape logs"| TB
    PROM -->|"scrape logs"| API
    PROM -->|"scrape logs"| EMB
```

---

## 2. Diagram Alur Data (Request Flow)

### 2.1 Alur `/ask` (Standard Question)

```mermaid
sequenceDiagram
    participant U as Telegram User
    participant TB as telegram-bot
    participant API as rag-api :8080
    participant LLM as vllm-rocm :8000
    participant EMB as embedding-service :8001
    participant QD as qdrant :6333

    U->>TB: /ask "Bagaimana cara membuat conda?"
    TB->>API: POST /ask {"question": "..."}

    Note over API: 1. Relevance Check
    API->>LLM: chat.completions (is_question_relevant)
    LLM-->>API: "YA"

    Note over API: 2. Hybrid Embedding
    API->>EMB: POST /embed/multi (dense + sparse)
    EMB-->>API: {dense: [...], sparse: {...}}

    Note over API: 3. Hybrid Search (RRF)
    API->>QD: query_points (Prefetch dense + sparse → RRF Fusion)
    QD-->>API: Top-10 chunks

    Note over API: 4. LLM Generation
    API->>LLM: chat.completions (context + question)
    LLM-->>API: answer

    Note over API: 5. Source Justification
    API->>LLM: chat.completions (justify sources)
    LLM-->>API: justifications per source

    API-->>TB: {answer, sources[]}
    TB-->>U: Formatted HTML reply + sources
```

### 2.2 Alur `/askscript` (Hybrid Script Review)

```mermaid
sequenceDiagram
    participant U as Telegram User
    participant TB as telegram-bot
    participant API as rag-api :8080
    participant LLM as vllm-rocm :8000
    participant EMB as embedding-service :8001
    participant QD as qdrant :6333

    U->>TB: /askscript + script content
    TB->>API: POST /review-script {"script": "..."}

    Note over API: 1. Relevance Check
    API->>LLM: is_question_relevant(script)
    LLM-->>API: "YA"

    Note over API: 2. Extract Resource Params
    API->>LLM: extract_resource_params(script)
    LLM-->>API: {partition, mem, time, ...}

    Note over API: 3. Policy Retrieval (per param)
    API->>EMB: embed_query_multi (targeted queries)
    EMB-->>API: dense + sparse vectors
    API->>QD: query_points per parameter
    QD-->>API: policy documents

    Note over API: 4. Hybrid Review (teknis + policy)
    API->>LLM: review script + policy context
    LLM-->>API: review text

    API-->>TB: {review, issues_found, policy_sources[]}
    TB-->>U: Formatted review + policy references
```

---

## 3. Detail Setiap Microservice

### 3.1 Embedding Service (`embedding-service`)

| Aspek | Detail |
|-------|--------|
| **Lokasi Kode** | `services/embedding/embedding_api.py` |
| **Dockerfile** | `services/embedding/Dockerfile.embedding` |
| **Base Image** | `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1` |
| **Framework** | FastAPI + Uvicorn |
| **Model** | `BAAI/bge-m3` via FlagEmbedding (`BGEM3FlagModel`) |
| **Port** | `8001` |
| **Profile Compose** | `infra`, `embedding` |

**Endpoints:**

| Endpoint | Method | Fungsi |
|----------|--------|--------|
| `POST /embed` | Dense-only | Backward-compatible, return `List[List[float]]` (1024-dim) |
| `POST /embed/multi` | Multi-mode | Dense + Sparse (lexical) + ColBERT, configurable via flags |
| `GET /health` | Health check | Status model |

**Arsitektur Internal:**
- Model di-load saat startup menggunakan `BGEM3FlagModel` dengan `use_fp16=True`
- Sparse output berupa `{indices: List[int], values: List[float]}` — format native Qdrant `SparseVector`
- ColBERT output berupa 2D array (tokens × 1024-dim) untuk late interaction
- Monkey-patch `is_torch_fx_available` untuk kompatibilitas transformers ≥4.47

---

### 3.2 LLM Service (`vllm-rocm`)

| Aspek | Detail |
|-------|--------|
| **Image** | `docker.io/rocm/vllm-dev:nightly` |
| **Model** | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` |
| **Port** | `8000` (OpenAI-compatible API) |
| **GPU** | AMD ROCm (`/dev/kfd`, `/dev/dri`) |
| **Profile Compose** | `infra`, `vllm` |

**Konfigurasi vLLM:**

```
--dtype float16
--enforce-eager
--gpu-memory-utilization 0.99
--max-model-len 262144
--max-num-seqs 16
--tensor-parallel-size 1
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
--reasoning-parser qwen3
--enable-prefix-caching
```

**API yang Digunakan:**
- `POST /v1/chat/completions` — OpenAI-compatible, diakses via Python `openai` SDK
- Parameter: `temperature=0.3`, `top_p=0.9`, `presence_penalty=1.5`, `top_k=20`
- Non-thinking mode: `extra_body.chat_template_kwargs.enable_thinking = False`

---

### 3.3 Vector Database (`qdrant`)

| Aspek | Detail |
|-------|--------|
| **Image** | `docker.io/qdrant/qdrant:latest` |
| **Port** | `6333` (REST), `6334` (gRPC) |
| **Storage** | Docker volume `qdrant-data` → `/qdrant/storage` |
| **Profile Compose** | `infra`, `qdrant` |

**Collection Utama: `wiki_aleleon_qdrant`**

```mermaid
graph LR
    subgraph "Collection: wiki_aleleon_qdrant"
        D["dense<br/>VectorParams(1024, COSINE)"]
        S["text-sparse<br/>SparseVectorParams()"]
        P["Payload<br/>text, title, source, Header 2/3"]
    end
```

- **Dense vector**: 1024-dim float, cosine distance
- **Sparse vector**: BM25-like lexical weights dari bge-m3
- **Payload**: `text` (chunk content), `title`, `source` (URL), `Header 2`, `Header 3`
- **Retrieval**: Hybrid search via `Prefetch` (dense + sparse) → `RRF Fusion`

---

### 3.4 RAG Application (`rag-app` / `rag-api`)

| Aspek | Detail |
|-------|--------|
| **Lokasi Kode** | `services/rag-app/rag_app.py` (core), `services/rag-app/rag_api.py` (API) |
| **Dockerfile** | `services/rag-app/Dockerfile.rag-app` |
| **Base Image** | `python:3.11-slim` |
| **Framework** | FastAPI + Uvicorn (API), standalone CLI (interactive) |
| **Port** | `8080` (API mode) |
| **Profile Compose** | `rag-app` (CLI), `api` (REST) |

**Dua Mode Operasi:**
1. **CLI Mode** (`rag-app` profile): Menjalankan `rag_app.py` secara interaktif dengan daftar pertanyaan benchmark
2. **API Mode** (`api` profile): Menjalankan `rag_api.py` sebagai REST server via Uvicorn

**Endpoints API (`rag_api.py`):**

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `POST /ask` | Question Answering | `{"question": "..."}` | `{answer, sources[{title, source_url, section, justification}]}` |
| `POST /review-script` | Script Review | `{"script": "..."}` | `{review, issues_found, policy_sources[]}` |
| `GET /health` | Health check | - | `{status, service}` |

**Startup Sequence (`rag_api.py`):**

```mermaid
graph TD
    A["1. Init EmbeddingServiceClient"] --> B["2. Check Qdrant collection exists?"]
    B -->|Ya| C["load_vectorstore()"]
    B -->|Tidak| D["build_vectorstore()<br/>Scrape wiki → embed → upsert"]
    C --> E["3. wait_for_vllm()"]
    D --> E
    E --> F["4. create_rag_chain()"]
    F --> G["✅ RAG chain ready"]
```

**Komponen Inti `rag_app.py`:**

| Komponen | Fungsi |
|----------|--------|
| `EmbeddingServiceClient` | LangChain-compatible wrapper untuk embedding API, support `embed_query`, `embed_multi`, `embed_query_multi` |
| `load_wiki_documents()` | Scrape wiki via sitemap XML → fetch HTML → `HTMLSectionSplitter` (h1/h2/h3) → fallback `RecursiveCharacterTextSplitter` (4500 chars, 900 overlap) |
| `build_vectorstore()` | Full ingestion pipeline: scrape → embed multi → upsert ke Qdrant (dense + sparse) |
| `is_question_relevant()` | LLM-based relevance filter sebelum RAG processing |
| `create_rag_chain()` | Hybrid retrieval (dense + sparse → RRF) → LLM generation → source justification → irrelevant source filtering |
| `extract_resource_params()` | LLM-based parser untuk #SBATCH directives dari script |
| `retrieve_policy_context()` | Targeted retrieval berdasarkan extracted params (partition, time, mem, GPU, dll) |
| `review_script_hybrid()` | 3-step: extract params → retrieve policy → LLM review (teknis + policy) |
| `generate_source_justifications()` | LLM menjelaskan relevansi setiap source, filter "TIDAK RELEVAN" |

---

### 3.5 Telegram Bot (`telegram-bot`)

| Aspek | Detail |
|-------|--------|
| **Lokasi Kode** | `services/telegram-bot/telegram_bot.py` |
| **Dockerfile** | `services/telegram-bot/Dockerfile.telegram` |
| **Base Image** | `python:3.11-slim` |
| **Library** | `python-telegram-bot[all]`, `requests` |
| **Profile Compose** | `telegram` |
| **Komunikasi** | HTTP ke `rag-api:8080` |

**Commands:**

| Command | Handler | Fungsi |
|---------|---------|--------|
| `/start` | `start()` | Welcome message |
| `/help` | `help_command()` | Usage instructions |
| `/status` | `status()` | Health check RAG API |
| `/ask <question>` | `ask_command()` | Standard RAG Q&A |
| `/askscript <script>` | `askscript_command()` | Hybrid script review |
| File upload (.sh/.slurm/.sbatch/.bash) | `handle_document()` | Auto script review |

**Fitur Teknis:**
- **Progress animation**: Background task dengan `asyncio` yang memutar emoji placeholder + typing indicator
- **Markdown→HTML converter**: `markdown_to_telegram_html()` — konversi code blocks, bold, italic, strikethrough, links
- **HTML splitter**: `split_html_for_telegram()` — otomatis track dan close/reopen HTML tags di batas 4000 char
- **Fallback**: Jika HTML parse gagal, strip semua tags dan kirim plain text

---

### 3.6 Benchmark Retrieval (`benchmark`)

| Aspek | Detail |
|-------|--------|
| **Lokasi Kode** | `services/benchmark_retrieval/benchmark_retrieval.py` |
| **Dockerfile** | `services/benchmark_retrieval/Dockerfile.benchmark` |
| **Base Image** | `python:3.12-slim` |
| **Profile Compose** | `benchmark` |
| **Output** | `./output/benchmark/` (bind mount) |

**4 Metode Retrieval yang Dibandingkan:**

```mermaid
graph LR
    subgraph "Benchmark Collections"
        D["bench_dense<br/>Cosine 1024-dim"]
        S["bench_sparse<br/>Lexical BM25-like"]
        M["bench_multivec<br/>ColBERT MaxSim"]
        H["bench_hybrid<br/>Dense+Sparse RRF"]
    end
```

| Metode | Collection | Teknik |
|--------|-----------|--------|
| Dense | `bench_dense` | Cosine similarity pada vektor 1024-dim |
| Sparse | `bench_sparse` | Lexical weights (BM25-like dari bge-m3) |
| Multi-Vector | `bench_multivec` | ColBERT late interaction (MaxSim) |
| Hybrid | `bench_hybrid` | Dense + Sparse → Reciprocal Rank Fusion (RRF) |

**Metrik:** Retrieval time (ms), E2E time (ms), Jaccard overlap antar metode, Ingestion time.

---

### 3.7 Benchmark TTFT (`benchmark-ttft`)

| Aspek | Detail |
|-------|--------|
| **Lokasi Kode** | `services/benchmark_ttft/run-rag-bench.py` |
| **Dockerfile** | `services/benchmark_ttft/Dockerfile.rag-bench` |
| **Base Image** | `python:3.11-slim` |
| **Library** | `aiohttp` (async HTTP) |
| **Profile Compose** | `benchmark-ttft` |
| **Output** | `./output/benchmark_ttft/` (bind mount) |

**Metrik:** E2E latency per request, P50/P99 latency, throughput (RPS).
**Concurrency levels:** 1, 2, 4, 8, 16 (configurable via env `CONCURRENCY_LEVELS`).

---

### 3.8 Promtail (`promtail`)

| Aspek | Detail |
|-------|--------|
| **Image** | `docker.io/grafana/promtail:latest` |
| **Config** | `services/promtail/config.yml` |
| **Profile Compose** | `monitoring` |
| **Target** | Loki di `http://172.16.1.10:3100` |

**Cara Kerja:**
- Scrape log container via Podman socket (`$XDG_RUNTIME_DIR/podman/podman.sock`)
- Label: `container` (nama), `image`, `service` (compose service), `user` (host user)
- Docker SD (service discovery) dengan refresh setiap 5s

---

## 4. Diagram Deployment (Container Topology)

```mermaid
graph TB
    subgraph "Profile: infra"
        EMB["embedding-service<br/>:8001<br/>BAAI/bge-m3<br/>ROCm PyTorch"]
        VLLM["vllm-rocm<br/>:8000<br/>Qwen3.5-35B<br/>AMD GPU /dev/kfd"]
        QD["qdrant<br/>:6333/:6334<br/>Volume: qdrant-data"]
    end

    subgraph "Profile: api"
        API["rag-api<br/>:8080<br/>FastAPI"]
    end

    subgraph "Profile: telegram"
        TB["telegram-bot<br/>Long Polling"]
    end

    subgraph "Profile: monitoring"
        PR["promtail<br/>:9080<br/>→ Loki"]
    end

    subgraph "Profile: benchmark"
        BR["benchmark<br/>Retrieval"]
    end

    subgraph "Profile: benchmark-ttft"
        BT["benchmark-ttft<br/>Latency"]
    end

    API --> EMB
    API --> VLLM
    API --> QD
    TB --> API
    BR --> EMB
    BR --> VLLM
    BR --> QD
    BT --> API
```

---

## 5. Tabel Ringkasan Service

| Service | Port | Base Image | Profile | Healthcheck | Restart |
|---------|------|-----------|---------|-------------|---------|
| `embedding-service` | 8001 | `rocm/pytorch:rocm7.2` | infra, embedding | `curl /health` | unless-stopped |
| `vllm-rocm` | 8000 | `rocm/vllm-dev:nightly` | infra, vllm | `curl /health` | unless-stopped |
| `qdrant` | 6333, 6334 | `qdrant/qdrant:latest` | infra, qdrant | TCP `/healthz` | unless-stopped |
| `rag-app` | - | `python:3.11-slim` | rag-app | - | - |
| `rag-api` | 8080 | `python:3.11-slim` | api | `curl /health` | unless-stopped |
| `telegram-bot` | - | `python:3.11-slim` | telegram | `pidof python` | unless-stopped |
| `benchmark` | - | `python:3.12-slim` | benchmark | - | - |
| `benchmark-ttft` | - | `python:3.11-slim` | benchmark-ttft | - | - |
| `promtail` | 9080 | `grafana/promtail:latest` | monitoring | `promtail --version` | always |

---

## 6. Environment Variables

| Variable | Service | Default | Deskripsi |
|----------|---------|---------|-----------|
| `EMBEDDING_API_URL` | rag-app, rag-api, benchmark | `http://embedding-service:8001` | URL embedding service |
| `LLM_API_URL` | rag-app, rag-api | `http://vllm-rocm:8000/v1` | URL vLLM OpenAI API |
| `QDRANT_URL` | rag-app, rag-api, benchmark | `http://qdrant:6333` | URL Qdrant REST |
| `QDRANT_API_KEY` | rag-app, rag-api, benchmark | `your-secret-key` | API key Qdrant |
| `TELEGRAM_TOKEN` | telegram-bot | - | Bot token dari BotFather |
| `RAG_API_URL` | telegram-bot, benchmark-ttft | `http://rag-api:8080` | URL RAG API internal |
| `MODEL_NAME` | embedding-service | `BAAI/bge-m3` | Nama model embedding |
| `LLM_MODEL_NAME` | benchmark | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | Nama model LLM |

---

## 7. Data Ingestion Pipeline

```mermaid
graph TD
    A["Wiki Sitemap XML<br/>wiki.efisonlt.com"] --> B["Parse URLs<br/>Filter non-webpage"]
    B --> C["Fetch HTML per page<br/>Rate limit: 2 req/s"]
    C --> D["Extract div#mw-content-text<br/>BeautifulSoup + lxml"]
    D --> E["HTMLSectionSplitter<br/>Split by h1/h2/h3"]
    E --> F{"Chunk > 4500 chars?"}
    F -->|Ya| G["RecursiveCharacterTextSplitter<br/>4500 chars, 900 overlap"]
    F -->|Tidak| H["Keep as-is"]
    G --> I["Add metadata prefix<br/>[Sumber: title] [Section: header]"]
    H --> I
    I --> J["embed_multi()<br/>Dense + Sparse batch=16"]
    J --> K["Upsert to Qdrant<br/>batch=32"]
    K --> L["Collection: wiki_aleleon_qdrant<br/>dense + text-sparse vectors"]
```

---

## 8. Hybrid Retrieval Pipeline

```mermaid
graph TD
    Q["User Question"] --> REL{"is_question_relevant()?"}
    REL -->|TIDAK| REJECT["Return: not relevant"]
    REL -->|YA| EMB["embed_query_multi()<br/>→ dense + sparse vectors"]
    EMB --> P1["Prefetch Dense<br/>limit=20, using='dense'"]
    EMB --> P2["Prefetch Sparse<br/>limit=20, using='text-sparse'"]
    P1 --> RRF["RRF Fusion<br/>Reciprocal Rank Fusion"]
    P2 --> RRF
    RRF --> TOP["Top-10 chunks"]
    TOP --> GEN["generate_response()<br/>vLLM chat.completions"]
    GEN --> JUST["generate_source_justifications()<br/>Per-source relevance"]
    JUST --> FILT["Filter 'TIDAK RELEVAN' sources"]
    FILT --> OUT["Return: answer + filtered sources"]
```

---

## 9. Perintah Deployment

```bash
# Start semua infrastructure
podman-compose --profile infra up -d

# Start full stack (API + Telegram)
podman-compose --profile infra --profile api --profile telegram up -d

# Benchmark retrieval
podman-compose --profile infra up -d
podman-compose --profile benchmark run benchmark

# Benchmark latency (TTFT)
podman-compose --profile infra --profile api up -d
podman-compose --profile benchmark-ttft run benchmark-ttft

# Monitoring
podman-compose --profile monitoring up -d

# Stop & cleanup
podman-compose --profile infra --profile api --profile telegram down

# Rebuild setelah edit code
podman-compose --profile infra --profile api --profile telegram up -d --build
```
