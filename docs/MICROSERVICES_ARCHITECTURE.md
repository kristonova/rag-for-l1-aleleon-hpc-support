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
        RAG["rag_app.py<br/>RAG Chain Engine<br/>Hybrid Retrieval + ColBERT Rerank"]
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
    RAG -->|"POST /embed<br/>POST /embed/multi<br/>POST /rerank"| EMB
    RAG -->|"OpenAI-compatible<br/>chat.completions"| VLLM
    RAG -->|"query_points<br/>upsert<br/>scroll/delete (sync)"| QD
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

    Note over API: 0. Log question ke file

    Note over API: 1. Relevance Check
    API->>LLM: chat.completions (is_question_relevant)
    LLM-->>API: "YA"

    Note over API: 2. Hybrid Embedding
    API->>EMB: POST /embed/multi (dense + sparse)
    EMB-->>API: {dense: [...], sparse: {...}}

    Note over API: 3. Hybrid Search (RRF) — over-fetch 2×
    API->>QD: query_points (Prefetch dense + sparse → RRF Fusion, limit=20)
    QD-->>API: Top-20 candidates

    Note over API: 4. ColBERT Reranking
    API->>EMB: POST /rerank (query, 20 passages)
    EMB-->>API: ColBERT scores[]
    Note over API: Sort by score → Top-10

    Note over API: 5. LLM Generation
    API->>LLM: chat.completions (context + question)
    LLM-->>API: answer

    Note over API: 6. Source Justification
    API->>LLM: chat.completions (justify sources)
    LLM-->>API: justifications per source

    Note over API: 7. Filter irrelevant sources

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

    Note over API: 5. Source Justification + Filter
    API->>LLM: justify policy sources
    LLM-->>API: justifications (filter TIDAK RELEVAN)

    API-->>TB: {review, issues_found, policy_sources[]}
    TB-->>U: Formatted review + policy references
```

### 2.3 Alur `/refresh` (Incremental Sync)

```mermaid
sequenceDiagram
    participant C as Client (curl/bot)
    participant API as rag-api :8080
    participant SM as Sitemap (wiki)
    participant EMB as embedding-service :8001
    participant QD as qdrant :6333

    C->>API: POST /refresh
    Note over API: Spawn background thread

    API->>SM: GET sitemap XML
    SM-->>API: URL + lastmod list

    Note over API: Compare sitemap vs Qdrant state

    API->>QD: scroll (get stored URLs + lastmod)
    QD-->>API: stored state

    Note over API: Identify new/updated/deleted URLs

    opt Deleted or Updated URLs
        API->>QD: delete points by source URL filter
    end

    opt New or Updated URLs
        API->>SM: scrape changed pages
        SM-->>API: HTML content
        API->>EMB: POST /embed/multi (dense + sparse)
        EMB-->>API: embeddings
        API->>QD: upsert new points
    end

    C->>API: GET /refresh/status
    API-->>C: {running, last_result, last_sync_time}
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
| **Version** | `2.0.0` |
| **Profile Compose** | `infra`, `embedding` |

**Endpoints:**

| Endpoint | Method | Fungsi |
|----------|--------|--------|
| `POST /embed` | Dense-only | Backward-compatible, return `List[List[float]]` (1024-dim) |
| `POST /embed/multi` | Multi-mode | Dense + Sparse (lexical) + ColBERT, configurable via flags |
| `POST /rerank` | ColBERT rerank | Rerank passages terhadap query menggunakan ColBERT late-interaction scoring |
| `GET /health` | Health check | Status model |

**Arsitektur Internal:**
- Model di-load saat startup menggunakan `BGEM3FlagModel` dengan `use_fp16=True`
- Sparse output berupa `{indices: List[int], values: List[float]}` — format native Qdrant `SparseVector`
- ColBERT output berupa 2D array (tokens × 1024-dim) untuk late interaction
- Monkey-patch `is_torch_fx_available` untuk kompatibilitas transformers ≥4.47
- Rerank menggunakan `compute_score()` dengan weights `[0.0, 0.0, 1.0]` (ColBERT-only)

**Alur Kerja Embedding Service:**
```mermaid
graph TD
    A[Client Request<br/>POST /embed/multi] --> B[Parse Payload JSON]
    B --> C{Extract Components}
    C -->|List of Strings| D[Text Inputs]
    C -->|Booleans| E[Config: dense, sparse, colbert]
    
    D --> F[BGEM3FlagModel Inference<br/>batch processing]
    E -.-> F
    
    F --> G{Formatting Output}
    
    G -->|if return_dense| H[Dense Vectors<br/>1024-dim float arrays]
    G -->|if return_sparse| I[Sparse Vectors<br/>indices & weight values]
    G -->|if return_colbert| J[ColBERT Vectors<br/>tokens x 1024-dim arrays]
    
    H --> K[Construct Response JSON]
    I --> K
    J --> K
    
    K --> L[Return Response to Caller]
```

**Alur Kerja Rerank Endpoint:**
```mermaid
graph TD
    A[Client Request<br/>POST /rerank] --> B["Parse: query + passages[]"]
    B --> C["Build sentence pairs<br/>[[query, p1], [query, p2], ...]"]
    C --> D["compute_score()<br/>weights: [0, 0, 1] (ColBERT-only)"]
    D --> E["Extract colbert scores"]
    E --> F["Return scores[]<br/>(higher = more relevant)"]
```

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
--trust-remote-code
```

**API yang Digunakan:**
- `POST /v1/chat/completions` — OpenAI-compatible, diakses via Python `openai` SDK
- Parameter: `temperature=0.3`, `top_p=0.9`, `presence_penalty=1.5`, `top_k=20`
- Non-thinking mode: `extra_body.chat_template_kwargs.enable_thinking = False`

**Environment Variables Compose:**
- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`

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
        P["Payload<br/>text, title, source, Header 2/3, lastmod"]
    end
```

- **Dense vector**: 1024-dim float, cosine distance
- **Sparse vector**: BM25-like lexical weights dari bge-m3
- **Payload**: `text` (chunk content), `title`, `source` (URL), `Header 2`, `Header 3`, `lastmod` (timestamp dari sitemap untuk incremental sync)
- **Retrieval**: Hybrid search via `Prefetch` (dense + sparse) → `RRF Fusion` → ColBERT Reranking

**Konfigurasi Qdrant:**
- `QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=100`

---

### 3.4 RAG Application (`rag-app` / `rag-api`)

| Aspek | Detail |
|-------|--------|
| **Lokasi Kode** | `services/rag-app/rag_app.py` (core), `services/rag-app/rag_api.py` (API) |
| **Dockerfile** | `services/rag-app/Dockerfile.rag-app` |
| **Base Image** | `python:3.11-slim` |
| **Framework** | FastAPI + Uvicorn (API), standalone CLI (interactive) |
| **Port** | `8080` (API mode) |
| **Version** | `1.3.0` |
| **Profile Compose** | `rag-app` (CLI), `api` (REST) |

**Dua Mode Operasi:**
1. **CLI Mode** (`rag-app` profile): Menjalankan `rag_app.py` secara interaktif dengan daftar pertanyaan benchmark
2. **API Mode** (`api` profile): Menjalankan `rag_api.py` sebagai REST server via Uvicorn

**Endpoints API (`rag_api.py`):**

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `POST /ask` | Question Answering | `{"question": "..."}` | `{answer, sources[{title, source_url, section, justification}]}` |
| `POST /review-script` | Script Review | `{"script": "..."}` | `{review, issues_found, policy_sources[]}` |
| `POST /refresh` | Incremental Sync | - | `{status, result?, message?}` |
| `GET /refresh/status` | Sync Status | - | `{running, last_result, last_sync_time}` |
| `GET /health` | Health check | - | `{status, service}` |
| `GET /` | Service info | - | `{service, version, endpoints}` |

**Startup Sequence (`rag_api.py`):**

```mermaid
graph TD
    A["1. Init EmbeddingServiceClient"] --> B["2. Check Qdrant collection exists?"]
    B -->|Ya| C["load_vectorstore()"]
    B -->|Tidak| D["build_vectorstore()<br/>Scrape wiki → embed → upsert"]
    C --> E["3. wait_for_vllm()"]
    D --> E
    E --> F["4. create_rag_chain()"]
    F --> G["5. Startup sync<br/>sync_vectorstore()<br/>Cek perubahan sitemap"]
    G --> H["✅ RAG chain ready"]
```

**Komponen Inti `rag_app.py`:**

| Komponen | Fungsi |
|----------|--------|
| `EmbeddingServiceClient` | LangChain-compatible wrapper untuk embedding API, support `embed_query`, `embed_multi`, `embed_query_multi`, `rerank` |
| `parse_sitemap()` | Parse sitemap XML → dict {url: lastmod_str}, filter non-webpage URLs |
| `scrape_wiki_pages()` | Scrape list of wiki URLs → list of Document splits, with lastmod metadata |
| `load_wiki_documents()` | Scrape wiki via sitemap XML → fetch HTML → `HTMLSectionSplitter` (h1/h2/h3) → fallback `RecursiveCharacterTextSplitter` (4500 chars, 900 overlap) |
| `build_vectorstore()` | Full ingestion pipeline: scrape → embed multi → upsert ke Qdrant (dense + sparse) |
| `sync_vectorstore()` | Incremental sync: bandingkan sitemap terkini vs Qdrant → scrape/embed/upsert hanya yang berubah/baru, hapus yang dihapus |
| `get_stored_sitemap_state()` | Scroll Qdrant → {source_url: lastmod} untuk semua unique source URLs |
| `_migrate_add_lastmod()` | One-time migration: tambahkan lastmod ke points lama yang belum punya |
| `is_question_relevant()` | LLM-based relevance filter sebelum RAG processing |
| `create_rag_chain()` | Hybrid retrieval (dense + sparse → RRF → ColBERT rerank) → LLM generation → source justification → irrelevant source filtering |
| `extract_resource_params()` | LLM-based parser untuk #SBATCH directives dari script |
| `retrieve_policy_context()` | Targeted retrieval berdasarkan extracted params (partition, time, mem, GPU, dll) |
| `review_script_hybrid()` | 3-step: extract params → retrieve policy → LLM review (teknis + policy) + source justification + filtering |
| `generate_source_justifications()` | LLM menjelaskan relevansi setiap source, filter "TIDAK RELEVAN" |

**Question Logging:**
- Setiap pertanyaan yang masuk via `POST /ask` di-log ke file `logs/user_questions.logs` dengan timestamp UTC
- Log directory: `/app/logs` (bind-mounted ke `./output/logs` di host)

**Incremental Sync (`sync_vectorstore`):**

```mermaid
graph TD
    A["Parse sitemap terkini"] --> B["Migrate: add lastmod<br/>(one-time, jika belum ada)"]
    B --> C["Get stored state dari Qdrant<br/>(scroll → {url: lastmod})"]
    C --> D["Bandingkan URL sets"]
    D --> E{Perubahan?}
    E -->|Tidak| F["✅ Semua up-to-date"]
    E -->|Ya| G["Delete points untuk<br/>URL yang berubah/dihapus"]
    G --> H["Scrape URL baru/berubah"]
    H --> I["Embed multi (dense + sparse)"]
    I --> J["Upsert ke Qdrant"]
    J --> K["✅ Sync selesai"]
```

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
| Plain text (tanpa command) | `handle_plain_text()` | Arahkan user gunakan /ask atau /askscript |

**Fitur Teknis:**
- **Progress animation**: Background task dengan `asyncio` yang memutar emoji placeholder + typing indicator (setiap 4 detik)
- **Markdown→HTML converter**: `markdown_to_telegram_html()` — konversi code blocks, bold, italic, strikethrough, headings, links. Code blocks diekstrak ke placeholder terlebih dahulu agar konten (termasuk `#` pada skrip Slurm) tidak terkena transformasi Markdown.
- **HTML splitter**: `split_html_for_telegram()` — otomatis track dan close/reopen HTML tags di batas 4000 char
- **Fallback**: Jika HTML parse gagal, strip semua tags dan kirim plain text
- **Post-init**: Daftarkan command menu ke Telegram API (`set_my_commands`) agar muncul di autocomplete

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

**Mode Operasi:**
- `--mode ingest` — Ingest data saja
- `--mode query` — Benchmark query saja (opsi: `--no-llm`, `--questions N`)
- `--mode all` — Ingest + query
- `--mode cleanup` — Hapus collection benchmark

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
| **Questions** | `services/benchmark_ttft/question.txt` (49 pertanyaan, 4 level) |

**Metrik:** E2E latency per request, P50/P99 latency, throughput (RPS).
**Concurrency levels:** 1, 2, 4, 8, 16 (configurable via env `CONCURRENCY_LEVELS`).
**Default requests:** 49 (configurable via env `NUM_REQUESTS`).

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
        API["rag-api<br/>:8080<br/>FastAPI<br/>Logs: ./output/logs"]
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
| `LLM_API_KEY` | rag-app, rag-api | `EMPTY` | API key vLLM (jika diperlukan autentikasi) |
| `LLM_MODEL_NAME` | rag-app, rag-api, benchmark | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | Nama model LLM |
| `QDRANT_URL` | rag-app, rag-api, benchmark | `http://qdrant:6333` | URL Qdrant REST |
| `QDRANT_API_KEY` | rag-app, rag-api, benchmark | `your-secret-key` | API key Qdrant |
| `TELEGRAM_TOKEN` | telegram-bot | - | Bot token dari BotFather |
| `RAG_API_URL` | telegram-bot, benchmark-ttft | `http://rag-api:8080` | URL RAG API internal |
| `MODEL_NAME` | embedding-service | `BAAI/bge-m3` | Nama model embedding |
| `NUM_REQUESTS` | benchmark-ttft | `49` | Jumlah request per concurrency level |
| `CONCURRENCY_LEVELS` | benchmark-ttft | `1,2,4,8,16` | Level concurrency benchmark |
| `HEALTH_CHECK_RETRIES` | benchmark-ttft | `30` | Retry count health check sebelum benchmark |
| `HEALTH_CHECK_INTERVAL` | benchmark-ttft | `10` | Interval (detik) antara health check retry |
| `RESULT_DIR` | benchmark-ttft | `/app/output/benchmark_ttft` | Direktori output hasil benchmark |

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
    K --> L["Collection: wiki_aleleon_qdrant<br/>dense + text-sparse vectors<br/>+ lastmod metadata"]
```

---

## 8. Hybrid Retrieval Pipeline

```mermaid
graph TD
    Q["User Question"] --> REL{"is_question_relevant()?"}
    REL -->|TIDAK| REJECT["Return: not relevant"]
    REL -->|YA| LOG["Log question to file"]
    LOG --> EMB["embed_query_multi()<br/>→ dense + sparse vectors"]
    EMB --> P1["Prefetch Dense<br/>limit=40, using='dense'"]
    EMB --> P2["Prefetch Sparse<br/>limit=40, using='text-sparse'"]
    P1 --> RRF["RRF Fusion<br/>Reciprocal Rank Fusion<br/>limit=20 candidates"]
    P2 --> RRF
    RRF --> RERANK["ColBERT Reranking<br/>POST /rerank<br/>Sort by ColBERT score"]
    RERANK --> TOP["Top-10 chunks"]
    TOP --> GEN["generate_response()<br/>vLLM chat.completions"]
    GEN --> JUST["generate_source_justifications()<br/>Per-source relevance"]
    JUST --> FILT["Filter 'TIDAK RELEVAN' sources"]
    FILT --> OUT["Return: answer + filtered sources"]
```

**Detail Retrieval Constants:**

| Constant | Value | Deskripsi |
|----------|-------|-----------|
| `TOP_K` | 10 | Jumlah final chunks yang dikirim ke LLM |
| `RERANK_FETCH_MULTIPLIER` | 2 | Over-fetch 2× TOP_K dari Qdrant untuk reranking |
| `DENSE_DIM` | 1024 | Dimensi dense vector dari bge-m3 |

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
podman-compose --profile benchmark run benchmark --mode ingest
podman-compose --profile benchmark run benchmark --mode query --no-llm
podman-compose --profile benchmark run benchmark --mode query --questions 5
podman-compose --profile benchmark run benchmark --mode cleanup

# Benchmark latency (TTFT)
podman-compose --profile infra --profile api up -d
podman-compose --profile benchmark-ttft run benchmark-ttft

# Monitoring
podman-compose --profile monitoring up -d

# Trigger manual sync (setelah API berjalan)
curl -X POST http://localhost:8080/refresh
curl http://localhost:8080/refresh/status

# Stop & cleanup
podman-compose --profile infra --profile api --profile telegram down

# Stop & cleanup (termasuk volumes)
podman-compose --profile infra --profile api --profile telegram down -v

# Rebuild setelah edit code
podman-compose --profile infra --profile api --profile telegram up -d --build
```
