# RAG-Kristo: Retrieval-Augmented Generation for HPC Support

A Retrieval-Augmented Generation (RAG) system that serves as an AI assistant for ALELEON Supercomputer administration. It answers user questions about HPC Conda environments, Slurm job management, and troubleshooting by ingesting wiki pages from **wiki.efisonlt.com** and using locally-hosted LLM inference on **AMD ROCm GPUs** via vLLM.

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                       Pipeline RAG (Podman Container)                         │
│                                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐            │
│  │   INGESTION  │→ │  EMBEDDING   │→ │ RETRIEVAL│→ │  GENERASI   │            │
│  │  (HTML Wiki) │  │ + Penyimpanan│  (Pencarian)│  │   (LLM)     │            │
│  └──────────────┘  └──────────────┘  └──────────┘  └─────────────┘            │
│                                                                               │
│  Fase 1: Ambil & Split   Fase 2: Vektorisasi   Fase 3: Menjawab               │
│                                                                               │
│  Layanan Infra:                                                               │
│  ┌─────────────────┐ ┌──────────────────────┐ ┌──────────┐                    │
│  │embedding-service│ │      vllm-rocm       │ │  qdrant  │                    │
│  │ (BAAI/bge-m3)   │ │ (Qwen3.5-35B-A3B     │ │ (Vektor) │                    │
│  │ Port 8001       │ │  GPTQ-Int4)          │ │ Port 6333│                    │
│  └─────────────────┘ │ Port 8000            │ └──────────┘                    │
│                       └──────────────────────┘                                │
│  Layanan Aplikasi:                                                            │
│  ┌──────────────┐ ┌────────────────────┐ ┌───────────────┐                    │
│  │   rag-app    │ │      rag-api       │ │ telegram-bot  │                    │
│  │ (CLI Interak)│ │ (REST API)         │ │ (/ask,        │                    │
│  │              │ │ Port 8080          │ │  /askscript)  │                    │
│  │              │ │ /ask               │ │               │                    │
│  │              │ │ /review-script     │ │               │                    │
│  │              │ │ /refresh           │ │               │                    │
│  │              │ │ (Mount: ./logs)    │ │               │                    │
│  └──────────────┘ └────────────────────┘ └───────────────┘                    │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
rag-for-l1-aleleon-hpc-support/
├── compose.yml                  # Podman multi-container orchestration
├── README.md                    # This file
│
├── services/                    # Microservices (masing-masing punya Dockerfile)
│   ├── rag-app/
│   │   ├── Dockerfile.rag-app
│   │   ├── rag_app.py           # Core RAG logic (ingestion, retrieval, generation)
│   │   └── rag_api.py           # FastAPI REST API (/ask, /review-script, /refresh)
│   ├── embedding/
│   │   ├── Dockerfile.embedding
│   │   └── embedding_api.py     # Embedding REST API (BAAI/bge-m3)
│   ├── telegram-bot/
│   │   ├── Dockerfile.telegram
│   │   └── telegram_bot.py      # Telegram Bot (/ask, /askscript commands)
│   ├── benchmark_retrieval/
│   │   ├── Dockerfile.benchmark
│   │   └── benchmark_retrieval.py  # Dense/Sparse/Hybrid retrieval benchmark
│   ├── benchmark_ttft/
│   │   ├── Dockerfile.rag-bench
│   │   ├── run-rag-bench.py     # TTFT & latency benchmark
│   │   ├── run-vllm-bench.sh    # vLLM benchmark script
│   │   └── question.txt         # Benchmark questions
│   ├── promtail/
│   │   └── config.yml           # Promtail log scraping config
│   └── Dockerfile.rocm          # Container image for AMD ROCm GPUs
│
├── docs/                        # Dokumentasi
│   ├── HOW_IT_WORKS.md          # Detailed explanation of the RAG pipeline
│   ├── MICROSERVICES_ARCHITECTURE.md  # Microservices architecture doc
│   ├── MONITORING_README.md     # Monitoring setup guide
│   └── plans/
│       └── rag-architecture-plan-updated.md
│
├── scripts/                     # Utility scripts
│   ├── benchmark_chart_generator.py   # Generate charts from benchmark results
│   ├── embedding_chart_generator.py   # Generate charts for embedding benchmarks
│   ├── debug_parse_sitemap.py         # Debug sitemap parsing
│   ├── extract_qa.py                  # Extract Q&A pairs
│   ├── inspect_chroma.py              # Inspect vector store contents
│   └── parse_tokens.py                # Token parsing utility
│
├── data/                        # Data & benchmark questions
│   └── question-benchmark-google.txt  # 69 test questions (5 difficulty levels)
│
├── tests/
│   └── test_services.py
│
├── archive/                     # Legacy/unused code (gitignored)
│   └── rag_slurm_vllm.py       # Standalone RAG script (superseded by services/)
│
└── output/                      # Experiment results (gitignored)
    ├── benchmark/               # Retrieval benchmark results
    ├── benchmark_embedding_*/   # Embedding benchmark results
    ├── benchmark_ttft/          # TTFT benchmark results
    └── logs/                    # User question logs
```

## How It Works

The application runs in three phases:

### Phase 1 — Data Ingestion (Incremental Sync)

1. **Sitemap Parsing & Incremental Sync** — Fetches the wiki sitemap XML and extracts all page URLs along with their `<lastmod>` (last modified) timestamp. The system compares this with metadata in the Qdrant database to perform an **incremental update** (only scraping new or updated pages, and deleting removed pages) instead of a full rebuild.
2. **HTML Fetching** — Downloads the required wiki pages and extracts the main content from `<div id="mw-content-text">` using BeautifulSoup (preserving raw HTML).
3. **Structure-Based Splitting** — Uses `HTMLSectionSplitter` to split content by heading tags (`h1`, `h2`, `h3`), preserving document structure.
4. **Fallback Splitting** — Chunks larger than 4500 characters are further split using `RecursiveCharacterTextSplitter` (chunk size: 4500, overlap: 900).
5. **Metadata Enrichment** — Each chunk gets labeled with `[Sumber: <page_title>] [Section: <heading>]` prefix for source attribution.
6. **Embedding via API** — Converts each chunk into a vector using `BAAI/bge-m3` served by `embedding-service` (batched, 32 texts per request).
7. **Vector Storage** — Stores embeddings in a persistent Qdrant collection, backed by a Podman named volume (`qdrant-data:/qdrant/storage`).

### Phase 2 — LLM Setup (vLLM on ROCm)

8. **Model Loading** — Loads `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` onto the AMD GPU using vLLM with these settings:

   #### vLLM Server Configuration (compose.yml)

   | Parameter | Value | Reason |
   |---|---|---|
   | `--dtype` | `float16` | Inference precision for quantized model |
   | `--enforce-eager` | True | Avoids CUDAGraph issues on ROCm/RDNA4 |
   | `--gpu-memory-utilization` | 0.99 | Use 99% of available VRAM |
   | `--max-model-len` | 262144 | Full 256K context window for large prompts |
   | `--max-num-seqs` | 16 | Max concurrent sequences |
   | `--tensor-parallel-size` | 1 | Single GPU inference |
   | `--enable-auto-tool-choice` | True | Enable tool/function calling support |
   | `--tool-call-parser` | `qwen3_coder` | Tool call parser for Qwen3 Coder |
   | `--reasoning-parser` | `qwen3` | Reasoning parser for Qwen3 |
   | `--enable-prefix-caching` | True | Cache common prefixes for faster inference |

   #### `/ask` — RAG Question Answering (`generate_response`)

   | Parameter | Value | Reason |
   |---|---|---|
   | `max_tokens` | 8192 | Max response length for RAG answers |
   | `temperature` | 0.3 | Lower randomness for factual RAG answers |
   | `top_p` | 0.9 | Nucleus sampling |
   | `top_k` | 20 | Top-k sampling constraint |
   | `presence_penalty` | 1.5 | Discourage repetition |
   | `enable_thinking` | False | Non-thinking mode for direct answers |

   #### `/askscript` — Script Review (`review_script_hybrid`)

   | Parameter | Value | Reason |
   |---|---|---|
   | `max_tokens` | 4096 | Max response length for script reviews |
   | `temperature` | 0.2 | Very low randomness for precise analysis |
   | `top_p` | 0.9 | Nucleus sampling |
   | `enable_thinking` | False | Non-thinking mode for direct output |

   #### Supporting LLM Calls

   | Function | `max_tokens` | `temperature` | Purpose |
   |---|---|---|---|
   | `is_question_relevant` | 10 | 0.0 | Relevance filter (YA/TIDAK) before embedding |
   | `generate_source_justifications` | 1024 | 0.1 | "Why This Source" justification per document |
   | `extract_resource_params` | 512 | 0.0 | Parse #SBATCH params from scripts as JSON |

### Phase 3 — Question Answering (RAG Chain)

9. **Retrieval** — For each user question, the retriever finds the **top-10** most semantically similar chunks from Qdrant via cosine similarity.
10. **Prompt Construction** — Builds OpenAI messages with system instructions (Bahasa Indonesia), retrieved context, and the user question.
11. **Generation** — vLLM generates an answer grounded in the retrieved documents using the OpenAI-compatible API.
12. **Anti-Hallucination** — The system prompt enforces 11 strict rules (0-10):
    - Answer ONLY from documents; quote commands exactly
    - Preserve exact numbers, versions, and specs
    - Never substitute commands (e.g., don't replace `source activate` with `conda activate`)
    - Distinguish "minimal" vs "maksimal"
    - Respond "Saya tidak menemukan informasi tersebut di sistem." when info is not in docs
    - Watch for LEGACY labels — do not apply Mk.III info to Mk.V
    - Always answer with at least 2 sentences; never return an empty response

### Multiprocessing Guard

The `if __name__ == '__main__'` guard is **required** because vLLM v1 uses `spawn` multiprocessing. Without it, the child process would re-execute the entire script and crash.

## Requirements

### Hardware

| Component | Minimum | Tested On |
|---|---|---|
| GPU | AMD GPU with ROCm support | Radeon AI PRO R9700 (gfx1201, 32GB VRAM) |
| RAM | 16GB system RAM | 48GB DDR5 |
| CPU | Any x86_64 | Intel i7-12700K/Ryzen 7 9800X3D |
| ROCm | 6.0+ | 7.0 (HIP 7.0.51831) |

### Software

- Podman (with podman-compose) or Docker
- ROCm drivers installed on host
- ~15GB disk for the container image
- ~25GB disk for the Qwen3.5-35B-A3B-GPTQ-Int4 model weights (auto-downloaded)

## Quick Start (Podman Compose)

### 1. Start Infrastructure Services

```bash
# Start embedding, vLLM, and Qdrant
podman-compose --profile infra up -d
```

### 2. Run the Interactive RAG CLI

```bash
podman-compose --profile rag-app run rag-app
```

### 3. Full Stack (API + Telegram Bot)

```bash
podman-compose --profile infra --profile api --profile telegram up -d
```

### 4. Run Benchmarks

```bash
# Retrieval benchmark (Dense vs Sparse vs Hybrid)
podman-compose --profile infra up -d
podman-compose --profile benchmark run benchmark

# TTFT / Latency benchmark
podman-compose --profile infra --profile api up -d
podman-compose --profile benchmark-ttft run benchmark-ttft
```

### 5. Stop & Cleanup

```bash
podman-compose --profile infra --profile api --profile telegram down

# Remove volumes too
podman-compose --profile infra --profile api --profile telegram down -v
```

### 6. Rebuild After Code Changes

```bash
podman-compose --profile infra --profile api --profile telegram up -d --build
```

## API Endpoints

The RAG API (`rag-api` service) exposes the following endpoints on port **8080**:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ask` | Send a question, get a RAG-generated answer with sources |
| `POST` | `/review-script` | Review a Bash/Slurm script (LLM analysis + HPC policy validation) |
| `POST` | `/refresh` | Trigger incremental sitemap sync to Qdrant (background) |
| `GET` | `/refresh/status` | Check status of the last sync |
| `GET` | `/health` | Health check |

**Example usage:**

```bash
# Ask a question
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Bagaimana cara membuat conda environment?"}'

# Review a script
curl -X POST http://localhost:8080/review-script \
  -H "Content-Type: application/json" \
  -d '{"script": "#!/bin/bash\n#SBATCH --mem=64G\nsrun gmx_mpi mdrun"}'

# Trigger sync
curl -X POST http://localhost:8080/refresh

# Check sync status
curl -X GET http://localhost:8080/refresh/status
```

## Compose Profiles

| Profile | Services | Description |
|---|---|---|
| `infra` | embedding-service, vllm-rocm, qdrant | Full backend stack |
| `embedding` | embedding-service | Embedding service only |
| `vllm` | vllm-rocm | LLM service only |
| `qdrant` | qdrant | Vector DB only |
| `rag-app` | rag-app | Interactive CLI |
| `api` | rag-api | REST API (port 8080) |
| `telegram` | telegram-bot | Telegram Bot |
| `benchmark` | benchmark | Retrieval benchmark |
| `benchmark-ttft` | benchmark-ttft | TTFT/Latency benchmark |
| `monitoring` | promtail | Promtail log scraping |

## Test Questions & Expected Results

The benchmark suite includes **69 test questions** across 5 difficulty levels:

| Level | Count | Description |
|---|---|---|
| **Level 1** — Direct Facts | 20 | RAM capacity, walltime limits, GPU types, portal URLs, conda commands, pricing, quotas, PKSPIAS, SFTP limits, OS version, Slurm version, job cancellation, storage limits, backup policy, support email, GROMACS binary, 2FA authenticators, squeue, pip support, GCC module |
| **Level 2** — Multi-Chunk | 10 | GPU partition comparison, JupyterLab batch job, squeue status meanings, institutional account limits, password change portal, file upload troubleshooting, AMD compiler modules, login node restrictions, VPN disconnect behavior, Core Hour types |
| **Level 3** — Reasoning / Deduction & Troubleshooting | 10 | Memory allocation calculation, Slurm syntax error diagnosis, QOS CPU limits, walltime limits, Slurm Array configuration, hybrid MPI+OpenMP core counting, high-memory partition selection, GPU Hour quota estimation, core rounding behavior, storage optimization |
| **Level 4** — Anti-Hallucination | 15 | SSD storage (not in docs), MATLAB (not in docs), storage fines, R packages, VPN bandwidth, Python reset, ANSYS Fluent, VSCode SSH, foreign currency, default password, CTO name, wiki editing, PKSPIAS cancellation, UPS battery, AutoGluon |
| **Level 5** — Additional Questions | 14 | Conda env creation, Jupyter with custom env, Python version, Mamba activation, pyload module, module listing, GPU partition, support email, office hours, computation methods, Job Composer vs terminal, full setup procedure, squeue statuses, EWS Jupyter form |

## Configuration

### Changing the Knowledge Base

The system loads documents dynamically from the wiki sitemap. To change the source, update the sitemap URL in `services/rag-app/rag_app.py`:

```python
splits = load_wiki_documents(
    sitemap_url="https://wiki.efisonlt.com/sitemap/sitemap-wiki.efisonlt.com-NS_0-0.xml",
    requests_per_second=2,
)
```

### Changing the LLM Model

Edit the `--model` argument in [compose.yml](compose.yml) (service `vllm-rocm`).

### Changing the Embedding Model

Edit `MODEL_NAME` in [compose.yml](compose.yml) (service `embedding-service`).

### Adjusting for Different GPUs

Set `HSA_OVERRIDE_GFX_VERSION` to match your GPU:

| GPU | Architecture | HSA_OVERRIDE_GFX_VERSION |
|---|---|---|
| RX 6800/6900 | gfx1030 (RDNA2) | 10.3.0 |
| RX 7900 XTX | gfx1100 (RDNA3) | 11.0.0 |
| RX 8060S | gfx1151 (RDNA3.5) | 11.5.1 |
| RX 9070 XT | gfx1201 (RDNA4) | 12.0.1 |

Check your GPU architecture:

```bash
rocminfo | grep "Name:" | grep "gfx"
```

### Tuning Chunk & Retrieval Parameters

| Parameter | Default | Effect |
|---|---|---|
| `chunk_size` | 4500 | Max chars per chunk before fallback splitting |
| `chunk_overlap` | 900 | Overlap between fallback chunks to preserve context |
| `k` (similarity_search) | 10 | Number of chunks retrieved per question (more = richer context, larger prompt) |
| `requests_per_second` | 2 | Rate limit for fetching wiki pages |

## Key Dependencies

Dependencies are managed per-service via Dockerfiles:

- **langchain-core** / **langchain-community** — Document loading and retriever utilities
- **langchain-text-splitters** — `HTMLSectionSplitter` and `RecursiveCharacterTextSplitter`
- **langchain-qdrant** — Qdrant vector store integration via LangChain
- **qdrant-client** — Qdrant Python client for collection management
- **openai** — OpenAI-compatible client for vLLM
- **FastAPI** / **uvicorn** — REST API framework
- **requests** — HTTP calls to embedding-service and health checks
- **beautifulsoup4** / **lxml** — HTML parsing for wiki page extraction
- **vLLM** — High-performance LLM inference engine (inside the vllm-rocm container)
- **python-telegram-bot** — Telegram Bot integration

## Features

### Persistent Question Logging

Every user question sent to the `/ask` endpoint is automatically logged with a UTC timestamp. These logs are saved persistently to a mounted volume (`./output/logs` on the host mapped to `/app/logs` in the container) in the `user_questions.logs` file. This is useful for auditing and understanding user behavior.

### Incremental Sitemap Sync

The `rag-api` service automatically detects changes to the wiki source by reading the `<lastmod>` tags in the `sitemap.xml`.
- **Startup Sync:** When the container starts, it automatically syncs with the wiki, making sure Qdrant is up-to-date.
- **Manual Sync via API:** You can manually trigger a background sync without restarting the container:
  ```bash
  # Start the sync in the background
  curl -X POST http://localhost:8080/refresh

  # Check the status of the sync
  curl -X GET http://localhost:8080/refresh/status
  ```
The sync process is smart: it skips unchanged pages, deletes pages removed from the sitemap, and only scrapes/embeds newly added or recently modified pages.

### Script Review (Hybrid)

The `/review-script` endpoint uses a hybrid approach:
- **LLM Technical Analysis** — Analyzes the script for syntax errors, best practices, and optimization opportunities.
- **RAG Policy Validation** — If the script contains resource parameters (`#SBATCH`), retrieves relevant HPC policies from the knowledge base to validate against institutional limits and quotas.

### Relevance Filtering

Both `/ask` and `/review-script` endpoints include a lightweight relevance check (`is_question_relevant`) that filters out off-topic questions before consuming expensive LLM inference cycles.

## Further Documentation

- [HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) — Detailed explanation of the RAG pipeline
- [MICROSERVICES_ARCHITECTURE.md](docs/MICROSERVICES_ARCHITECTURE.md) — Microservices architecture documentation
- [MONITORING_README.md](docs/MONITORING_README.md) — Monitoring setup with Promtail

## License

Private project — EFISON HPC Support.

## Author

Kristo Nova (krisostomus.nova.r@efisonlt.com)
