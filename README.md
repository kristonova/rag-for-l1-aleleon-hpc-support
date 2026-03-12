# RAG-Kristo: Retrieval-Augmented Generation for HPC Support

A Retrieval-Augmented Generation (RAG) system that serves as an AI assistant for ALELEON Supercomputer administration. It answers user questions about HPC Conda environments, Slurm job management, and troubleshooting by ingesting wiki pages from **wiki.efisonlt.com** and using locally-hosted LLM inference on **AMD ROCm GPUs** via vLLM.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Pipeline RAG (Podman Container)                       │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐       │
│  │   INGESTION  │→ │  EMBEDDING   │→ │ RETRIEVAL│→ │  GENERASI   │       │
│  │  (HTML Wiki) │  │ + Penyimpanan│  (Pencarian)│  │   (LLM)     │       │
│  └──────────────┘  └──────────────┘  └──────────┘  └─────────────┘       │
│                                                                          │
│  Fase 1: Ambil & Split   Fase 2: Vektorisasi   Fase 3: Menjawab          │
│                                                                          │
│  Layanan:                                                                │
│  ┌─────────────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────────┐       │
│  │embedding-service│ │  vllm-rocm  │ │ chromadb │ │   rag-app    │       │
│  │ (BAAI/bge-m3)   │ │ (Qwen3.5)   │ │ (Vektor) │ │ (Orkestrator)│       │
│  │ Port 8001       │ │ Port 8000   │ │ Port 8002│ │              │       │
│  └─────────────────┘ └─────────────┘ └──────────┘ └──────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
rag-for-l1-aleleon-hpc-support/
├── services/
│   ├── embedding/
│   │   ├── Dockerfile.embedding
│   │   └── embedding_api.py
│   └── rag-app/
│       ├── Dockerfile.rag-app
│       └── rag_app.py
├── compose.yml                  # Podman multi-container orchestration
├── Dockerfile.rocm              # Container image for AMD ROCm GPUs (legacy)
├── rag_slurm_vllm.py            # Standalone script (legacy)
├── HOW IT WORKS.md              # Detailed explanation document
├── inspect_chroma.py            # Inspect ChromaDB contents
├── debug_parse_sitemap.py       # Debug sitemap parsing
├── pyproject.toml               # Python project metadata & dependencies
├── tests/
│   └── test_services.py
└── README.md                    # This file
```

## How It Works

The application runs in three phases:

### Phase 1 — Data Ingestion (Wiki Sitemap → HTML Structural Splitting)

1. **Sitemap Parsing** — Fetches the wiki sitemap XML from `https://wiki.efisonlt.com/sitemap/sitemap-wiki.efisonlt.com-NS_0-0.xml` and extracts all page URLs.
2. **HTML Fetching** — Downloads each wiki page and extracts the main content from `<div id="mw-content-text">` using BeautifulSoup (preserving raw HTML).
3. **Structure-Based Splitting** — Uses `HTMLSectionSplitter` to split content by heading tags (`h1`, `h2`, `h3`), preserving document structure.
4. **Fallback Splitting** — Chunks larger than 4500 characters are further split using `RecursiveCharacterTextSplitter` (chunk size: 4500, overlap: 900).
5. **Metadata Enrichment** — Each chunk gets labeled with `[Sumber: <page_title>] [Section: <heading>]` prefix for source attribution.
6. **Embedding via API** — Converts each chunk into a vector using `BAAI/bge-m3` served by `embedding-service` (batched, 32 texts per request).
7. **Vector Storage** — Stores embeddings in a persistent ChromaDB directory mounted via Podman volume.

### Phase 2 — LLM Setup (vLLM on ROCm)

8. **Model Loading** — Loads `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` onto the AMD GPU using vLLM with these settings:

   | Parameter | Value | Reason |
   |---|---|---|
   | `gpu_memory_utilization` | 0.80 | Use 80% of available VRAM |
   | `enforce_eager` | True | Avoids CUDAGraph issues on ROCm/RDNA4 |
    | `max_model_len` | 131072 | Full 128K context window for large prompts |
    | `temperature` | 0.3 | Lower randomness for RAG |
    | `top_p` | 0.9 | Nucleus sampling |
    | `top_k` | 20 | Top-k sampling constraint |
    | `max_tokens` | 32768 | Max response length |

### Phase 3 — Question Answering (RAG Chain)

9. **Retrieval** — For each user question, the retriever finds the **top-10** most semantically similar chunks from ChromaDB.
10. **Prompt Construction** — Builds OpenAI messages with system instructions (Bahasa Indonesia), retrieved context, and the user question.
11. **Generation** — vLLM generates an answer grounded in the retrieved documents using the OpenAI-compatible API.
12. **Anti-Hallucination** — The system prompt enforces 11 strict rules (0-10):
    - Answer ONLY from documents; quote commands exactly
    - Preserve exact numbers, versions, and specs
    - Never substitute commands (e.g., don't replace `source activate` with `conda activate`)
    - Distinguish "minimal" vs "maksimal"
    - Respond "Saya tidak menemukan informasi tersebut." when info is not in docs

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

- Podman or Docker
- ROCm drivers installed on host
- ~15GB disk for the container image
- ~25GB disk for the Qwen3.5-35B-A3B-GPTQ-Int4 model weights (auto-downloaded)

## Quick Start (Podman Compose)

### 1. Start Services

```bash
podman-compose --profile backend up -d embedding-service chromadb vllm-rocm
```

### 2. Run the RAG App

```bash
podman-compose --profile rag-app run --rm rag-app
```

### 3. Optional: Start All in One

```bash
podman-compose --profile test up
```

## Dockerfile.rocm — Build Details (Legacy)

The Dockerfile uses `rocm/vllm-dev:nightly` as the base image (includes PyTorch ROCm + vLLM). Key design decisions:

| Challenge | Solution |
|---|---|
| `pip install chromadb` pulls CUDA torch | Install chromadb with `--no-deps`, then add safe dependencies manually |
| `pip install sentence-transformers` pulls CUDA torch | Install with `--no-deps`, add sub-dependencies separately |
| `pip install onnxruntime` pulls CUDA torch | Install with `--no-deps` |
| LangChain 1.x API changes | Uses direct Python functions instead of chain helpers |
| Wiki HTML parsing | Installs `lxml` and `beautifulsoup4` for `HTMLSectionSplitter` |
| Build-time verification | `assert torch.version.hip is not None` ensures PyTorch ROCm survives all pip installs |

## Test Questions & Expected Results

The script includes **23 test questions** across 4 difficulty levels:

| Level | Count | Tests |
|---|---|---|
| **Level 1** — Direct Facts | 9 | Conda env creation, Jupyter setup, Anaconda version, Mamba activation, pyload module, GPU partition, support email, office hours |
| **Level 2** — Multi-Chunk | 5 | Computation methods overview, Job Composer vs terminal Slurm, full conda+pyload setup, squeue statuses, EWS Jupyter form |
| **Level 3** — Reasoning / Deduction | 6 | TensorFlow CUDA version, Anaconda version recommendation, .ipynb batch prep, bash -l header reasoning, multi-GPU packages, storage cleanup |
| **Level 4** — Anti-Hallucination | 3 | Pricing (not in docs), Docker in conda (not in docs), max GPU limit (not in docs) |

## Configuration

### Changing the Knowledge Base

The system loads documents dynamically from the wiki sitemap. To change the source, update the sitemap URL in the script:

```python
splits = load_wiki_documents(
    sitemap_url="https://wiki.efisonlt.com/sitemap/sitemap-wiki.efisonlt.com-NS_0-0.xml",
    requests_per_second=2,
)
```

### Changing the LLM Model

Edit the `--model` argument in [compose.yml](compose.yml#L33) (service `vllm-rocm`).

### Changing the Embedding Model

Edit `MODEL_NAME` in [compose.yml](compose.yml#L28) (service `embedding-service`).

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
| `search_kwargs["k"]` | 10 | Number of chunks retrieved per question (more = richer context, larger prompt) |
| `requests_per_second` | 2 | Rate limit for fetching wiki pages |

## Dependencies

Managed via Poetry (`pyproject.toml`):

- **langchain-core** / **langchain-community** — Document loading and retriever utilities
- **langchain-text-splitters** — `HTMLSectionSplitter` and `RecursiveCharacterTextSplitter`
- **langchain-chroma** — ChromaDB vector store integration
- **openai** — OpenAI-compatible client for vLLM
- **requests** — HTTP calls to embedding-service and health checks
- **beautifulsoup4** / **lxml** — HTML parsing for wiki page extraction
- **vLLM** — High-performance LLM inference engine (inside the vllm-rocm container)

## License

Private project — EFISON HPC Support.

## Author

Kristo Nova (krisostomus.nova.r@efisonlt.com)
