# RAG-Kristo: Retrieval-Augmented Generation for HPC Support

A Retrieval-Augmented Generation (RAG) system that serves as an AI assistant for ALELEON Supercomputer administration. It answers user questions about HPC Conda environments, Slurm job management, and troubleshooting by ingesting wiki pages from **wiki.efisonlt.com** and using locally-hosted LLM inference on **AMD ROCm GPUs** via vLLM.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          RAG Pipeline                                │
│                                                                      │
│  ┌───────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │  Wiki Sitemap     │─▶│ HTMLSectionSplit │─▶│  Embedding Model. │  │
│  │  Loader (XML→HTML)│  │ (h1/h2/h3)       │  │  (multilingual-   │  │
│  │  + BeautifulSoup  │  │ fallback: 4500ch │  │   e5-large)       │  │
│  └───────────────────┘  └──────────────────┘  └────────┬──────────┘  │
│                                                         │            │
│                                               ┌─────────▼────────┐   │
│                                               │  ChromaDB        │   │
│                                               │  (Vector Store)  │   │
│                                               └─────────┬────────┘   │
│                                                         │            │
│  ┌──────────────────┐    ┌────────────────┐    ┌────────▼────────┐   │
│  │  vLLM Engine     │◀───│  LangChain     │◀───│  Retriever      │   │
│  │  (Qwen2.5-7B     │    │  RAG Chain     │    │  (Top-15 chunks)│   │
│  │   Instruct)      │    │  (ChatML)      │    └─────────────────┘   │
│  │  ROCm / HIP      │    └────────────────┘                         │
│  └──────────────────┘                                                │
└──────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
rag-for-l1-aleleon-hpc-support/
├── rag_slurm_vllm.py            # Main RAG application script
├── Dockerfile.rocm              # Container image for AMD ROCm GPUs
├── compose.yml           # Multi-container orchestration (optional)
├── pyproject.toml               # Python project metadata & dependencies
├── poetry.lock                  # Locked dependency versions
├── HOW IT WORKS.md              # Detailed explanation document
├── afo_tune_device_0_full.csv   # GPU tuning data
├── wiki/                        # Local copies of wiki pages (reference)
│   ├── Komputasi_Python_dengan_Conda_Environment_User.txt
│   ├── komputasi_python_venv_user.txt
│   ├── metode_komputasi_efison.txt
│   ├── mpi_aleleon_superkomputer.txt
│   ├── spesifikasi_aleleon.txt
│   └── tutorial_akun_trial_a6.txt
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
6. **Embedding** — Converts each chunk into a vector using `intfloat/multilingual-e5-large` (~1.2GB multilingual model, runs on CPU/GPU).
7. **Vector Storage** — Stores embeddings in an in-memory ChromaDB instance for fast similarity search.

### Phase 2 — LLM Setup (vLLM on ROCm)

8. **Model Loading** — Loads `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` onto the AMD GPU using vLLM with these settings:

   | Parameter | Value | Reason |
   |---|---|---|
   | `gpu_memory_utilization` | 0.80 | Use 80% of available VRAM |
   | `enforce_eager` | True | Avoids CUDAGraph issues on ROCm/RDNA4 |
   | `max_model_len` | 32768 | Full 32K context window for large prompts |
   | `temperature` | 0.6 | Balanced factual/creative answers |
   | `top_p` | 0.95 | Nucleus sampling |
   | `top_k` | 20 | "INSERT_REASON" |
   | `max_new_tokens` | 32768 | Max response length |

### Phase 3 — Question Answering (RAG Chain)

9. **Retrieval** — For each user question, the retriever finds the **top-15** most semantically similar chunks from ChromaDB.
10. **Prompt Construction** — Builds a ChatML-formatted prompt (`<|im_start|>` / `<|im_end|>`) with system instructions (in Bahasa Indonesia), retrieved context, and the user question.
11. **Generation** — vLLM generates an answer grounded in the retrieved documents.
12. **Anti-Hallucination** — The system prompt enforces 7 strict rules:
    - Answer ONLY from documents; quote commands exactly
    - Preserve exact numbers, versions, and specs
    - Never substitute commands (e.g., don't replace `source activate` with `conda activate`)
    - Distinguish "minimal" vs "maksimal"
    - Respond "Saya tidak menemukan informasi tersebut di sistem" when info is not in docs

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

## Quick Start

### 1. Build the Container

```bash
podman build -t rag-kristo-rocm -f Dockerfile.rocm .
```

### 2. Run

```bash
podman run -it --rm \
    --cap-add=SYS_PTRACE \
    --device=/dev/kfd \
    --device=/dev/dri \
    -v $(pwd):/app:Z \
    -v ~/.cache/huggingface:/root/.cache/huggingface:Z \
    --group-add keep-groups \
    rag-kristo-rocm \
    bash
```

### 3. Run Interactively (mount local files)

```bash
podman run \
  --cap-add=SYS_PTRACE \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add keep-groups \
  -v $(pwd):/app:Z \
  -v ~/.cache/huggingface:/root/.cache/huggingface:Z \
  -it rag-kristo-rocm bash
```

Then inside the container:

```bash
python rag_slurm_vllm.py
```

## Dockerfile.rocm — Build Details

The Dockerfile uses `rocm/vllm-dev:nightly` as the base image (includes PyTorch ROCm + vLLM). Key design decisions:

| Challenge | Solution |
|---|---|
| `pip install chromadb` pulls CUDA torch | Install chromadb with `--no-deps`, then add safe dependencies manually |
| `pip install sentence-transformers` pulls CUDA torch | Install with `--no-deps`, add sub-dependencies separately |
| `pip install onnxruntime` pulls CUDA torch | Install with `--no-deps` |
| LangChain 1.x API changes | Uses `langchain-classic` for `create_retrieval_chain` / `create_stuff_documents_chain` |
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

Edit the `model` parameter in the script:

```python
llm = VLLM(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",  # Change this
    ...
)
```

### Changing the Embedding Model

Edit the `model_name` parameter:

```python
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")  # Change this
```

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
| `search_kwargs["k"]` | 15 | Number of chunks retrieved per question (more = richer context, larger prompt) |
| `requests_per_second` | 2 | Rate limit for fetching wiki pages |

## Dependencies

Managed via Poetry (`pyproject.toml`):

- **langchain** / **langchain-core** / **langchain-community** — RAG chain orchestration
- **langchain-classic** — Provides `create_retrieval_chain` and `create_stuff_documents_chain` (LangChain 1.x API)
- **langchain-text-splitters** — `HTMLSectionSplitter` and `RecursiveCharacterTextSplitter`
- **langchain-huggingface** — HuggingFace embedding integration
- **langchain-chroma** — ChromaDB vector store integration
- **sentence-transformers** — Embedding model runtime (`intfloat/multilingual-e5-large`)
- **beautifulsoup4** / **lxml** — HTML parsing for wiki page extraction
- **vLLM** — High-performance LLM inference engine (included in base Docker image)

## License

Private project — EFISON HPC Support.

## Author

Kristo Nova (krisostomus.nova.r@efisonlt.com)
