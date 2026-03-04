# RAG-Kristo: Retrieval-Augmented Generation for HPC Support

A Retrieval-Augmented Generation (RAG) system that serves as an AI assistant for ALELEON Supercomputer administration. It answers user questions about HPC specifications, partitions, and troubleshooting using locally-hosted LLM inference on **AMD ROCm GPUs** via vLLM.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                              │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐  │
│  │  TextLoader  │───▶│  Chunking    │───▶│  Embedding Model    │  │
│  │  (.txt file) │    │  (1000 chars)│    │  (all-MiniLM-L6-v2) │  │
│  └─────────────┘    └──────────────┘    └────────┬────────────┘  │
│                                                   │              │
│                                         ┌─────────▼──────────┐   │
│                                         │  ChromaDB (Vector) │   │
│                                         └─────────┬──────────┘   │
│                                                   │              │
│  ┌──────────────────┐    ┌────────────┐    ┌──────▼──────────┐   │
│  │  vLLM Engine     │◀───│  LangChain │◀───│  Retriever      │   │
│  │  (Qwen2.5-7B)    │    │  RAG Chain │    │  (Top-5 chunks) │   │
│  │  ROCm / HIP      │    └────────────┘    └─────────────────┘   │
│  └──────────────────┘                                            │
└──────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
rag-kristo/
├── rag_slurm_vllm.py.py       # Main RAG application script
├── spesifikasi_aleleon.txt     # Knowledge base document (ALELEON specs)
├── Dockerfile.rocm             # Container image for AMD ROCm GPUs
├── docker-compose.yml          # Multi-container orchestration (optional)
├── pyproject.toml              # Python project metadata & dependencies
├── poetry.lock                 # Locked dependency versions
└── README.md                   # This file
```

## How It Works

The application runs in three phases:

### Phase 1 — Data Ingestion

1. **Document Loading** — Reads `spesifikasi_aleleon.txt` (ALELEON Supercomputer specs) using LangChain's `TextLoader`.
2. **Text Chunking** — Splits the document into chunks of 1000 characters with 200-character overlap, using Markdown-aware separators (`---`, `## `, `\n\n`) to avoid breaking tables mid-row.
3. **Embedding** — Converts each chunk into a vector using `all-MiniLM-L6-v2` (runs on CPU, ~80MB model).
4. **Vector Storage** — Stores embeddings in an in-memory ChromaDB instance for fast similarity search.

### Phase 2 — LLM Setup (vLLM on ROCm)

5. **Model Loading** — Loads `Qwen/Qwen2.5-Coder-7B-Instruct` onto the AMD GPU using vLLM with these settings:

   | Parameter | Value | Reason |
   |---|---|---|
   | `gpu_memory_utilization` | 0.90 | Use 90% of 16GB VRAM |
   | `enforce_eager` | True | Avoids CUDAGraph issues on RDNA4 |
   | `max_model_len` | 4096 | Limits KV cache to save VRAM |
   | `temperature` | 0.1 | Factual, low-creativity answers |
   | `max_new_tokens` | 512 | Max response length |

### Phase 3 — Question Answering (RAG Chain)

6. **Retrieval** — For each user question, the retriever finds the top-5 most semantically similar chunks from ChromaDB.
7. **Prompt Construction** — Builds a ChatML-formatted prompt with system instructions, retrieved context, and the user question.
8. **Generation** — vLLM generates an answer grounded in the retrieved documents.
9. **Anti-Hallucination** — The system prompt instructs the model to respond "Saya tidak menemukan informasi tersebut di sistem" when the answer is not in the documents.

### Multiprocessing Guard

The `if __name__ == '__main__'` guard is **required** because vLLM v1 uses `spawn` multiprocessing. Without it, the child process would re-execute the entire script and crash.

## Requirements

### Hardware

| Component | Minimum | Tested On |
|---|---|---|
| GPU | AMD GPU with ROCm support | R97000 (gfx1201, 32GB VRAM) |
| RAM | 16GB system RAM | 48GB DDR5 |
| CPU | Any x86_64 | Intel i7-12700K |
| ROCm | 6.0+ | 7.0 (HIP 7.0.51831) |

### Software

- Podman or Docker
- ROCm drivers installed on host
- ~15GB disk for the container image
- ~15GB disk for the Qwen2.5-7B model weights (auto-downloaded)

## Quick Start

### 1. Build the Container

```bash
podman build -f Dockerfile.rocm -t rag-kristo-rocm --no-cache .
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
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -v $(pwd):/app:Z \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HSA_OVERRIDE_GFX_VERSION=12.0.1 \
  -it rag-kristo-rocm bash
```

Then inside the container:

```bash
python rag_slurm_vllm.py.py
```

## Dockerfile.rocm — Build Details

The Dockerfile uses `rocm/vllm-dev:nightly` as the base image (includes PyTorch ROCm + vLLM). Key design decisions:

| Challenge | Solution |
|---|---|
| `pip install chromadb` pulls CUDA torch | Install chromadb with `--no-deps`, then add safe dependencies manually |
| `pip install sentence-transformers` pulls CUDA torch | Install with `--no-deps`, add sub-dependencies separately |
| `pip install onnxruntime` pulls CUDA torch | Install with `--no-deps` |
| Build-time verification | `assert torch.version.hip is not None` ensures PyTorch ROCm survives all pip installs |

## Test Questions & Expected Results

The script includes 11 test questions across 4 difficulty levels:

| Level | Questions | Tests |
|---|---|---|
| **Level 1** — Direct Facts | RAM, GPU specs, OS | Exact info retrieval from a single chunk |
| **Level 2** — Multi-Chunk | Partition comparisons, GPU overview | Combining info from multiple chunks |
| **Level 3** — Reasoning | 400GB RAM job, Docker support, Python 2 | Inference and deduction from context |
| **Level 4** — Anti-Hallucination | Pricing, AMD MI300X | Model must refuse to answer (info not in docs) |

### Benchmark Results (RX 9070 XT)

| Metric | Value |
|---|---|
| Model load time | ~7 seconds |
| VRAM usage (model) | 14.37 GiB |
| KV cache available | 12.75 GiB |
| Input throughput | ~800 tokens/s |
| Output throughput | ~32 tokens/s |
| Accuracy | 97.3% (10.7/11) |
| Total time (11 questions) | ~21 seconds |

## Configuration

### Changing the Knowledge Base

Replace `spesifikasi_aleleon.txt` with any plain text file. Update the filename in the script:

```python
loader = TextLoader("your_document.txt")
```

### Changing the LLM Model

Edit the `model` parameter in the script:

```python
llm = VLLM(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",  # Change this
    ...
)
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

### Tuning Chunk Parameters

| Parameter | Default | Effect |
|---|---|---|
| `chunk_size` | 1000 | Larger = more context per chunk, fewer chunks |
| `chunk_overlap` | 200 | Larger = less info lost at chunk boundaries |
| `search_kwargs["k"]` | 5 | More chunks = more context for LLM, but slower |

## Dependencies

Managed via Poetry (`pyproject.toml`):

- **langchain** / **langchain-core** / **langchain-community** — RAG chain orchestration
- **langchain-text-splitters** — Document chunking
- **langchain-huggingface** — HuggingFace embedding integration
- **langchain-chroma** — ChromaDB vector store integration
- **sentence-transformers** — Embedding model (`all-MiniLM-L6-v2`)
- **vLLM** — High-performance LLM inference engine (included in base Docker image)

## License

Private project — EFISON HPC Support.

## Author

Kristo Nova (vista.indonesia@gmail.com)
