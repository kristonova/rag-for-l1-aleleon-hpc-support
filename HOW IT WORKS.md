# Complete Explanation of the RAG Code Logic

## Overall Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     RAG Pipeline (Podman Containers)                     │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐      │
│  │  INGESTION   │→ │  EMBEDDING   │→ │ RETRIEVAL │→ │ GENERATION  │      │
│  │  (Wiki HTML) │  │  + Store     │  │  (Search) │  │   (LLM)     │      │
│  └──────────────┘  └──────────────┘  └──────────┘  └─────────────┘      │
│                                                                          │
│  Phase 1: Fetch & Split   Phase 2: Vectorize   Phase 3: Answer          │
│                                                                          │
│  Services:                                                               │
│  ┌─────────────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────────┐      │
│  │ embedding-service│ │  vllm-rocm  │ │ chromadb │ │   rag-app    │      │
│  │ (BAAI/bge-m3)   │ │ (Qwen3.5)   │ │ (Vector) │ │ (Orchestrator│      │
│  │ Port 8001       │ │ Port 8000   │ │ Port 8002│ │              │      │
│  └─────────────────┘ └─────────────┘ └──────────┘ └──────────────┘      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## PHASE 1: Data Ingestion Pipeline

### Persistence Check — Skip Scraping if ChromaDB Exists

```python
if chroma_db_exists():
    vectorstore = load_vectorstore(embeddings)
else:
    vectorstore = build_vectorstore(embeddings)
```

**What happens:**
- Before scraping, the system checks if a ChromaDB directory already exists on disk (`./chroma_db`).
- If it exists and contains data → **skip scraping entirely** and load vectors from disk.
- If it doesn't exist → proceed with full ingestion pipeline below.
- ChromaDB is persisted via a Podman named volume (`rag-chroma-db:/app/chroma_db`), so data survives container restarts.
- To force re-scraping (e.g., wiki content changed): delete the volume with `podman volume rm rag-for-l1-aleleon-hpc-support_rag-chroma-db`.

### Step 1 — Parse Sitemap XML

```python
splits = load_wiki_documents(
    sitemap_url="https://wiki.efisonlt.com/sitemap/sitemap-wiki.efisonlt.com-NS_0-0.xml",
    requests_per_second=2,
)
```

**What happens:**
- The function fetches the wiki's **sitemap XML** file.
- It parses the XML to extract all `<loc>` URLs — these are all the wiki page addresses.
- Rate-limited to 2 requests per second to avoid overwhelming the wiki server.

```python
resp = requests.get(sitemap_url)
root = ElementTree.fromstring(resp.content)
ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
urls = [loc.text for loc in root.findall(".//ns:loc", ns)]
```

```
Sitemap XML
    │
    ▼  Parse <loc> tags
┌──────────────────────────────────────────────────┐
│ URL 1: https://wiki.efisonlt.com/wiki/Spesifikasi│
│ URL 2: https://wiki.efisonlt.com/wiki/Conda_Env  │
│ URL 3: https://wiki.efisonlt.com/wiki/MPI_Guide  │
│ ...                                              │
└──────────────────────────────────────────────────┘
```

### Step 2 — Fetch & Extract HTML Content

```python
page_resp = requests.get(url, timeout=30)
soup = BeautifulSoup(page_resp.content, "lxml")
content_div = soup.find("div", {"id": "mw-content-text"})
content_html = str(content_div)
```

**What happens:**
- For each URL, the page is downloaded.
- **BeautifulSoup** (with `lxml` parser) extracts only the `<div id="mw-content-text">` — this is the main content area of a MediaWiki page, excluding navigation, sidebar, footer, etc.
- The content is kept as **raw HTML** (not plain text) — this is critical because the next step uses HTML heading tags for splitting.

```
Full Wiki Page HTML
        │
        ▼  BeautifulSoup → find("div", {"id": "mw-content-text"})
┌──────────────────────────────┐
│ <div id="mw-content-text">   │
│   <h2>Spesifikasi</h2>       │  ← Heading tags preserved
│   <p>ALELEON memiliki...</p> │
│   <h3>Compute Node</h3>     │
│   <table>...</table>         │  ← Tables preserved
│   <h3>Interactive Node</h3>  │
│   <p>...</p>                 │
│ </div>                       │
└──────────────────────────────┘
```

### Step 3 — Structure-Based Splitting (HTMLSectionSplitter)

```python
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]
html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)
html_docs = html_splitter.split_text(content_html)
```

**What happens:**

Unlike the old approach (splitting by character count), the current code splits by **HTML document structure**. The `HTMLSectionSplitter` looks for `<h1>`, `<h2>`, and `<h3>` tags and creates one chunk per section.

```
Split strategy:
  <h1> → New chunk (Header 1)
  <h2> → New chunk (Header 2)
  <h3> → New chunk (Header 3)
  
  Content between headings → belongs to the chunk above it
```

**Why structure-based splitting is better than character-based:**

```
Character-based (OLD):
  "...cara membuat conda env:        ← Chunk 1 ends mid-instruction
   1. module load anaconda3          ← Chunk 2 starts here
   2. conda create -n myenv..."

Structure-based (NEW):
  <h3>Membuat Conda Environment</h3>  ← Chunk boundary = section boundary
  1. module load anaconda3
  2. conda create -n myenv
  3. source activate myenv            ← Entire section stays together
```

**Metadata is automatically added:**

Each chunk gets metadata about which heading it came from:
```python
doc.metadata["source"] = url         # e.g., "https://wiki.efisonlt.com/wiki/..."
doc.metadata["title"] = page_title   # e.g., "Komputasi Python dengan Conda"
# HTMLSectionSplitter also adds:
doc.metadata["Header 2"] = "..."     # The h2 heading text
doc.metadata["Header 3"] = "..."     # The h3 heading text (if any)
```

### Step 4 — Fallback Splitting (RecursiveCharacterTextSplitter)

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4500,
    chunk_overlap=900,
    separators=["\n---", "\n\n", "\n", " "],
)

for doc in html_docs:
    if len(doc.page_content) > 4500:
        sub_splits = text_splitter.split_documents([doc])
        all_splits.extend(sub_splits)
    else:
        all_splits.append(doc)
```

**What happens:**

Some wiki sections are very long (e.g., a single `<h2>` section with many subsections that don't have `<h3>` tags). If any chunk exceeds **4500 characters**, it falls back to `RecursiveCharacterTextSplitter`:

```
Split priority (fallback):
  1. "\n---"   ← Split at horizontal rules
  2. "\n\n"    ← Split at empty paragraphs
  3. "\n"      ← Split at new lines
  4. " "       ← Split at spaces (last resort)
```

```
chunk_size=4500     → Maximum 4500 characters per chunk
chunk_overlap=900   → 900 characters repeated between consecutive chunks
```

**Why 900 overlap?** Chunks are much larger now (4500 vs old 1000), so the overlap must be proportionally larger to preserve context at boundaries.

```
HTMLSectionSplitter output:
┌──────────┐ ┌──────────┐ ┌────────────────┐ ┌──────────┐
│ Section 1│ │ Section 2│ │ Section 3      │ │ Section 4│
│ 2100 chr │ │ 3800 chr │ │ 7200 chr ← BIG│ │ 1500 chr │
│    OK    │ │    OK    │ │ needs fallback │ │    OK    │
└──────────┘ └──────────┘ └───────┬────────┘ └──────────┘
                                  │
                          RecursiveCharacterTextSplitter
                                  │
                          ┌───────┴───────┐
                          │ Sub-chunk 3a  │ Sub-chunk 3b │
                          │ 4200 chr      │ 3900 chr     │
                          └───────────────┴──────────────┘
```

### Step 5 — Metadata Enrichment (Source Labels)

```python
for s in splits:
    title = s.metadata.get("title", "Unknown")
    header = s.metadata.get("Header 2", s.metadata.get("Header 3", ""))
    prefix = f"[Sumber: {title}]"
    if header:
        prefix += f" [Section: {header}]"
    s.page_content = f"{prefix}\n{s.page_content}"
```

**What happens:**

After all splitting is done, each chunk's `page_content` is **prefixed** with a source label. This means when the LLM reads the context, it knows **where** each piece of information came from.

```
Before:
  "Untuk membuat conda environment, jalankan perintah..."

After:
  "[Sumber: Komputasi Python dengan Conda Environment User] [Section: Membuat Conda Environment]
   Untuk membuat conda environment, jalankan perintah..."
```

This is important for:
1. **LLM grounding** — The model can cite which wiki page it's referencing.
2. **Source attribution** — Each answer can be traced back to its origin.
3. **Debugging** — We can verify which chunks are being retrieved.

---

## PHASE 2: Embedding + Vector Database

### Step 6 — Vector Embedding via API Service

```python
embeddings = EmbeddingServiceClient()
```

Embedding tidak lagi dijalankan secara lokal. Sekarang menggunakan **embedding-service** — sebuah container Podman terpisah yang melayani model `BAAI/bge-m3` via REST API.

```python
class EmbeddingServiceClient(Embeddings):
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            f"{self.api_url}/embed",
            json={"texts": texts},
            timeout=600,
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._call_api(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._call_api([text])[0]
```

**Architecture:**

```
rag-app container                    embedding-service container
┌──────────────────┐                ┌──────────────────────────┐
│ EmbeddingService │  HTTP POST     │ FastAPI + SentenceTransf.│
│ Client           │ ──────────→    │ BAAI/bge-m3              │
│ (LangChain       │  /embed       │ model.encode(texts)      │
│  Embeddings)     │ ←──────────    │                          │
│                  │  JSON response │ Port 8001                │
└──────────────────┘                └──────────────────────────┘
```

**Batching:** Dokumen di-embed dalam batch @32 teks per request, bukan semua sekaligus. Ini mencegah timeout karena model besar.

```
450 chunks total:
  Batch  1/15: texts[  0: 32] → POST /embed → 32 vectors
  Batch  2/15: texts[ 32: 64] → POST /embed → 32 vectors
  ...
  Batch 14/15: texts[416:448] → POST /embed → 32 vectors
  Batch 15/15: texts[448:450] → POST /embed →  2 vectors
  ───────────────────────────────────────────────────
  Total: 450 vectors returned
```

**Model used: `BAAI/bge-m3`**

| Property | Detail |
|---|---|
| Architecture | XLM-RoBERTa based (multilingual transformer) |
| Parameters | ~568M |
| Output Dimensions | **1024 dimensions** |
| Max Sequence | 8192 tokens |
| Languages | 100+ languages including **Bahasa Indonesia** |
| Features | Dense + Sparse + ColBERT multi-vector retrieval |
| Runs on | CPU or GPU, served via embedding-service container |

**Why `BAAI/bge-m3` instead of `intfloat/multilingual-e5-large`?**

| Feature | multilingual-e5-large (old) | BAAI/bge-m3 (current) |
|---|---|---|
| Dimensions | 1024 | **1024** (same) |
| Max Tokens | 512 | **8192** (16x longer context) |
| Prefix Required | Yes ("query: " / "passage: ") | **No** (no prefix needed) |
| Retrieval Modes | Dense only | **Dense + Sparse + ColBERT** |
| MTEB Score | Strong | **Stronger** (state-of-the-art multilingual) |

BGE-M3 tidak memerlukan prefix "query: " atau "passage: " seperti E5, sehingga kode lebih sederhana — teks dikirim langsung tanpa modifikasi.

**What is an embedding?**

An embedding converts text into a **vector of numbers** in a 1024-dimensional space. Texts with **similar meanings** will have vectors that are **close** to each other.

```
"Cara membuat conda environment di ALELEON"
        │
        ▼  BAAI/bge-m3
[0.032, -0.118, 0.245, ..., 0.067]    ← 1024 numbers

"Bagaimana membuat conda env baru?"
        │
        ▼  BAAI/bge-m3
[0.029, -0.121, 0.238, ..., 0.071]    ← 1024 numbers (SIMILAR!)

"Berapa harga berlangganan ALELEON?"
        │
        ▼  BAAI/bge-m3
[-0.156, 0.089, -0.034, ..., 0.193]   ← 1024 numbers (DISTANT!)
```

**This is NOT TF-IDF or BM25.**

| Method | How it works | Used in this code? |
|---|---|---|
| **TF-IDF** | Counts word frequency. "conda" appearing 3x = relevant. Doesn't understand meaning. | ❌ |
| **BM25** | Advanced TF-IDF with document length normalization. | ❌ |
| **Sparse Retrieval** | Large vectors, mostly zeros. Matches keywords. | ❌ |
| **Dense Retrieval** ✅ | Text → dense 1024D vector via neural network. Matches **meaning**. | ✅ **Used here** |

**Advantages of Dense Retrieval:**

```
Query: "Saya butuh banyak memori untuk job saya"
  │
  ├── TF-IDF/BM25: Search for word "memori" → NOT FOUND (document says "RAM")
  │
  └── Dense (bge-m3): Understands "memori" ≈ "RAM" semantically → FOUND ✅
```

### Step 7 — Vector Database (Chroma — Persistent)

```python
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=CHROMA_PERSIST_DIR,
    collection_name=CHROMA_COLLECTION_NAME,
)
```

**What happens:**

1. Each chunk is embedded into a 1024D vector (via embedding-service API, in batches of 32).
2. The vector + original text + metadata is stored in the Chroma database **on disk** (persistent).

```
Chroma DB (persistent on disk — ./chroma_db)
┌─────────────────────────────────────────────────────────────────────┐
│ ID │ Vector (1024D)             │ Original Text        │ Metadata  │
├────┼────────────────────────────┼──────────────────────┼───────────┤
│ 0  │ [0.03, -0.12, 0.24, ...]  │ "[Sumber: Spesifika- │ title,    │
│    │                            │  si] Compute Node.." │ source,   │
│ 1  │ [0.08, -0.05, 0.19, ...]  │ "[Sumber: Conda Env] │ Header 2, │
│    │                            │  Membuat conda..."   │ Header 3  │
│ 2  │ [-0.07, 0.14, 0.03, ...]  │ "[Sumber: MPI Guide] │           │
│    │                            │  Cara submit MPI..." │           │
│ ...│ ...                        │ ...                  │ ...       │
└────┴────────────────────────────┴──────────────────────┴───────────┘
```

**Chroma** is a vector database that is:
- Lightweight, runs as **persistent local storage** (using `persist_directory`).
- Data survives container restarts via Podman named volume (`rag-chroma-db:/app/chroma_db`).
- Supports **cosine similarity search**.
- On first run: scraping + embedding + storing (~450 chunks). On subsequent runs: loads from disk instantly.

---

## PHASE 3: Retrieval + Generation

### Retrieval — Search Relevant Chunks

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

**Retrieval type: Approximate Nearest Neighbor (ANN) with Cosine Similarity**

When a user asks a question, the process is:

```
User: "Bagaimana cara membuat conda environment?"
         │
         ▼ BAAI/bge-m3 (via embedding-service API)
Query Vector: [0.029, -0.121, 0.238, ..., 0.071]    (1024D)
         │
         ▼ Cosine Similarity against ALL chunks in Chroma
         │
┌────────┬──────────────────────────────────────────┬────────────┐
│ Chunk  │ Content (with source label)              │ Similarity │
├────────┼──────────────────────────────────────────┼────────────┤
│ 3      │ "[Sumber: Conda Env] Membuat Conda..."   │ 0.91 ← #1 │
│ 7      │ "[Sumber: Conda Env] Module Pyload..."   │ 0.78 ← #2 │
│ 1      │ "[Sumber: Spesifikasi] Compute Node..."  │ 0.65 ← #3 │
│ 12     │ "[Sumber: MPI Guide] Running MPI..."     │ 0.58 ← #4 │
│ ...    │ ...                                      │ ...        │
│ 22     │ "[Sumber: Job Script] GPU Slurm..."      │ 0.41 ← #10│
└────────┴──────────────────────────────────────────┴────────────┘
         │
         ▼ Get Top-K (k=10)
    Top 10 chunks → sent to LLM as context
```

**Why k=10?** Memberikan lebih banyak konteks ke LLM sehingga jawaban lebih lengkap. Qwen3.5-35B memiliki context window 131072 tokens, cukup untuk menampung 10 chunks.

**Cosine Similarity Formula:**

```
                    A · B           Σ(Aᵢ × Bᵢ)
cos(θ) = ─────────────────── = ─────────────────────
              ||A|| × ||B||     √Σ(Aᵢ²) × √Σ(Bᵢ²)

Result: -1 (opposite) to +1 (identical)
```

### Prompt — OpenAI Messages Format (Bahasa Indonesia)

Prompt tidak lagi menggunakan ChatML template string. Sekarang menggunakan format **OpenAI messages** — array of `{role, content}` objects yang dikirim ke vLLM via OpenAI-compatible API.

```python
def generate_response(question: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": """Kamu adalah agen AI asisten admin HPC Slurm yang ahli.

Aturan:
0. Berbicaralah dalam Bahasa Indonesia.
1. Jawab HANYA berdasarkan dokumen referensi di bawah.
2. Sertakan angka, nama, versi PERSIS seperti di dokumen.
3. Jika informasi bisa DISIMPULKAN dari dokumen, berikan kesimpulan logis.
4. Jika informasi TIDAK ADA, katakan "Saya tidak menemukan informasi tersebut."
5. Jangan mengarang angka, rumus, perintah, URL, atau langkah-langkah.
6. JANGAN mengganti perintah dari dokumen dengan alternatif.
7. Bedakan "minimal" dan "maksimal".
8. Langkah-langkah yang anda berikan harus diberikan dalam URUTAN yang BENAR sesuai dengan konteks yang diberikan.
9. Untuk pertanyaan yang jawabannya berisi prosedur langkah-langkah, berikan langkah-langkah LENGKAP (jangan potong/ringkas).
10. Berikan informasi semua yang ada di dalam dokumen secara LENGKAP.""",
        },
        {
            "role": "user",
            "content": f"Dokumen Referensi:\n{context}\n\nPertanyaan: {question}",
        },
    ]

    response = client.chat.completions.create(
        model=VLLM_MODEL_NAME,
        messages=messages,
        temperature=0.3,
        top_p=0.9,
        max_tokens=32768,
        extra_body={"top_k": 20, "presence_penalty": 1.5, "enable_thinking": False},
    )
    return response.choices[0].message.content
```

**Format: OpenAI Messages (bukan ChatML string)**

vLLM menyediakan OpenAI-compatible API. Kita menggunakan `openai.OpenAI` client untuk mengirim request — vLLM otomatis mengkonversi messages ke format ChatML yang dipahami Qwen.

```
client.chat.completions.create(
    messages=[
        {"role": "system", "content": "..."},    ← System prompt + rules
        {"role": "user", "content": "..."},      ← Context + question
    ]
)
        │
        ▼ vLLM converts to ChatML internally
        │
<|im_start|>system
...<|im_end|>
<|im_start|>user
...<|im_end|>
<|im_start|>assistant
```

**The 11 Anti-Hallucination Rules (0-10):**

| Rule | Purpose |
|---|---|
| 0. Bahasa Indonesia | Ensures responses are in Indonesian |
| 1. Answer ONLY from documents | Prevents generating info from pre-training knowledge |
| 2. Exact numbers/versions | Prevents rounding ">=11" to "11.0" |
| 3. Allow deduction | Lets LLM infer logical conclusions from data |
| 4. "Not found" response | Forces refusal when info doesn't exist |
| 5. No fabrication | Blocks fake commands, URLs, procedures |
| 6. No command substitution | Prevents replacing `source activate` with `conda activate` |
| 7. Min vs Max distinction | Prevents confusing "at least X" with "at most X" |
| 8. Correct ordering | Steps must be in the RIGHT ORDER from context |
| 9. Complete procedures | Don't truncate/summarize step-by-step procedures |
| 10. Complete information | Include ALL information from documents fully |

### Generation — vLLM + Qwen3.5 via OpenAI API

```python
from openai import OpenAI

client = OpenAI(base_url=VLLM_API_URL, api_key="not-needed")
```

Model dijalankan di container **vllm-rocm** pada AMD GPU menggunakan vLLM dengan OpenAI-compatible API.

**vLLM launch command (from compose.yml):**

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
    --dtype float16 \
    --enforce-eager \
    --max-model-len 131072
```

**Generation flow:**

```
┌──────────────────────────────────────────────────────────────┐
│ OpenAI API call to vLLM:                                     │
│                                                              │
│ client.chat.completions.create(                              │
│   model="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",                   │
│   messages=[                                                 │
│     {"role": "system", "content": "Kamu adalah agen AI...    │
│      Aturan: 0-10 (11 anti-hallucination rules)"},           │
│     {"role": "user", "content": "Dokumen Referensi:\n...     │
│      Pertanyaan: Bagaimana cara membuat conda environment?"}│
│   ],                                                         │
│   temperature=0.3, top_p=0.9, max_tokens=32768,             │
│   extra_body={top_k=20, presence_penalty=1.5,                │
│               enable_thinking=False}                         │
│ )                                                            │
│         │                                                    │
│         ▼ vLLM converts to ChatML + generates                │
│                                                              │
│ "Untuk membuat conda environment di ALELEON,                 │
│  jalankan perintah berikut:                                  │
│  1. module load anaconda3/2025.06-1                          │
│  2. conda create -n myenv python=3.12..."                    │
└──────────────────────────────────────────────────────────────┘
```

**Generation parameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `temperature=0.3` | Low → more deterministic, factual | Best for RAG — reduces hallucination |
| `top_p=0.9` | Nucleus sampling — top 90% probability | Reduces random answers |
| `top_k=20` | Only consider top 20 tokens at each step | Further constrains randomness |
| `presence_penalty=1.5` | Strongly penalize repeating tokens | Prevents repetitive output |
| `max_tokens=32768` | Max 32K output tokens | Allows very detailed answers |
| `enable_thinking=False` | Disable Qwen3.5 "thinking" mode | Direct answers without reasoning trace |
| `--max-model-len 131072` | Max 128K total tokens (prompt + output) | Full context window for large prompts |
| `--dtype float16` | FP16 precision | Required for GPTQ models on ROCm |
| `--enforce-eager` | Disable CUDAGraph | ROCm / AMD GPU compatibility |

**Model: Qwen/Qwen3.5-35B-A3B-GPTQ-Int4**

| Property | Detail |
|---|---|
| Parameters | 35B total, ~3B active (MoE architecture) |
| Quantization | GPTQ 4-bit |
| Context Window | 131072 tokens (128K) |
| Architecture | Mixture of Experts (MoE) |
| Served via | vLLM on AMD ROCm GPU |

### Source Attribution — Showing Document Sources

```python
for i, inp in enumerate(inputs, 1):
    context_text, relevant_docs = create_rag_chain(inp, retriever)
    answer = generate_response(inp, context_text)
    print(answer)

    # Tampilkan sumber dokumen yang digunakan
    seen = []
    for doc in relevant_docs:
        title = doc.metadata.get("title", "Unknown")
        source = doc.metadata.get("source", "")
        header = doc.metadata.get("Header 2", doc.metadata.get("Header 3", ""))
        key = (title, header)
        if key not in seen:
            seen.append(key)
            label = f"    • {title}"
            if header:
                label += f" → {header}"
            if source:
                label += f"  ({source})"
            print(label)
```

**What happens:**

After each answer, the system displays which wiki pages and sections were used to generate the response. **De-duplication** is applied so the same source/section pair is only shown once.

```
Output example:
============================================================
[Q1/23] Bagaimana cara membuat conda environment di aleleon?
------------------------------------------------------------
Untuk membuat conda environment di ALELEON, jalankan...

    📚 Sumber (10 chunks):
    • Komputasi Python dengan Conda Environment User → Membuat Conda Environment
      (https://wiki.efisonlt.com/wiki/Komputasi_Python_dengan_Conda_Environment_User)
    • Komputasi Python dengan Conda Environment User → Module Pyload
      (https://wiki.efisonlt.com/wiki/Komputasi_Python_dengan_Conda_Environment_User)
    • ...
```

### RAG Chain — Custom Python Function

Tidak lagi menggunakan `create_stuff_documents_chain` atau `create_retrieval_chain` dari LangChain. Sekarang menggunakan fungsi Python sederhana:

```python
def create_rag_chain(question: str, retriever):
    """Retrieve relevant docs and build context string."""
    relevant_docs = retriever.invoke(question)

    context_parts = []
    for doc in relevant_docs:
        context_parts.append(doc.page_content)

    context_text = "\n\n".join(context_parts)
    return context_text, relevant_docs
```

**Strategy: "Stuff" (manual)**

Sama seperti sebelumnya — semua chunks digabung ke 1 prompt. Bedanya, sekarang dilakukan secara eksplisit dengan Python, bukan via LangChain chain abstraction.

```
User Question
    │
    ▼
┌──────────────┐     ┌────────────────────┐     ┌──────────────────┐
│ retriever    │ ──→ │ create_rag_chain() │ ──→ │ generate_response│
│ .invoke(q)   │     │ join chunks        │     │ (OpenAI client)  │
│ (Top-10)     │     │ → context_text     │     │ → answer text    │
└──────────────┘     └────────────────────┘     └──────────────────┘
    │                        │                          │
    │ 10 relevant            │ context_text =           │ answer = LLM text
    │ Documents              │ chunk1\n\nchunk2\n\n...  │ relevant_docs for
    ▼                        ▼                          ▼ source attribution
 From Chroma           To generate_response()     Display to user
```

**Why custom function instead of LangChain chains?**

- Lebih transparan — bisa di-debug dengan print statement
- Tidak perlu `langchain_classic` dependency
- Mudah dikustomisasi (filter, reranking, etc.)
- `generate_response()` menggunakan OpenAI client langsung

### Wait for vLLM — Health Check

Sebelum memulai RAG, sistem menunggu vLLM siap:

```python
def wait_for_vllm(url, timeout=600, interval=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/health")
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)
    raise TimeoutError("vLLM did not become healthy")
```

Model besar (35B params) memerlukan waktu loading ke VRAM. Fungsi ini polling `/health` setiap 10 detik, timeout setelah 10 menit.

---

## Full End-to-End Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    STARTUP PHASE                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [0] wait_for_vllm() — poll /health every 10s (max 10min)  │
│         │                                                   │
│         ▼                                                   │
│  [1] chroma_db_exists()? ────── YES ──→ load_vectorstore() │
│         │                                   (skip to [7])  │
│         NO                                                  │
│         │                                                   │
│  Wiki Sitemap XML                                           │
│  (https://wiki.efisonlt.com/sitemap/...)                    │
│         │                                                   │
│  [2] Parse XML → extract all wiki page URLs                │
│         │                                                   │
│  [3] For each URL:                                          │
│      requests.get() → BeautifulSoup                        │
│      → extract <div id="mw-content-text">                  │
│         │                                                   │
│  [4] HTMLSectionSplitter (split by h1/h2/h3 headings)      │
│      → Fallback: RecursiveCharacterTextSplitter            │
│        (4500 chars, 900 overlap)                            │
│         │                                                   │
│  [5] Add source labels:                                     │
│      "[Sumber: title] [Section: header]"                   │
│         │                                                   │
│  ~450 Chunks                                                │
│         │                                                   │
│  [6] BAAI/bge-m3 via embedding-service API                 │
│      Batched @32 chunks per request                        │
│      Each chunk → 1024-dimensional vector                  │
│         │                                                   │
│      build_vectorstore() →                                  │
│  [7] Chroma DB (persistent — ./chroma_db)                  │
│      ~450 vectors + texts + metadata stored on disk        │
│      Podman volume: rag-chroma-db:/app/chroma_db           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    PER-QUESTION PHASE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User: "Bagaimana cara membuat conda env?"                  │
│         │                                                   │
│  [a] Embed question → 1024D vector                         │
│      (BAAI/bge-m3 via embedding-service API)               │
│         │                                                   │
│  [b] Cosine similarity vs ~450 chunks in Chroma            │
│         │                                                   │
│  [c] Retrieve top-10 most relevant chunks                  │
│         │                                                   │
│  [d] create_rag_chain() → join chunks into context_text    │
│         │                                                   │
│  [e] generate_response() → OpenAI messages format          │
│      with 11 anti-hallucination rules (0-10)               │
│         │                                                   │
│  [f] Send to Qwen3.5-35B-A3B-GPTQ-Int4 via vLLM           │
│      (OpenAI-compatible API, AMD ROCm GPU)                 │
│      temperature=0.3, presence_penalty=1.5                 │
│         │                                                   │
│  [g] Model generates answer                                │
│         │                                                   │
│  [h] Display answer + source attribution                   │
│      (de-duplicated title/section/URL)                     │
│         │                                                   │
│         ▼                                                   │
│  "Untuk membuat conda environment di ALELEON:               │
│   1. module load anaconda3/2025.06-1                        │
│   2. conda create -n myenv python=3.12..."                  │
│                                                             │
│      📚 Sumber (10 chunks):                                 │
│      • Conda Environment User → Membuat Conda Environment  │
│        (https://wiki.efisonlt.com/wiki/...)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```