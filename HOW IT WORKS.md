# Complete Explanation of the RAG Code Logic

## Overall Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          RAG Pipeline                                │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐  │
│  │  INGESTION   │→ │  EMBEDDING   │→ │ RETRIEVAL │→ │ GENERATION  │  │
│  │  (Wiki HTML) │  │  + Store     │  │  (Search) │  │   (LLM)     │  │
│  └──────────────┘  └──────────────┘  └──────────┘  └─────────────┘  │
│                                                                      │
│  Phase 1: Fetch & Split   Phase 2: Vectorize   Phase 3: Answer      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## PHASE 1: Data Ingestion Pipeline

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

### Step 6 — Vector Embedding

```python
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
```

**Model used: `intfloat/multilingual-e5-large`**

| Property | Detail |
|---|---|
| Architecture | XLM-RoBERTa (multilingual transformer) |
| Parameters | ~560M |
| Output Dimensions | **1024 dimensions** |
| Max Sequence | 512 tokens |
| Languages | 100+ languages including **Bahasa Indonesia** |
| Runs on | CPU or GPU (~1.2GB model) |

**Why `multilingual-e5-large` instead of `all-MiniLM-L6-v2`?**

| Feature | all-MiniLM-L6-v2 (old) | multilingual-e5-large (current) |
|---|---|---|
| Dimensions | 384 | **1024** (richer representation) |
| Parameters | 22.7M | **~560M** (more capable) |
| Indonesian | Weak | **Strong** (trained on 100+ languages) |
| Semantic quality | Good for English | **Excellent for multilingual** |

Since the wiki documents are in **Bahasa Indonesia**, a multilingual model is essential for accurate semantic matching.

**What is an embedding?**

An embedding converts text into a **vector of numbers** in a 1024-dimensional space. Texts with **similar meanings** will have vectors that are **close** to each other.

```
"Cara membuat conda environment di ALELEON"
        │
        ▼  multilingual-e5-large
[0.032, -0.118, 0.245, ..., 0.067]    ← 1024 numbers

"Bagaimana membuat conda env baru?"
        │
        ▼  multilingual-e5-large
[0.029, -0.121, 0.238, ..., 0.071]    ← 1024 numbers (SIMILAR!)

"Berapa harga berlangganan ALELEON?"
        │
        ▼  multilingual-e5-large
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
  └── Dense (E5-large): Understands "memori" ≈ "RAM" semantically → FOUND ✅
```

### Step 7 — Vector Database (Chroma)

```python
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
```

**What happens:**

1. Each chunk is embedded into a 1024D vector.
2. The vector + original text + metadata is stored in the Chroma database (in-memory).

```
Chroma DB (in-memory)
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
- Lightweight, runs **in-memory** (no separate server needed).
- Supports **cosine similarity search**.
- Suitable for prototyping (production usually uses Pinecone, Weaviate, Milvus).

---

## PHASE 3: Retrieval + Generation

### Retrieval — Search Relevant Chunks

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

**Retrieval type: Approximate Nearest Neighbor (ANN) with Cosine Similarity**

When a user asks a question, the process is:

```
User: "Bagaimana cara membuat conda environment?"
         │
         ▼ multilingual-e5-large
Query Vector: [0.029, -0.121, 0.238, ..., 0.071]    (1024D)
         │
         ▼ Cosine Similarity against ALL chunks
         │
┌────────┬──────────────────────────────────────────┬────────────┐
│ Chunk  │ Content (with source label)              │ Similarity │
├────────┼──────────────────────────────────────────┼────────────┤
│ 3      │ "[Sumber: Conda Env] Membuat Conda..."   │ 0.91 ← #1 │
│ 7      │ "[Sumber: Conda Env] Module Pyload..."   │ 0.78 ← #2 │
│ 1      │ "[Sumber: Spesifikasi] Compute Node..."  │ 0.65 ← #3 │
│ 12     │ "[Sumber: MPI Guide] Running MPI..."     │ 0.32       │
│ ...    │ ...                                      │ ...        │
└────────┴──────────────────────────────────────────┴────────────┘
         │
         ▼ Get Top-K (k=3)
    Chunks 3, 7, 1 → sent to LLM as context
```

**Cosine Similarity Formula:**

```
                    A · B           Σ(Aᵢ × Bᵢ)
cos(θ) = ─────────────────── = ─────────────────────
              ||A|| × ||B||     √Σ(Aᵢ²) × √Σ(Bᵢ²)

Result: -1 (opposite) to +1 (identical)
```

### Prompt Template — ChatML Format (Bahasa Indonesia)

```python
template_qwen = """<|im_start|>system
Kamu adalah agen AI asisten admin HPC Slurm yang ahli...

Aturan:
1. Jawab HANYA berdasarkan dokumen referensi...
2. Sertakan angka, nama, versi PERSIS seperti di dokumen...
3. Jika informasi bisa DISIMPULKAN dari dokumen, berikan kesimpulan...
4. Jika informasi TIDAK ADA, katakan "Saya tidak menemukan..."
5. Jangan mengarang angka, rumus, perintah, URL...
6. JANGAN mengganti perintah dari dokumen dengan alternatif...
7. Bedakan "minimal" dan "maksimal"...<|im_end|>

<|im_start|>user
Dokumen Referensi:
{context}

Pertanyaan: {input}<|im_end|>
<|im_start|>assistant
"""
```

**Why the `<|im_start|>` / `<|im_end|>` format?**

This is the **ChatML format** — the format used during Qwen model training to distinguish roles:

```
<|im_start|>system     ← Instructions for the model (persona, rules)
...<|im_end|>
<|im_start|>user       ← User input (context + question)
...<|im_end|>
<|im_start|>assistant   ← Model starts generating from here
```

**The 7 Anti-Hallucination Rules:**

The system prompt includes 7 strict rules to prevent the model from making things up:

| Rule | Purpose |
|---|---|
| 1. Answer ONLY from documents | Prevents generating info from pre-training knowledge |
| 2. Exact numbers/versions | Prevents rounding ">=11" to "11.0" |
| 3. Allow deduction | Lets LLM infer logical conclusions from data |
| 4. "Not found" response | Forces refusal when info doesn't exist |
| 5. No fabrication | Blocks fake commands, URLs, procedures |
| 6. No command substitution | Prevents replacing `source activate` with `conda activate` |
| 7. Min vs Max distinction | Prevents confusing "at least X" with "at most X" |

**Template variables:**
- `{context}` → Automatically filled by LangChain with the 3 retrieved chunks.
- `{input}` → Filled with the user's question.

### Generation — vLLM + Qwen2.5

```python
llm = VLLM(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    trust_remote_code=True,
    max_new_tokens=1024,
    temperature=0.5,
    top_p=0.9,
    tensor_parallel_size=1,
    vllm_kwargs={
        "gpu_memory_utilization": 0.80,
        "enforce_eager": True,
        "max_model_len": 32768,
    }
)
```

**Generation flow:**

```
┌──────────────────────────────────────────────────────────────┐
│ Prompt sent to LLM:                                          │
│                                                              │
│ <|im_start|>system                                           │
│ Kamu adalah agen AI asisten admin HPC Slurm yang ahli...     │
│ Aturan: 1. Jawab HANYA berdasarkan dokumen... (7 rules)      │
│ <|im_end|>                                                   │
│ <|im_start|>user                                             │
│ Dokumen Referensi:                                           │
│ [Sumber: Conda Env] [Section: Membuat Conda Environment]    │
│ Untuk membuat conda env, jalankan: module load anaconda3...  │
│                                                              │
│ [Sumber: Conda Env] [Section: Module Pyload]                 │
│ Setelah conda env aktif, buat modul pyload...                │
│                                                              │
│ [Sumber: Spesifikasi] [Section: Compute Node]                │
│ Partisi GPU: gpu-a100, gpu-rtx...                            │
│                                                              │
│ Pertanyaan: Bagaimana cara membuat conda environment?        │
│ <|im_end|>                                                   │
│ <|im_start|>assistant                                        │
│                                                              │
│         ▼ Model generates token by token                     │
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
| `temperature=0.5` | Moderate → balanced between factual and natural phrasing | Good for RAG with Indonesian text |
| `top_p=0.9` | Nucleus sampling — only select tokens from the top 90% probability | Reduces random answers |
| `max_new_tokens=1024` | Max 1024 output tokens | Allows longer, more detailed answers |
| `max_model_len=32768` | Max 32K total tokens (prompt + output) | Full context window for large prompts |
| `gpu_memory_utilization=0.80` | Use 80% VRAM | Save 20% for overhead and stability |
| `enforce_eager=True` | Disable CUDAGraph | ROCm / RDNA4 compatibility |
| `tensor_parallel_size=1` | Single GPU | No multi-GPU parallelism |

### Source Attribution — Showing Document Sources

```python
for i, inp in enumerate(inputs, 1):
    hasil = rag_chain.invoke(inp)
    print(hasil['answer'].strip())

    # Tampilkan sumber dokumen yang digunakan
    if 'context' in hasil and hasil['context']:
        seen = []
        for doc in hasil['context']:
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

    📚 Sumber (3 chunks):
    • Komputasi Python dengan Conda Environment User → Membuat Conda Environment
      (https://wiki.efisonlt.com/wiki/Komputasi_Python_dengan_Conda_Environment_User)
    • Komputasi Python dengan Conda Environment User → Module Pyload
      (https://wiki.efisonlt.com/wiki/Komputasi_Python_dengan_Conda_Environment_User)
```

### RAG Chain — Combining Everything

```python
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

**`create_stuff_documents_chain`** — Strategy: **"Stuff"**

"Stuff" means: **put ALL chunks into 1 prompt at once**.

```
Other strategies (not used in this code):
┌─────────────────────────────────────────────────────────┐
│ Stuff     : All chunks → 1 prompt → 1 answer      ✅   │
│ Map-Reduce: Each chunk → answer → combine all          │
│ Refine    : Chunk 1 → answer → + Chunk 2 → refine      │
│ Map-Rerank: Each chunk → answer + score → pick best    │
└─────────────────────────────────────────────────────────┘
```

**Note:** These chain functions come from `langchain_classic` (not `langchain`), because the LangChain 1.x API moved `create_retrieval_chain` and `create_stuff_documents_chain` to the `langchain-classic` package.

**`create_retrieval_chain`** combines the retriever + stuff chain:

```
User Input
    │
    ▼
┌──────────┐     ┌──────────────┐     ┌──────────────────┐
│ Retriever│ ──→ │ Stuff Chain  │ ──→ │     Output       │
│ (Top-3)  │     │ (Prompt+LLM) │     │ {answer, context}│
└──────────┘     └──────────────┘     └──────────────────┘
    │                    │                      │
    │ 3 relevant         │ Prompt with          │ answer = LLM text
    │ chunks             │ context + question   │ context = Document[]
    ▼                    ▼                      ▼
 From Chroma        To vLLM/GPU          Source attribution
```

---

## Full End-to-End Diagram

```
Wiki Sitemap XML
(https://wiki.efisonlt.com/sitemap/...)
        │
   [1] Parse XML → extract all wiki page URLs
        │
        ▼
   [2] For each URL:
       requests.get() → BeautifulSoup → extract <div id="mw-content-text">
        │
        ▼
   [3] HTMLSectionSplitter (split by h1/h2/h3 headings)
       → Fallback: RecursiveCharacterTextSplitter (4500 chars, 900 overlap)
        │
        ▼
   [4] Add source labels: "[Sumber: title] [Section: header]"
        │
        ▼
  N Chunks (variable, depends on wiki content)
        │
   [5] intfloat/multilingual-e5-large (~560M params)
       Each chunk → 1024-dimensional vector
        │
        ▼
   [6] Chroma DB (in-memory)
       N vectors + N texts + metadata stored
        │
        │
   [7] vLLM + Qwen2.5-Coder-7B-Instruct (GPU, 7B params)
       Model loaded into VRAM
        │
        │
  ══════╪══════════════════════════════════════
  Per Question:
        │
  User: "Bagaimana cara membuat conda env?"
        │
        ▼
  [a] Embed question → 1024D vector (multilingual-e5-large)
        │
  [b] Cosine similarity vs N chunks in Chroma
        │
  [c] Retrieve top-3 most relevant chunks
        │
  [d] Insert into ChatML prompt template (Bahasa Indonesia)
      with 7 anti-hallucination rules
        │
  [e] Send prompt to Qwen2.5 via vLLM (GPU)
        │
  [f] Model generates answer token-by-token
        │
  [g] Display answer + source attribution
      (de-duplicated title/section/URL)
        │
        ▼
  "Untuk membuat conda environment di ALELEON:
   1. module load anaconda3/2025.06-1
   2. conda create -n myenv python=3.12..."

    📚 Sumber (3 chunks):
    • Conda Environment User → Membuat Conda Environment
      (https://wiki.efisonlt.com/wiki/...)
```