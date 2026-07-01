# Penjelasan Lengkap Logika Kode RAG

## Arsitektur Keseluruhan

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                Pipeline RAG — Microservices (Kontainer Podman)               │
│                                                                              │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────┐  ┌───────────┐    │
│  │   INGESTI    │→ │    EMBEDDING     │→ │   RETRIEVAL   │→ │  GENERASI │    │
│  │  (HTML Wiki) │  │ Dense+Sparse+    │  │ Hybrid Search │  │   (LLM)   │    │
│  │              │  │ ColBERT Rerank   │  │ + RRF Fusion  │  │           │    │
│  └──────────────┘  └──────────────────┘  └───────────────┘  └───────────┘    │
│                                                                              │
│  Fase 1: Ambil & Split   Fase 2: Vektorisasi   Fase 3: Menjawab              │
│                                                                              │
│  Layanan:                                                                    │
│  ┌─────────────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────────┐           │
│  │embedding-service│ │  vllm-rocm  │ │  qdrant  │ │   rag-api    │           │
│  │ (BAAI/bge-m3)   │ │ (Qwen3.5)   │ │ (Vektor) │ │ (REST v1.3.0)│           │
│  │ Port 8001       │ │ Port 8000   │ │ Port 6333│ │ Port 8080    │           │
│  │ Dense+Sparse+   │ │ OpenAI API  │ │ REST+gRPC│ │ /ask         │           │
│  │ ColBERT Rerank  │ │             │ │          │ │/review-script│           │
│  │ v2.0.0          │ │             │ │          │ │ /refresh     │           │
│  └─────────────────┘ └─────────────┘ └──────────┘ │ Semaphore(2) │           │
│                                                   └──────┬───────┘           │
│                                                          │                   │
│                                                   ┌──────┴───────┐           │
│                                                   │ telegram-bot │           │
│                                                   │ /ask         │           │
│                                                   │ /askscript   │           │
│                                                   └──────────────┘           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Komponen Microservices

| Service | Deskripsi | Port | Profile |
|---|---|---|---|
| `embedding-service` | REST API embedding BAAI/bge-m3 (dense + sparse + ColBERT reranking), v2.0.0 | 8001 | infra |
| `vllm-rocm` | LLM inference Qwen3.5-35B via OpenAI-compatible API (AMD ROCm) | 8000 | infra |
| `qdrant` | Database vektor persisten (dense + sparse hybrid collection) | 6333, 6334 | infra |
| `rag-api` | FastAPI REST API v1.3.0 — orkestrator RAG (`/ask`, `/review-script`, `/refresh`) + concurrency limiter (Semaphore=2) | 8080 | api |
| `telegram-bot` | Bot Telegram — `/ask` dan `/askscript` via RAG API | — | telegram |
| `benchmark` | Benchmark retrieval (Dense vs Sparse vs Multi-Vector vs Hybrid) | — | benchmark |
| `benchmark-ttft` | Benchmark TTFT/latency concurrency test | — | benchmark-ttft |
| `promtail` | Log scraping ke Grafana Loki | — | monitoring |

---

## FASE 1: Pipeline Ingesti Data

### Pemeriksaan Persistensi — Lewati Scraping jika Koleksi Qdrant Sudah Ada

```python
if qdrant_collection_exists():
    qdrant_client = load_vectorstore(embeddings)
else:
    qdrant_client = build_vectorstore(embeddings)
```

**Yang terjadi:**
- Sebelum scraping, sistem memeriksa apakah koleksi Qdrant (`wiki_aleleon_qdrant`) sudah ada di **server Qdrant** (`http://qdrant:6333`).
- Jika ada dan berisi data → **scraping dilewati sepenuhnya** dan koneksi ke Qdrant dimuat langsung.
- Jika belum ada → lanjutkan ke pipeline ingesti lengkap di bawah.
- Qdrant dipersistenkan melalui named volume Podman (`qdrant-data:/qdrant/storage`), sehingga data tetap ada saat container restart.
- Untuk memaksa scraping ulang (misalnya konten wiki berubah): hapus koleksi di Qdrant melalui dashboard (`http://localhost:6333/dashboard`), gunakan endpoint `POST /refresh` untuk incremental sync, atau hapus volume dengan `podman volume rm rag-for-l1-aleleon-hpc-support_qdrant-data`.

```python
def qdrant_collection_exists() -> bool:
    """Cek apakah collection Qdrant sudah ada dan berisi data."""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections().collections
        return any(c.name == QDRANT_COLLECTION_NAME for c in collections)
    except Exception:
        return False
```

### Langkah 1 — Parse Sitemap XML + Filter Non-Webpage

```python
splits = load_wiki_documents(
    sitemap_url="https://wiki.efisonlt.com/sitemap/sitemap-wiki.efisonlt.com-0.xml",
    requests_per_second=2,
)
```

**Yang terjadi:**
- Fungsi mengambil file **sitemap XML** wiki.
- XML diparse untuk mengekstrak semua URL `<loc>` — ini adalah seluruh alamat halaman wiki.
- **URL non-webpage difilter** — halaman file/gambar (Berkas:, File:) dan halaman spesial (Istimewa:, Special:) dibuang.
- Dibatasi 2 request per detik agar tidak membebani server wiki.

```python
resp = requests.get(sitemap_url)
root = ElementTree.fromstring(resp.content)
ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
all_urls = [loc.text for loc in root.findall(".//ns:loc", ns)]

# Filter: buang URL halaman file (Berkas:) yang bukan webpage
NON_WEBPAGE_PATTERNS = [
    "/wiki/Berkas:",   # MediaWiki file description pages
    "/wiki/File:",     # English alias for file pages
    "/wiki/Istimewa:", # Special pages
    "/wiki/Special:",  # Special pages (English)
]
urls = [
    u for u in all_urls
    if u and not any(pattern in u for pattern in NON_WEBPAGE_PATTERNS)
]
```

```
Sitemap XML
    │
    ▼  Parse tag <loc> + filter non-webpage
┌───────────────────────────────────────────────────┐
│ URL 1: https://wiki.efisonlt.com/wiki/Spesifikasi │ ✅ webpage
│ URL 2: https://wiki.efisonlt.com/wiki/Conda_Env   │ ✅ webpage
│ URL 3: https://wiki.efisonlt.com/wiki/Berkas:x.png│ ❌ di-skip
│ URL 4: https://wiki.efisonlt.com/wiki/MPI_Guide   │ ✅ webpage
│ ...                                               │
└───────────────────────────────────────────────────┘
```

### Langkah 2 — Ambil & Ekstrak Konten HTML

```python
page_resp = requests.get(url, timeout=30)
soup = BeautifulSoup(page_resp.content, "lxml")
content_div = soup.find("div", {"id": "mw-content-text"})
content_html = str(content_div)
```

**Yang terjadi:**
- Untuk setiap URL, halaman diunduh.
- **BeautifulSoup** (dengan parser `lxml`) mengekstrak hanya `<div id="mw-content-text">` — ini adalah area konten utama halaman MediaWiki, tanpa navigasi, sidebar, footer, dll.
- Konten dipertahankan sebagai **raw HTML** (bukan plain text) — ini penting karena langkah berikutnya menggunakan tag heading HTML untuk splitting.

```
HTML Halaman Wiki Penuh
        │
        ▼  BeautifulSoup → find("div", {"id": "mw-content-text"})
┌──────────────────────────────┐
│ <div id="mw-content-text">   │
│   <h2>Spesifikasi</h2>       │  ← Tag heading dipertahankan
│   <p>ALELEON memiliki...</p> │
│   <h3>Compute Node</h3>      │
│   <table>...</table>         │  ← Tabel dipertahankan
│   <h3>Interactive Node</h3>  │
│   <p>...</p>                 │
│ </div>                       │
└──────────────────────────────┘
```

### Langkah 3 — Pemotongan Berbasis Struktur (HTMLSectionSplitter)

```python
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]
html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)
html_docs = html_splitter.split_text(content_html)
```

**Yang terjadi:**

Berbeda dengan pendekatan lama (memotong berdasarkan jumlah karakter), kode saat ini memotong berdasarkan **struktur dokumen HTML**. `HTMLSectionSplitter` mencari tag `<h1>`, `<h2>`, dan `<h3>` lalu membuat satu chunk per section.

```
Strategi pemotongan:
    <h1> → Chunk baru (Header 1)
    <h2> → Chunk baru (Header 2)
    <h3> → Chunk baru (Header 3)
  
    Konten di antara heading → menjadi bagian dari chunk di atasnya
```

**Mengapa pemotongan berbasis struktur lebih baik daripada berbasis karakter:**

```
Berbasis karakter (LAMA):
  "...cara membuat conda env:        ← Chunk 1 ends mid-instruction
   1. module load anaconda3          ← Chunk 2 starts here
   2. conda create -n myenv..."

Berbasis struktur (BARU):
  <h3>Membuat Conda Environment</h3>  ← Chunk boundary = section boundary
  1. module load anaconda3
  2. conda create -n myenv
  3. source activate myenv            ← Entire section stays together
```

**Metadata ditambahkan otomatis:**

Setiap chunk mendapatkan metadata tentang heading asalnya:
```python
doc.metadata["source"] = url         # e.g., "https://wiki.efisonlt.com/wiki/..."
doc.metadata["title"] = page_title   # e.g., "Komputasi Python dengan Conda"
# HTMLSectionSplitter also adds:
doc.metadata["Header 2"] = "..."     # The h2 heading text
doc.metadata["Header 3"] = "..."     # The h3 heading text (if any)
```

### Langkah 4 — Pemotongan Fallback (RecursiveCharacterTextSplitter)

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

**Yang terjadi:**

Beberapa section wiki sangat panjang (misalnya satu section `<h2>` dengan banyak subseksi tanpa tag `<h3>`). Jika ada chunk melebihi **4500 karakter**, sistem menggunakan `RecursiveCharacterTextSplitter` sebagai fallback:

```
Prioritas pemotongan (fallback):
    1. "\n---"   ← Potong pada horizontal rule
    2. "\n\n"    ← Potong pada paragraf kosong
    3. "\n"      ← Potong pada baris baru
    4. " "       ← Potong pada spasi (opsi terakhir)
```

```
chunk_size=4500     → Maksimum 4500 karakter per chunk
chunk_overlap=900   → 900 karakter diulang antar chunk berurutan
```

**Mengapa overlap 900?** Chunk sekarang jauh lebih besar (4500 vs 1000 sebelumnya), jadi overlap harus proporsional agar konteks di batas chunk tetap terjaga.

```
HTMLSectionSplitter output:
┌──────────┐ ┌──────────┐ ┌────────────────┐ ┌──────────┐
│ Section 1│ │ Section 2│ │ Section 3      │ │ Section 4│
│ 2100 chr │ │ 3800 chr │ │ 7200 chr ← BIG │ │ 1500 chr │
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

### Langkah 5 — Pengayaan Metadata (Label Sumber)

```python
for s in splits:
    title = s.metadata.get("title", "Unknown")
    header = s.metadata.get("Header 2", s.metadata.get("Header 3", ""))
    prefix = f"[Sumber: {title}]"
    if header:
        prefix += f" [Section: {header}]"
    s.page_content = f"{prefix}\n{s.page_content}"
```

**Yang terjadi:**

Setelah seluruh proses splitting selesai, `page_content` setiap chunk diberi **prefix** label sumber. Artinya saat LLM membaca konteks, model tahu **asal** setiap potongan informasi.

```
Sebelum:
  "Untuk membuat conda environment, jalankan perintah..."

Sesudah:
  "[Sumber: Komputasi Python dengan Conda Environment User] [Section: Membuat Conda Environment]
   Untuk membuat conda environment, jalankan perintah..."
```

Ini penting untuk:
1. **LLM grounding** — Model dapat menyebut halaman wiki yang dijadikan referensi.
2. **Pelacakan sumber** — Setiap jawaban dapat ditelusuri ke asalnya.
3. **Debugging** — Kita bisa memverifikasi chunk mana yang diretrieval.

---

## FASE 2: Embedding + Basis Data Vektor (Hybrid: Dense + Sparse)

### Langkah 6 — Embedding Vektor Multi-Mode via API

```python
embeddings = EmbeddingServiceClient()
```

Embedding menggunakan **embedding-service** — sebuah container Podman terpisah yang melayani model `BAAI/bge-m3` via REST API. Service ini mendukung **tiga mode embedding**:

```python
class EmbeddingServiceClient(Embeddings):
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Dense-only embedding via /embed."""
        response = requests.post(
            f"{self.api_url}/embed",
            json={"texts": texts},
            timeout=600,
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_multi(self, texts: List[str], batch_size: int = 16) -> Dict[str, Any]:
        """Multi-mode embedding (dense + sparse) via /embed/multi."""
        all_dense, all_sparse = [], []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = requests.post(
                f"{self.api_url}/embed/multi",
                json={
                    "texts": batch,
                    "return_dense": True,
                    "return_sparse": True,
                    "return_colbert": False,
                },
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("dense"):
                all_dense.extend(data["dense"])
            if data.get("sparse"):
                all_sparse.extend(data["sparse"])
        return {"dense": all_dense, "sparse": all_sparse}

    def rerank(self, query: str, passages: List[str]) -> List[float]:
        """ColBERT reranking via /rerank."""
        resp = requests.post(
            f"{self.api_url}/rerank",
            json={"query": query, "passages": passages},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["scores"]
```

**Arsitektur Multi-Mode Embedding Service:**

```
rag-app / rag-api container          embedding-service container
┌──────────────────────┐            ┌──────────────────────────────────┐
│ EmbeddingService     │            │ FastAPI + FlagEmbedding          │
│ Client               │            │ BGEM3FlagModel (BAAI/bge-m3)     │
│                      │            │                                  │
│ embed_documents()    │  /embed    │ → Dense vectors (1024D)          │
│ embed_query()        │ ─────────→ │                                  │
│                      │            │                                  │
│ embed_multi()        │ /embed/    │ → Dense + Sparse (lexical        │
│ embed_query_multi()  │  multi     │   weights) + ColBERT (optional)  │
│                      │ ─────────→ │                                  │
│                      │            │                                  │
│ rerank()             │ /rerank    │ → ColBERT late-interaction       │
│                      │ ─────────→ │   scoring (passage reranking)    │
│                      │            │                                  │
│                      │ ←───────── │ JSON response                    │
│                      │            │ Port 8001                        │
└──────────────────────┘            └──────────────────────────────────┘
```

**Tiga Mode Embedding:**

| Mode | Endpoint | Deskripsi | Digunakan saat |
|---|---|---|---|
| **Dense** | `POST /embed` | Vektor padat 1024 dimensi | Query sederhana, backward-compatible |
| **Multi (Dense + Sparse)** | `POST /embed/multi` | Dense + Sparse (lexical weights) sekaligus | Ingestion (simpan kedua vektor ke Qdrant) |
| **ColBERT Rerank** | `POST /rerank` | ColBERT late-interaction scoring | Post-retrieval reranking |

**Batching:** Saat ingestion, dokumen di-embed dalam batch @16 teks per request via `/embed/multi`. Saat query, `embed_query_multi()` mengirim 1 teks.

```
~450 chunks total (ingestion via /embed/multi):
  Batch  1/29: texts[  0: 16] → POST /embed/multi → 16 × {dense + sparse}
  Batch  2/29: texts[ 16: 32] → POST /embed/multi → 16 × {dense + sparse}
  ...
  Batch 29/29: texts[448:450] → POST /embed/multi →  2 × {dense + sparse}
  ──────────────────────────────────────────────────────────────
  Total: 450 dense vectors + 450 sparse vectors returned
```

**Model yang digunakan: `BAAI/bge-m3` (via FlagEmbedding)**

| Properti | Detail |
|---|---|
| Arsitektur | Berbasis XLM-RoBERTa (transformer multibahasa) |
| Parameter | ~568M |
| Dimensi Output Dense | **1024 dimensi** |
| Panjang Sekuens Maksimum | 8192 token |
| Bahasa | 100+ bahasa termasuk **Bahasa Indonesia** |
| Mode Retrieval | **Dense + Sparse (lexical) + ColBERT (multi-vector reranking)** |
| Library | FlagEmbedding (`BGEM3FlagModel`) — bukan SentenceTransformers |
| Berjalan di | CPU atau GPU, disajikan via container embedding-service |

**Apa itu Sparse Embedding?**

Berbeda dengan dense embedding yang menghasilkan vektor padat, sparse embedding menghasilkan **vektor jarang** — sebagian besar elemennya nol. Setiap dimensi merepresentasikan sebuah **token/kata** dan nilainya menunjukkan pentingnya token tersebut.

```
Dense:  [0.032, -0.118, 0.245, ..., 0.067]   ← 1024 angka, semua terisi
Sparse: {token_id_42: 0.83, token_id_1505: 0.61, token_id_789: 0.44, ...}
        ← hanya token penting yang memiliki bobot
```

**Mengapa Dense + Sparse (Hybrid)?**

| Situasi | Dense saja | Sparse saja | **Hybrid (keduanya)** |
|---|---|---|---|
| Kueri semantik ("cara memakai memori besar") | ✅ Paham "memori" ≈ "RAM" | ❌ Gagal jika kata berbeda | ✅ Dense menangani |
| Kueri keyword spesifik ("epyc-jumbo") | ❌ Bisa miss nama partisi eksak | ✅ Cocokkan token persis | ✅ Sparse menangani |
| Campuran keduanya | Tergantung | Tergantung | ✅ **RRF menggabungkan kedua sinyal** |

### Langkah 7 — Basis Data Vektor Hybrid (Qdrant — Dense + Sparse)

```python
# Buat hybrid collection (dense + sparse)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=300)
client.create_collection(
    QDRANT_COLLECTION_NAME,
    vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
    sparse_vectors_config={"text-sparse": SparseVectorParams()},
)
```

**Yang terjadi:**

1. Dibuat **hybrid collection** di Qdrant dengan dua jenis vektor:
   - `"dense"` — vektor padat 1024 dimensi (cosine similarity)
   - `"text-sparse"` — vektor jarang (sparse, untuk keyword matching)
2. Setiap chunk di-embed menghasilkan **dense + sparse** sekaligus via `/embed/multi`.
3. Dense vector, sparse vector, teks asli, dan metadata di-upsert sebagai point ke Qdrant.

```python
# Embed semua chunk (dense + sparse sekaligus)
multi = embeddings.embed_multi(texts, batch_size=16)

# Upsert ke Qdrant
for j in range(i, min(i + batch_size, n)):
    sp = multi["sparse"][j]
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": multi["dense"][j],
                "text-sparse": SparseVector(
                    indices=sp["indices"], values=sp["values"]
                ),
            },
            payload={"text": texts[j], **metadatas[j]},
        )
    )
client.upsert(QDRANT_COLLECTION_NAME, points)
```

```
Qdrant Hybrid Collection (persistent — server di http://qdrant:6333)
┌──────────────────────────────────────────────────────────────────────────────┐
│ ID (UUID)│ Dense (1024D)            │ Sparse (token→weight) │ Payload        │
├──────────┼──────────────────────────┼───────────────────────┼────────────────┤
│ a1b2c3.. │ [0.03, -0.12, 0.24,...]  │ {42:0.83, 1505:0.61}  │ text, title,   │
│          │                          │                       │ source,        │
│ d4e5f6.. │ [0.08, -0.05, 0.19,...]  │ {789:0.44, 33:0.71}   │ Header 2,      │
│          │                          │                       │ Header 3,      │
│ g7h8i9.. │ [-0.07, 0.14, 0.03,...]. │ {102:0.55, 45:0.32}   │ lastmod        │
│ ...      │ ...                      │ ...                   │ ...            │
└──────────┴──────────────────────────┴───────────────────────┴────────────────┘
```

**Qdrant** adalah database vektor yang:
- Berjalan sebagai **server terpisah** dalam container Podman (port 6333 REST, port 6334 gRPC).
- Data dipersistenkan melalui named volume Podman (`qdrant-data:/qdrant/storage`).
- Mendukung **hybrid search** (dense + sparse + RRF fusion).
- Dilindungi oleh API key (`QDRANT__SERVICE__API_KEY`).
- Pada run pertama: scraping + embedding + penyimpanan. Pada run berikutnya: langsung load dari Qdrant.
- Memiliki dashboard web di `http://localhost:6333/dashboard` untuk inspeksi data.
- Menggunakan **QdrantClient langsung** (bukan LangChain `QdrantVectorStore`) untuk kontrol penuh atas hybrid search.

---

## FASE 3: Retrieval + Generasi

### Pre-Filter — Pengecekan Relevansi Pertanyaan

Sebelum melakukan embedding dan retrieval, sistem **memeriksa apakah pertanyaan relevan** dengan domain HPC/ALELEON menggunakan LLM:

```python
def is_question_relevant(question, api_url=None) -> bool:
    """Cek apakah pertanyaan relevan dengan HPC/Aleleon sebelum proses embedding."""
    messages = [
        {
            "role": "system",
            "content": """Kamu adalah filter pertanyaan untuk layanan support ALELEON HPC.
Tugasmu menentukan apakah pertanyaan user MUNGKIN berkaitan dengan topik-topik berikut:
- High Performance Computing (HPC), Supercomputer, Cluster, Komputasi
- Slurm, batch job, partisi, node, CPU, GPU, RAM, storage
- Linux, terminal, command line, module, environment
- Layanan ALELEON, EFIRO, EWS, akun, kuota, billing, Core Hour, GPU Hour
- Server, VPN, SSH, SFTP, file transfer
- Software ilmiah (GROMACS, FLACS, VASP, Conda, Python, dll)
- IT Support, troubleshooting, error

Jawab HANYA dengan kata 'YA' atau 'TIDAK'."""
        },
        ...
    ]
```

```
User: "resep nasi goreng"
       │
       ▼ is_question_relevant() → "TIDAK"
       │
       ▼ Return "Pertanyaan anda tidak relevan..."
       (TANPA membuang resource embedding/retrieval)

User: "cara membuat conda environment"
       │
       ▼ is_question_relevant() → "YA"
       │
       ▼ Lanjut ke hybrid retrieval
```

**Mengapa?** Ini menghemat resource — pertanyaan yang jelas tidak relevan (resep masak, gosip, cuaca) tidak perlu melalui proses embedding dan retrieval yang mahal.

### Retrieval — Hybrid Search (Dense + Sparse + RRF Fusion + ColBERT Reranking)

```python
def create_rag_chain(client: QdrantClient, embeddings: EmbeddingServiceClient, llm_api_url=None):
    """Create RAG chain with hybrid retrieval (dense + sparse + RRF fusion)."""

    def retrieve_and_answer(question):
        # 1. Embed query → dense + sparse
        query_multi = embeddings.embed_query_multi(question)

        # 2. Hybrid search: dense + sparse → RRF fusion (over-fetch for reranking)
        fetch_limit = TOP_K * RERANK_FETCH_MULTIPLIER  # 10 × 2 = 20
        results = client.query_points(
            QDRANT_COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=query_multi["dense"],
                    using="dense",
                    limit=fetch_limit * 2,      # 40 candidates dari dense
                ),
                Prefetch(
                    query=SparseVector(
                        indices=query_multi["sparse"]["indices"],
                        values=query_multi["sparse"]["values"],
                    ),
                    using="text-sparse",
                    limit=fetch_limit * 2,      # 40 candidates dari sparse
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),  # Reciprocal Rank Fusion
            limit=fetch_limit,                      # 20 candidates setelah RRF
        )

        # 3. ColBERT reranking: rerank 20 candidates → ambil top 10
        if len(candidate_docs) > TOP_K:
            passages = [doc.page_content for doc in candidate_docs]
            scores = embeddings.rerank(question, passages)  # POST /rerank
            scored_docs = sorted(
                zip(candidate_docs, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            docs = [doc for doc, _ in scored_docs[:TOP_K]]

        ...
    return retrieve_and_answer
```

**Pipeline Retrieval 4-Tahap:**

```
User: "Bagaimana cara membuat conda environment?"
         │
    ┌────┴─────────────────────────────────────────────────────────────┐
    │  TAHAP 1: Multi-Mode Embedding                                   │
    │  embed_query_multi(question) → POST /embed/multi                 │
    │                                                                  │
    │  → Dense vector: [0.029, -0.121, 0.238, ..., 0.071] (1024D)      │
    │  → Sparse vector: {42:0.83, 1505:0.61, ...}                      │
    └────┬─────────────────────────────────────────────────────────────┘
         │
    ┌────┴─────────────────────────────────────────────────────────────┐
    │  TAHAP 2: Dual-Path Retrieval dari Qdrant                        │
    │                                                                  │
    │  Dense path:  query dense vector → cosine similarity → 40 hits   │
    │  Sparse path: query sparse vector → keyword match    → 40 hits   │
    └────┬─────────────────────────────────────────────────────────────┘
         │
    ┌────┴─────────────────────────────────────────────────────────────┐
    │  TAHAP 3: RRF Fusion (Reciprocal Rank Fusion)                    │
    │                                                                  │
    │  Gabungkan ranking dari dense dan sparse:                        │
    │                                                                  │
    │           1                1                                     │
    │  RRF = ────── + ──────  (k = 60 default)                         │
    │        k+rank_d   k+rank_s                                       │
    │                                                                  │
    │  → 20 candidates teratas (RERANK_FETCH_MULTIPLIER × TOP_K)       │
    └────┬─────────────────────────────────────────────────────────────┘
         │
    ┌────┴─────────────────────────────────────────────────────────────┐
    │  TAHAP 4: ColBERT Reranking                                      │
    │                                                                  │
    │  embeddings.rerank(question, 20 passages) → POST /rerank         │
    │  ColBERT late-interaction scoring:                               │
    │                                                                  │
    │  Setiap token query ↔ setiap token passage                       │
    │  → Max similarity per query token → Sum → Score                  │
    │                                                                  │
    │  Sort by score → Ambil top 10 (TOP_K)                            │
    └────┬─────────────────────────────────────────────────────────────┘
         │
         ▼
    10 chunk paling relevan → dikirim ke LLM sebagai konteks
```

**Mengapa 4 tahap? (Dense → Sparse → RRF → ColBERT)**

| Tahap | Kekuatan | Kelemahan |
|---|---|---|
| Dense saja | Paham semantik ("memori" ≈ "RAM") | Bisa miss nama eksak ("epyc-jumbo") |
| Sparse saja | Cocokkan keyword persis | Tidak paham sinonim |
| RRF Fusion | Gabungkan kedua sinyal | Ranking masih kasar |
| **ColBERT Rerank** | **Token-level interaction yang sangat presisi** | Mahal komputasi, jadi hanya 20 candidates |

**Konfigurasi Retrieval:**

| Parameter | Nilai | Makna |
|---|---|---|
| `TOP_K` | 10 | Jumlah chunk final yang dikirim ke LLM |
| `RERANK_FETCH_MULTIPLIER` | 2 | Fetch 2× TOP_K = 20 dari RRF, rerank ke 10 |
| Dense prefetch limit | 40 | Over-fetch dari dense path |
| Sparse prefetch limit | 40 | Over-fetch dari sparse path |
| Fusion | `Fusion.RRF` | Reciprocal Rank Fusion |

### Prompt — Format OpenAI Messages (Bahasa Indonesia)

Prompt menggunakan format **OpenAI messages** — array objek `{role, content}` yang dikirim ke vLLM melalui OpenAI-compatible API.

```python
def generate_response(question, context, api_url=None):
    client = OpenAI(
        base_url=api_url,
        api_key=os.getenv("LLM_API_KEY", "EMPTY")
    )

    messages = [
        {
            "role": "system",
            "content": """Kamu adalah agen AI asisten admin HPC Slurm yang ahli. ...

Aturan:
0. Sapa user dengan ramah seperti customer service layanan HPC Aleleon
   Supercomputer support yang memiliki hospitality tinggi. Ucapkan salam,
   terima kasih, dan akhiri dengan konfirmasi.
0. Jangan bilang "berdasarkan dokumen referensi yang diberikan"...
1. Jawab HANYA berdasarkan dokumen referensi. KUTIP PERSIS...
2a. Angka/versi harus presisi...
2b. Gunakan penomoran, bukan bullet...
3. Deduksi diperbolehkan...
4. "Saya tidak menemukan informasi tersebut di sistem."
5. Dilarang mengarang...
6. Jangan ganti perintah...
7. Bedakan minimum vs maksimum...
8. Perhatikan label LEGACY...
9. Jawab LENGKAP dengan contoh...
10. Minimal 2 kalimat...""",
        },
        {
            "role": "user",
            "content": f"Dokumen Referensi:\n{context}\n\nPertanyaan: {question}",
        },
    ]

    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"),
        messages=messages,
        max_tokens=8192,
        temperature=0.3,
        top_p=0.9,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    return response.choices[0].message.content
```

**11 Aturan Anti-Halusinasi (0-10):**

| Aturan | Tujuan |
|---|---|
| 0. Sapa user ramah, hospitality tinggi | Customer service yang sopan, ucapkan salam dan terima kasih |
| 0. Langsung jawab, tanpa chain of thought | Tidak menyebut "berdasarkan dokumen", jawab ringkas dan jelas |
| 1. Jawab HANYA dari dokumen, KUTIP PERSIS | Mencegah generasi info dari pengetahuan pra-latih. L1 Support bot ALELEON |
| 2a. Angka/versi harus presisi | Mencegah pembulatan ">=11" menjadi "11.0" |
| 2b. Gunakan penomoran, bukan bullet | Langkah-langkah dalam format 1, 2, 3 |
| 3. Deduksi diperbolehkan | Memungkinkan LLM menyimpulkan secara logis dari data |
| 4. Respons "tidak ditemukan di sistem" | Memaksa model menolak saat info tidak tersedia |
| 5. Dilarang mengarang, termasuk nama partisi | Memblokir perintah, URL, partisi palsu seperti "bigmem" |
| 6. Jangan ganti perintah | Mencegah `source activate` diganti `conda activate` |
| 7. Bedakan minimum vs maksimum | Mencegah salah tafsir "minimal X" dan "maksimal X" |
| 8. Perhatikan label LEGACY | Jangan terapkan info Mk.III untuk Mk.V |
| 9. Jawab LENGKAP dengan contoh | Jangan hanya kalimat pembuka lalu berhenti |
| 10. Minimal 2 kalimat | Mencegah jawaban kosong |

### Generasi — vLLM + Qwen3.5 via OpenAI API

```python
from openai import OpenAI

client = OpenAI(base_url=VLLM_API_URL, api_key="EMPTY")
```

Model dijalankan di container **vllm-rocm** pada AMD GPU menggunakan vLLM dengan OpenAI-compatible API.

**Perintah menjalankan vLLM (dari compose.yml):**

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
    --dtype float16 \
    --enforce-eager \
    --gpu-memory-utilization 0.99 \
    --max-model-len 262144 \
    --max-num-seqs 16 \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --trust-remote-code
```

**Parameter generasi:**

| Parameter | Nilai | Makna |
|---|---|---|
| `temperature=0.3` | Rendah → lebih deterministik, faktual | Terbaik untuk RAG — mengurangi halusinasi |
| `top_p=0.9` | Nucleus sampling — probabilitas 90% teratas | Mengurangi jawaban acak |
| `top_k=20` | Hanya mempertimbangkan 20 token teratas tiap langkah | Semakin membatasi randomness |
| `presence_penalty=1.5` | Penalti kuat untuk token berulang | Mencegah output repetitif |
| `max_tokens=8192` | Maks 8K token output | Cukup untuk jawaban detail |
| `enable_thinking=False` | Menonaktifkan mode "thinking" Qwen3.5 (via `chat_template_kwargs`) | Jawaban langsung tanpa jejak reasoning |
| `--max-model-len 262144` | Maks 256K token total (prompt + output) | Context window penuh untuk prompt besar |
| `--gpu-memory-utilization 0.99` | Gunakan 99% VRAM | Maksimalkan kapasitas model di GPU |
| `--max-num-seqs 16` | Maks 16 sequence paralel | Batasi concurrency untuk stabilitas |
| `--enable-prefix-caching` | Cache prefix prompt yang berulang | Mempercepat inference untuk prompt serupa |
| `--enable-auto-tool-choice` | Aktifkan tool calling otomatis | Untuk fitur tool-use Qwen3.5 |
| `--dtype float16` | Presisi FP16 | Dibutuhkan model GPTQ di ROCm |
| `--enforce-eager` | Menonaktifkan CUDAGraph | Kompatibilitas ROCm / GPU AMD |

**Model: Qwen/Qwen3.5-35B-A3B-GPTQ-Int4**

| Properti | Detail |
|---|---|
| Parameter | 35B total, ~3B aktif (arsitektur MoE) |
| Kuantisasi | GPTQ 4-bit |
| Context Window | 262144 token (256K) |
| Arsitektur | Mixture of Experts (MoE) |
| Disajikan via | vLLM pada GPU AMD ROCm |

### Token Logging & Filter Output LLM

1. **Token Logging:** Setiap pemanggilan API ke LLM mencatat rincian penggunaan token (`prompt_tokens`, `completion_tokens`, `total_tokens`) di logs untuk memonitor biaya komputasi.
2. **Filter `<think>`:** Meskipun `enable_thinking=False` (menonaktifkan model output chain of thought), terkadang model (terutama Qwen3.5 varian Coder) masih menyisipkan tag `<think>...</think>`. Aplikasi menggunakan RegExp (`re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)`) pada setiap hasil LLM untuk membersihkan pemikiran sisa tersebut agar tidak tampil ke user.

### Pelacakan Sumber + Justifikasi "Why This Source"

Setelah setiap jawaban, sistem **menampilkan sumber dokumen** yang dipakai dan **justifikasi relevansi** setiap sumber. Justifikasi di-generate oleh LLM terpisah:

```python
def generate_source_justifications(question, answer, docs, api_url=None):
    """Generate 'Why This Source' justification for each unique retrieved source."""
    # Deduplicate sources by (title, header)
    unique_sources = [...]

    messages = [
        {
            "role": "system",
            "content": "Kamu adalah asisten yang menjelaskan relevansi sumber dokumen. "
                       "Berikan justifikasi singkat (1 kalimat) untuk setiap sumber. "
                       "Jika TIDAK relevan, jawab 'TIDAK RELEVAN'."
        },
        {
            "role": "user",
            "content": f"Pertanyaan: {question}\nJawaban: {answer[:500]}\n\nSumber:\n{source_list}..."
        }
    ]
```

**Filter sumber tidak relevan:** Setelah justifikasi di-generate, sumber yang dijustifikasi sebagai "TIDAK RELEVAN" **dibuang** dari daftar sumber yang ditampilkan ke user.

```
Contoh output:
============================================================
[Q1/69] Bagaimana cara membuat conda environment di aleleon?
------------------------------------------------------------
Selamat datang! Terima kasih telah menghubungi layanan support
ALELEON. Untuk membuat conda environment di ALELEON, silakan
ikuti langkah-langkah berikut...

    📚 Sumber (10 chunks):
    • Komputasi Python dengan Conda Environment User → Membuat Conda Environment
      (https://wiki.efisonlt.com/wiki/Komputasi_Python_dengan_Conda_Environment_User)
      💡 Why: Berisi langkah lengkap pembuatan conda environment di ALELEON
    • Komputasi Python dengan Conda Environment User → Module Pyload
      (https://wiki.efisonlt.com/wiki/Komputasi_Python_dengan_Conda_Environment_User)
      💡 Why: Menjelaskan cara membuat modul pyload setelah conda env aktif
    • ...
```

### RAG Chain — Fungsi Python Kustom (Closure)

Tidak menggunakan `create_stuff_documents_chain` atau `create_retrieval_chain` dari LangChain. Menggunakan fungsi Python kustom dengan pola *closure*:

```python
def create_rag_chain(client: QdrantClient, embeddings: EmbeddingServiceClient, llm_api_url=None):
    """Create RAG chain with hybrid retrieval (dense + sparse + RRF fusion)."""

    def retrieve_and_answer(question):
        # 0. Cek relevansi pertanyaan (tanpa embedding)
        if not is_question_relevant(question, llm_api_url):
            return {"answer": "Pertanyaan anda tidak relevan...", "context": [], "justifications": []}

        # 1. Embed query → dense + sparse (via /embed/multi)
        # 2. Hybrid search: dense + sparse → RRF fusion
        # 3. ColBERT reranking: 20 candidates → top 10
        # 4. Generate response using vLLM API
        # 5. Generate "Why This Source" justifications
        # 6. Filter out irrelevant sources

        return {
            "answer": answer,
            "context": filtered_docs,
            "justifications": filtered_justifications
        }

    return retrieve_and_answer
```

```
Pertanyaan Pengguna
    │
    ▼
┌────────────────────┐
│is_question_relevant│ → TIDAK → return "tidak relevan"
│ (LLM filter)       │
└─────────┬──────────┘
          │ YA
    ┌─────┴───────────────┐     ┌───────────────────────┐
    │ embed_query_multi() │ ──→ │ query_points()        │
    │ (dense + sparse)    │     │ (Prefetch dense +     │
    │                     │     │  Prefetch sparse +    │
    │                     │     │  RRF Fusion)          │
    └─────────────────────┘     └───────────┬───────────┘
                                            │ 20 candidates
                                ┌───────────┴────────────┐
                                │ rerank()               │
                                │ (ColBERT scoring)      │
                                │ → top 10 docs          │
                                └───────────┬────────────┘
                                            │
                                ┌───────────┴────────────┐
                                │ generate_response()    │
                                │(OpenAI messages → vLLM)│
                                └───────────┬────────────┘
                                            │
                                ┌───────────┴────────────┐
                                │ generate_source_       │
                                │ justifications()       │
                                │ → filter TIDAK RELEVAN │
                                └───────────┬────────────┘
                                            │
                                            ▼
                                return {answer, context, justifications}
```

**Mengapa fungsi kustom, bukan chain LangChain?**

- Lebih transparan — bisa di-debug dengan print statement
- Hybrid retrieval memerlukan `QdrantClient.query_points()` langsung (bukan `as_retriever()`)
- Mendukung ColBERT reranking sebagai post-processing
- Mendukung source justification dan filtering
- `generate_response()` menggunakan OpenAI client langsung

---

## FASE 4: Fitur Tambahan

### Incremental Sync — Sinkronisasi Sitemap

Saat startup API dan via endpoint `POST /refresh`, sistem melakukan **incremental sync** antara sitemap wiki terkini dan data di Qdrant:

```python
def sync_vectorstore(client, embeddings):
    """
    Incremental sync: bandingkan sitemap terkini dengan data di Qdrant,
    scrape + embed + upsert hanya halaman yang berubah/baru.
    Hapus halaman yang sudah tidak ada di sitemap.
    """
    # 1. Parse sitemap terkini → {url: lastmod}
    sitemap_data = parse_sitemap(SITEMAP_URL)

    # 2. Get stored state dari Qdrant → {url: lastmod}
    stored_data = get_stored_sitemap_state(client)

    # 3. Bandingkan
    new_urls = sitemap_urls - stored_urls          # Halaman baru
    deleted_urls = stored_urls - sitemap_urls       # Halaman dihapus
    changed_urls = {url for url in common_urls      # Halaman diubah
                    if sitemap_data[url] != stored_data[url]}

    # 4. Hapus points untuk URL yang berubah atau dihapus
    # 5. Scrape + embed + upsert URL baru dan yang berubah
```

```
Sync Flow:
┌─────────────────┐       ┌──────────────────┐
│ Sitemap terkini │       │  Qdrant stored   │
│ (URL + lastmod  │       │  (URL + lastmod) │
└────────┬────────┘       └────────┬─────────┘
         │                         │
         └──────────┬──────────────┘
                    │ Bandingkan
         ┌──────────┴──────────┐
         │ New: 3 URL          │ → Scrape + embed + upsert
         │ Updated: 2 URL      │ → Delete old + scrape + embed + upsert
         │ Deleted: 1 URL      │ → Delete from Qdrant
         │ Unchanged: 45 URL   │ → Skip
         └─────────────────────┘
```

### Hybrid Script Review — Review Skrip Bash/Slurm

Endpoint `POST /review-script` melakukan **review skrip 3 tahap** (hybrid):

```python
def review_script_hybrid(script_content, api_url=None, qdrant_client=None, embeddings=None):
    """
    1. LLM ekstrak resource parameters dari skrip (#SBATCH directives)
    2. Jika ada resource params → retrieval kebijakan HPC dari Qdrant
    3. LLM review teknis + validasi kebijakan berdasarkan dokumen
    """
```

```
Script Slurm dari user
       │
  ┌────┴─────────────────────────────────────────────┐
  │ STEP 0: Cek batas maksimal panjang skrip         │
  │ (Maksimal 10.000 karakter, jika lebih ditolak)   │
  └────┬─────────────────────────────────────────────┘
       │
  ┌────┴─────────────────────────────────────────────┐
  │ STEP 1: extract_resource_params()                │
  │ LLM → parsing #SBATCH → JSON                     │
  │ {"partition": "ampere", "mem": "64G", ...}       │
  └────┬─────────────────────────────────────────────┘
       │
  ┌────┴─────────────────────────────────────────────┐
  │ STEP 2: retrieve_policy_context()                │
  │ Berdasarkan params → targeted queries ke Qdrant  │
  │ Misal: "kapasitas RAM partisi ampere"            │
  │ → Retrieve kebijakan HPC yang relevan            │
  └────┬─────────────────────────────────────────────┘
       │
  ┌────┴─────────────────────────────────────────────┐
  │ STEP 3: LLM Review                               │
  │ Review teknis (syntax, best practice) +          │
  │ Validasi kebijakan (limit partisi, walltime, dll)│
  │ → Output: review text + issues count +           │
  │   policy sources + template skrip standar        │
  └────┬─────────────────────────────────────────────┘
       │
  ┌────┴────────────────────────────────────────────┐
  │ STEP 4: generate_source_justifications()         │
  │ LLM menilai tiap policy source. Jika             │
  │ "TIDAK RELEVAN", buang. Dibatasi maks 10 sumber. │
  └──────────────────────────────────────────────────┘
```

**Review mencakup:**
1. Syntax Bash (shebang, quoting, variable expansion)
2. Format #SBATCH (spasi, satuan, double dash)
3. Best practice Slurm (format --mem, --time, dll.)
4. Potensi error (variabel tidak didefinisikan, path salah)
5. Keamanan (`rm -rf` tanpa konfirmasi, hardcoded password)
6. **Validasi kebijakan HPC ALELEON** (dari dokumen di Qdrant — limit partisi, walltime, RAM, dll.)

**Template Standar ALELEON:**
LLM akan memformat ulang skrip ke dalam template standar ALELEON jika perlu perbaikan. Template menggunakan format:
```bash
#!/bin/bash
# --------------------------------------------------
# [NAMA SOFTWARE/PROGRAM]
# rev.[TANGGAL]
# ...
#SBATCH --partition=////
#SBATCH --cpus-per-task=////
#SBATCH --mem=////GB
#SBATCH --time=////
# ...
```

### Telegram Bot — Interface Chat

Bot Telegram menyediakan interface chat yang terhubung ke RAG API:

| Command | Fungsi |
|---|---|
| `/start` | Pesan selamat datang |
| `/ask <pertanyaan>` | Tanya jawab RAG (standard question) |
| `/askscript <skrip>` | Review skrip Bash/Slurm (paste teks) |
| Upload file `.sh`/`.slurm`/`.sbatch`/`.bash` | Review skrip dari file upload |
| `/status` | Cek status RAG API |
| `/help` | Bantuan penggunaan |

**Fitur:**
- Animasi placeholder saat menunggu respons RAG (typing indicator + rotating text)
- Konversi Markdown → HTML untuk Telegram (code blocks, bold, italic, headings, links)
- Auto-split pesan panjang (>4000 karakter) dengan HTML tag tracking
- Fallback ke plain text jika HTML parsing gagal
- Justifikasi sumber ditampilkan dengan emoji 💡

### REST API — FastAPI Endpoints (v1.3.0)

```
POST /ask
  Request:  {"question": "Bagaimana cara membuat conda environment?"}
  Response: {"answer": "...", "sources": [{title, source_url, section, justification}]}

POST /review-script
  Request:  {"script": "#!/bin/bash\n#SBATCH --mem= 64 GB\nsrun gmx_mpi mdrun"}
  Response: {"review": "...", "issues_found": N, "policy_sources": [...]}

POST /refresh
  Response: {"status": "started", "message": "Sync dimulai di background..."}

GET /refresh/status
  Response: {"running": false, "last_result": {new, updated, deleted, unchanged}, ...}

GET /health
  Response: {"status": "ready", "service": "rag-api"}

GET /
  Response: {"service": "ALELEON HPC RAG API", "version": "1.3.0", "endpoints": {...}}
```

**Fitur API:**

| Fitur | Detail |
|---|---|
| **Inference Concurrency Limiter** | `asyncio.Semaphore(2)` — maksimal 2 request inference berjalan paralel. Request ke-3+ di-queue (menunggu), bukan ditolak/connection reset. Nilai 2 dipilih berdasarkan benchmark: concurrency 2 = sweet spot (0% failure, throughput optimal 0.051 req/s). |
| **Non-blocking Inference** | `asyncio.to_thread()` — setiap panggilan blocking (`rag_chain()`, `review_script_hybrid()`, `is_question_relevant()`) dijalankan di thread pool agar event loop uvicorn tetap bisa menerima koneksi. |
| **Relevance Filter untuk /review-script** | Sebelum review, skrip divalidasi dengan `is_question_relevant()` — jika skrip tidak relevan dengan HPC, langsung ditolak tanpa membuang resource LLM. |
| **Question Logging** | Setiap pertanyaan di `/ask` dicatat ke `logs/user_questions.logs` dengan timestamp UTC. |
| **Startup Sync** | Saat API mulai, otomatis cek perubahan sitemap via `sync_vectorstore()`. |
| **Background Sync** | `POST /refresh` menjalankan sync di background thread (daemon), tidak blocking API. Dilindungi `threading.Lock()` agar tidak ada sync duplikat. |

```python
# Inference Concurrency Limiter
_inference_semaphore = asyncio.Semaphore(2)

@app.post("/ask")
async def ask(req: AskRequest):
    _log_question(req.question)  # Log ke file
    async with _inference_semaphore:  # Queue jika sudah ada 2 request
        result = await asyncio.to_thread(rag_chain, req.question)  # Non-blocking
    ...

@app.post("/review-script")
async def review_script_endpoint(req: ReviewScriptRequest):
    async with _inference_semaphore:
        # Pre-filter: cek relevansi skrip sebelum review
        is_relevant = await asyncio.to_thread(is_question_relevant, req.script, llm_api_url)
        if not is_relevant:
            return ReviewScriptResponse(review="Skrip tidak relevan...", issues_found=0)
        result = await asyncio.to_thread(review_script_hybrid, req.script, ...)
    ...
```

```
Concurrency Flow:
  Request 1 (→ acquire semaphore) [██████████ processing ]
  Request 2 (→ acquire semaphore) [      ██████████ processing ]
  Request 3 (→ waiting...)        [              ▒▒▒▒ queue ████████ processing]
                                    ↑                 ↑
                              Semaphore(2)      Request 1 selesai,
                              penuh             slot tersedia
```

### Menunggu vLLM — Pemeriksaan Kesehatan

Sebelum memulai RAG, sistem menunggu vLLM siap:

```python
def wait_for_vllm(api_url, timeout=600, interval=10):
    base_url = api_url.rstrip("/").removesuffix("/v1")
    health_url = f"{base_url}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)
    raise RuntimeError(f"vLLM tidak ready setelah {timeout}s")
```

Model besar (35B params) memerlukan waktu loading ke VRAM. Fungsi ini polling `/health` setiap 10 detik, timeout setelah 10 menit.

---

## Diagram End-to-End Lengkap

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FASE STARTUP                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [0] wait_for_vllm() — polling /health tiap 10 dtk (maks 10 mnt)    │
│         │                                                           │
│         ▼                                                           │
│  [1] qdrant_collection_exists()? ── YA ──→ load_vectorstore()       │
│         │                                   (skip to [7])           │
│         TIDAK                                                       │
│         │                                                           │
│  Wiki Sitemap XML                                                   │
│  (https://wiki.efisonlt.com/sitemap/sitemap-wiki.efisonlt.com-0.xml)│
│         │                                                           │
│  [2] Parse XML → filter non-webpage → ekstrak URL halaman wiki      │
│         │                                                           │
│  [3] Untuk setiap URL:                                              │
│      requests.get() → BeautifulSoup                                 │
│      → extract <div id="mw-content-text">                           │
│         │                                                           │
│  [4] HTMLSectionSplitter (split berdasarkan heading h1/h2/h3)       │
│      → Fallback: RecursiveCharacterTextSplitter                     │
│        (4500 chars, 900 overlap)                                    │
│         │                                                           │
│  [5] Add source labels:                                             │
│      "[Sumber: title] [Section: header]"                            │
│         │                                                           │
│  ~450 Chunks                                                        │
│         │                                                           │
│  [6] BAAI/bge-m3 via API /embed/multi                               │
│      Dibatch @16 chunk per request                                  │
│      Each chunk → Dense vector (1024D) + Sparse vector              │
│         │                                                           │
│      build_vectorstore() →                                          │
│  [7] Qdrant Hybrid Collection (http://qdrant:6333)                  │
│      Collection: "wiki_aleleon_qdrant"                              │
│      ~450 points: dense + sparse + text + metadata                  │
│      Podman volume: qdrant-data:/qdrant/storage                     │
│         │                                                           │
│  [8] sync_vectorstore() — incremental sync sitemap                  │
│      (pada startup API dan via POST /refresh)                       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                  FASE PER PERTANYAAN                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User: "Bagaimana cara membuat conda env?"                          │
│  (via Telegram /ask, REST API POST /ask, atau CLI)                  │
│         │                                                           │
│  [a] is_question_relevant() — LLM filter (YA/TIDAK)                 │
│         │ YA                                                        │
│  [b] Embed question → Dense + Sparse                                │
│      (BAAI/bge-m3 via /embed/multi)                                 │
│         │                                                           │
│  [c] Hybrid search: Dense path + Sparse path                        │
│      → RRF Fusion → 20 candidates                                   │
│         │                                                           │
│  [d] ColBERT reranking (/rerank) → top 10 docs                      │
│         │                                                           │
│  [e] generate_response() → OpenAI messages format                   │
│      with 11 anti-hallucination rules (0-10)                        │
│      + hospitality tone                                             │
│         │                                                           │
│  [f] Send to Qwen3.5-35B-A3B-GPTQ-Int4 via vLLM                     │
│      (OpenAI-compatible API, AMD ROCm GPU)                          │
│      temperature=0.3, presence_penalty=1.5                          │
│         │                                                           │
│  [g] Model menghasilkan jawaban                                     │
│         │                                                           │
│  [h] generate_source_justifications()                               │
│      LLM → 1 kalimat justifikasi per sumber                         │
│      → Filter "TIDAK RELEVAN"                                       │
│         │                                                           │
│  [i] Display answer + source attribution + justifications           │
│      (de-duplicated title/section/URL + 💡 Why)                     │
│         │                                                           │
│         ▼                                                           │
│  "Selamat datang! Terima kasih telah menghubungi layanan            │
│   support ALELEON. Untuk membuat conda environment:                 │
│   1. module load anaconda3/2025.06-1                                │
│   2. conda create -n myenv python=3.12..."                          │
│                                                                     │
│      📚 Sumber (10 chunks):                                         │
│      • Conda Environment User → Membuat Conda Environment           │
│        (https://wiki.efisonlt.com/wiki/...)                         │
│        💡 Why: Berisi langkah lengkap pembuatan conda env           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```