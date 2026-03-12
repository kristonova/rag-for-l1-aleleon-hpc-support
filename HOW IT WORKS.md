# Penjelasan Lengkap Logika Kode RAG

## Arsitektur Keseluruhan

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Pipeline RAG (Kontainer Podman)                       │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐      │
│  │   INGESTI    │→ │  EMBEDDING   │→ │ RETRIEVAL │→ │  GENERASI   │      │
│  │  (HTML Wiki) │  │ + Penyimpanan│  │  (Pencarian)│ │   (LLM)     │      │
│  └──────────────┘  └──────────────┘  └──────────┘  └─────────────┘      │
│                                                                          │
│  Fase 1: Ambil & Split   Fase 2: Vektorisasi   Fase 3: Menjawab          │
│                                                                          │
│  Layanan:                                                                │
│  ┌─────────────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────────┐      │
│  │ embedding-service│ │  vllm-rocm  │ │ chromadb │ │   rag-app    │      │
│  │ (BAAI/bge-m3)   │ │ (Qwen3.5)   │ │ (Vektor) │ │ (Orkestrator)│      │
│  │ Port 8001       │ │ Port 8000   │ │ Port 8002│ │              │      │
│  └─────────────────┘ └─────────────┘ └──────────┘ └──────────────┘      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## FASE 1: Pipeline Ingesti Data

### Pemeriksaan Persistensi — Lewati Scraping jika ChromaDB Sudah Ada

```python
if chroma_db_exists():
    vectorstore = load_vectorstore(embeddings)
else:
    vectorstore = build_vectorstore(embeddings)
```

**Yang terjadi:**
- Sebelum scraping, sistem memeriksa apakah direktori ChromaDB sudah ada di disk (`./chroma_db`).
- Jika ada dan berisi data → **scraping dilewati sepenuhnya** dan vektor dimuat dari disk.
- Jika belum ada → lanjutkan ke pipeline ingesti lengkap di bawah.
- ChromaDB dipersistenkan melalui named volume Podman (`rag-chroma-db:/app/chroma_db`), sehingga data tetap ada saat container restart.
- Untuk memaksa scraping ulang (misalnya konten wiki berubah): hapus volume dengan `podman volume rm rag-for-l1-aleleon-hpc-support_rag-chroma-db`.

### Langkah 1 — Parse Sitemap XML

```python
splits = load_wiki_documents(
    sitemap_url="https://wiki.efisonlt.com/sitemap/sitemap-wiki.efisonlt.com-NS_0-0.xml",
    requests_per_second=2,
)
```

**Yang terjadi:**
- Fungsi mengambil file **sitemap XML** wiki.
- XML diparse untuk mengekstrak semua URL `<loc>` — ini adalah seluruh alamat halaman wiki.
- Dibatasi 2 request per detik agar tidak membebani server wiki.

```python
resp = requests.get(sitemap_url)
root = ElementTree.fromstring(resp.content)
ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
urls = [loc.text for loc in root.findall(".//ns:loc", ns)]
```

```
Sitemap XML
    │
    ▼  Parse tag <loc>
┌──────────────────────────────────────────────────┐
│ URL 1: https://wiki.efisonlt.com/wiki/Spesifikasi│
│ URL 2: https://wiki.efisonlt.com/wiki/Conda_Env  │
│ URL 3: https://wiki.efisonlt.com/wiki/MPI_Guide  │
│ ...                                              │
└──────────────────────────────────────────────────┘
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
│   <h3>Compute Node</h3>     │
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

## FASE 2: Embedding + Basis Data Vektor

### Langkah 6 — Embedding Vektor via Layanan API

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

**Arsitektur:**

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

**Model yang digunakan: `BAAI/bge-m3`**

| Properti | Detail |
|---|---|
| Arsitektur | Berbasis XLM-RoBERTa (transformer multibahasa) |
| Parameter | ~568M |
| Dimensi Output | **1024 dimensi** |
| Panjang Sekuens Maksimum | 8192 token |
| Bahasa | 100+ bahasa termasuk **Bahasa Indonesia** |
| Fitur | Dense + Sparse + ColBERT multi-vector retrieval |
| Berjalan di | CPU atau GPU, disajikan via container embedding-service |

**Mengapa `BAAI/bge-m3` dibanding `intfloat/multilingual-e5-large`?**

| Fitur | multilingual-e5-large (lama) | BAAI/bge-m3 (saat ini) |
|---|---|---|
| Dimensi | 1024 | **1024** (sama) |
| Token Maksimum | 512 | **8192** (konteks 16x lebih panjang) |
| Perlu Prefix | Ya ("query: " / "passage: ") | **Tidak** (tanpa prefix) |
| Mode Retrieval | Hanya dense | **Dense + Sparse + ColBERT** |
| Skor MTEB | Kuat | **Lebih kuat** (state-of-the-art multibahasa) |

BGE-M3 tidak memerlukan prefix "query: " atau "passage: " seperti E5, sehingga kode lebih sederhana — teks dikirim langsung tanpa modifikasi.

**Apa itu embedding?**

Embedding mengubah teks menjadi **vektor angka** di ruang 1024 dimensi. Teks dengan **makna serupa** akan memiliki vektor yang **berdekatan**.

```
"Cara membuat conda environment di ALELEON"
        │
        ▼  BAAI/bge-m3
[0.032, -0.118, 0.245, ..., 0.067]    ← 1024 angka

"Bagaimana membuat conda env baru?"
        │
        ▼  BAAI/bge-m3
[0.029, -0.121, 0.238, ..., 0.071]    ← 1024 angka (MIRIP!)

"Berapa harga berlangganan ALELEON?"
        │
        ▼  BAAI/bge-m3
[-0.156, 0.089, -0.034, ..., 0.193]   ← 1024 angka (JAUH!)
```

**Ini BUKAN TF-IDF atau BM25.**

| Metode | Cara kerja | Digunakan di kode ini? |
|---|---|---|
| **TF-IDF** | Menghitung frekuensi kata. "conda" muncul 3x = relevan. Tidak memahami makna. | ❌ |
| **BM25** | TF-IDF lanjutan dengan normalisasi panjang dokumen. | ❌ |
| **Sparse Retrieval** | Vektor besar, sebagian besar nol. Mencocokkan kata kunci. | ❌ |
| **Dense Retrieval** ✅ | Teks → vektor dense 1024D via neural network. Mencocokkan **makna**. | ✅ **Dipakai di sini** |

**Kelebihan Dense Retrieval:**

```
Kueri: "Saya butuh banyak memori untuk job saya"
  │
    ├── TF-IDF/BM25: Cari kata "memori" → TIDAK DITEMUKAN (dokumen memakai "RAM")
  │
    └── Dense (bge-m3): Memahami "memori" ≈ "RAM" secara semantik → DITEMUKAN ✅
```

### Langkah 7 — Basis Data Vektor (Chroma — Persisten)

```python
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=CHROMA_PERSIST_DIR,
    collection_name=CHROMA_COLLECTION_NAME,
)
```

**Yang terjadi:**

1. Each chunk is embedded into a 1024D vector (via embedding-service API, in batches of 32).
2. Vektor + teks asli + metadata disimpan ke database Chroma **di disk** (persisten).

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

**Chroma** adalah database vektor yang:
- Lightweight, runs as **persistent local storage** (using `persist_directory`).
- Data survives container restarts via Podman named volume (`rag-chroma-db:/app/chroma_db`).
- Supports **cosine similarity search**.
- Pada run pertama: scraping + embedding + penyimpanan (~450 chunk). Pada run berikutnya: langsung load dari disk.

---

## FASE 3: Retrieval + Generasi

### Retrieval — Cari Chunk yang Relevan

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

**Tipe retrieval: Approximate Nearest Neighbor (ANN) dengan kesamaan kosinus**

Ketika pengguna mengajukan pertanyaan, prosesnya adalah:

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
         ▼ Ambil Top-K (k=10)
    Top 10 chunk → dikirim ke LLM sebagai konteks
```

**Mengapa k=10?** Memberikan lebih banyak konteks ke LLM sehingga jawaban lebih lengkap. Qwen3.5-35B memiliki context window 131072 token, cukup untuk menampung 10 chunk.

**Rumus Cosine Similarity:**

```
                    A · B           Σ(Aᵢ × Bᵢ)
cos(θ) = ─────────────────── = ─────────────────────
              ||A|| × ||B||     √Σ(Aᵢ²) × √Σ(Bᵢ²)

Hasil: -1 (berlawanan) sampai +1 (identik)
```

### Prompt — Format OpenAI Messages (Bahasa Indonesia)

Prompt tidak lagi menggunakan string template ChatML. Sekarang menggunakan format **OpenAI messages** — array objek `{role, content}` yang dikirim ke vLLM melalui OpenAI-compatible API.

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

**Format: OpenAI Messages (bukan string ChatML)**

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

**11 Aturan Anti-Halusinasi (0-10):**

| Aturan | Tujuan |
|---|---|
| 0. Bahasa Indonesia | Memastikan jawaban dalam Bahasa Indonesia |
| 1. Jawab HANYA dari dokumen | Mencegah generasi info dari pengetahuan pra-latih |
| 2. Angka/versi harus presisi | Mencegah pembulatan ">=11" menjadi "11.0" |
| 3. Deduksi diperbolehkan | Memungkinkan LLM menyimpulkan secara logis dari data |
| 4. Respons "tidak ditemukan" | Memaksa model menolak saat info tidak tersedia |
| 5. Dilarang mengarang | Memblokir perintah, URL, atau prosedur palsu |
| 6. Jangan ganti perintah | Mencegah `source activate` diganti `conda activate` |
| 7. Bedakan minimum vs maksimum | Mencegah salah tafsir "minimal X" dan "maksimal X" |
| 8. Urutan harus benar | Langkah harus dalam URUTAN yang BENAR dari konteks |
| 9. Prosedur harus lengkap | Jangan memotong/meringkas instruksi langkah demi langkah |
| 10. Informasi lengkap | Sertakan SELURUH informasi dari dokumen secara utuh |

### Generasi — vLLM + Qwen3.5 via OpenAI API

```python
from openai import OpenAI

client = OpenAI(base_url=VLLM_API_URL, api_key="not-needed")
```

Model dijalankan di container **vllm-rocm** pada AMD GPU menggunakan vLLM dengan OpenAI-compatible API.

**Perintah menjalankan vLLM (dari compose.yml):**

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
    --dtype float16 \
    --enforce-eager \
    --max-model-len 131072
```

**Alur generasi:**

```
┌──────────────────────────────────────────────────────────────┐
│ Panggilan OpenAI API ke vLLM:                                │
│                                                              │
│ client.chat.completions.create(                              │
│   model="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",                   │
│   messages=[                                                 │
│     {"role": "system", "content": "Kamu adalah agen AI...    │
│      Aturan: 0-10 (11 aturan anti-halusinasi)"},             │
│     {"role": "user", "content": "Dokumen Referensi:\n...     │
│      Pertanyaan: Bagaimana cara membuat conda environment?"}│
│   ],                                                         │
│   temperature=0.3, top_p=0.9, max_tokens=32768,             │
│   extra_body={top_k=20, presence_penalty=1.5,                │
│               enable_thinking=False}                         │
│ )                                                            │
│         │                                                    │
│         ▼ vLLM mengonversi ke ChatML + menghasilkan jawaban  │
│                                                              │
│ "Untuk membuat conda environment di ALELEON,                 │
│  jalankan perintah berikut:                                  │
│  1. module load anaconda3/2025.06-1                          │
│  2. conda create -n myenv python=3.12..."                    │
└──────────────────────────────────────────────────────────────┘
```

**Parameter generasi:**

| Parameter | Nilai | Makna |
|---|---|---|
| `temperature=0.3` | Rendah → lebih deterministik, faktual | Terbaik untuk RAG — mengurangi halusinasi |
| `top_p=0.9` | Nucleus sampling — probabilitas 90% teratas | Mengurangi jawaban acak |
| `top_k=20` | Hanya mempertimbangkan 20 token teratas tiap langkah | Semakin membatasi randomness |
| `presence_penalty=1.5` | Penalti kuat untuk token berulang | Mencegah output repetitif |
| `max_tokens=32768` | Maks 32K token output | Memungkinkan jawaban sangat detail |
| `enable_thinking=False` | Menonaktifkan mode "thinking" Qwen3.5 | Jawaban langsung tanpa jejak reasoning |
| `--max-model-len 131072` | Maks 128K token total (prompt + output) | Context window penuh untuk prompt besar |
| `--dtype float16` | Presisi FP16 | Dibutuhkan model GPTQ di ROCm |
| `--enforce-eager` | Menonaktifkan CUDAGraph | Kompatibilitas ROCm / GPU AMD |

**Model: Qwen/Qwen3.5-35B-A3B-GPTQ-Int4**

| Properti | Detail |
|---|---|
| Parameter | 35B total, ~3B aktif (arsitektur MoE) |
| Kuantisasi | GPTQ 4-bit |
| Context Window | 131072 token (128K) |
| Arsitektur | Mixture of Experts (MoE) |
| Disajikan via | vLLM pada GPU AMD ROCm |

### Pelacakan Sumber — Menampilkan Sumber Dokumen

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

**Yang terjadi:**

Setelah setiap jawaban, sistem menampilkan halaman dan section wiki yang dipakai untuk menghasilkan respons. **De-duplication** diterapkan agar pasangan sumber/section yang sama hanya tampil sekali.

```
Contoh output:
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

### RAG Chain — Fungsi Python Kustom

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

**Strategi: "Stuff" (manual)**

Sama seperti sebelumnya — semua chunks digabung ke 1 prompt. Bedanya, sekarang dilakukan secara eksplisit dengan Python, bukan via LangChain chain abstraction.

```
Pertanyaan Pengguna
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

**Mengapa fungsi kustom, bukan chain LangChain?**

- Lebih transparan — bisa di-debug dengan print statement
- Tidak perlu `langchain_classic` dependency
- Mudah dikustomisasi (filter, reranking, etc.)
- `generate_response()` menggunakan OpenAI client langsung

### Menunggu vLLM — Pemeriksaan Kesehatan

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

## Diagram End-to-End Lengkap

```
┌─────────────────────────────────────────────────────────────┐
│                     FASE STARTUP                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [0] wait_for_vllm() — polling /health tiap 10 dtk (maks 10 mnt) │
│         │                                                   │
│         ▼                                                   │
│  [1] chroma_db_exists()? ────── YA ──→ load_vectorstore()   │
│         │                                   (skip to [7])  │
│         TIDAK                                               │
│         │                                                   │
│  Wiki Sitemap XML                                           │
│  (https://wiki.efisonlt.com/sitemap/...)                    │
│         │                                                   │
│  [2] Parse XML → ekstrak semua URL halaman wiki            │
│         │                                                   │
│  [3] Untuk setiap URL:                                      │
│      requests.get() → BeautifulSoup                         │
│      → extract <div id="mw-content-text">                  │
│         │                                                   │
│  [4] HTMLSectionSplitter (split berdasarkan heading h1/h2/h3) │
│      → Fallback: RecursiveCharacterTextSplitter             │
│        (4500 chars, 900 overlap)                            │
│         │                                                   │
│  [5] Add source labels:                                     │
│      "[Sumber: title] [Section: header]"                   │
│         │                                                   │
│  ~450 Chunks                                                │
│         │                                                   │
│  [6] BAAI/bge-m3 via API embedding-service                  │
│      Dibatch @32 chunk per request                          │
│      Each chunk → 1024-dimensional vector                  │
│         │                                                   │
│      build_vectorstore() →                                  │
│  [7] Chroma DB (persistent — ./chroma_db)                  │
│      ~450 vectors + texts + metadata stored on disk        │
│      Podman volume: rag-chroma-db:/app/chroma_db           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                  FASE PER PERTANYAAN                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User: "Bagaimana cara membuat conda env?"                  │
│         │                                                   │
│  [a] Embed question → 1024D vector                         │
│      (BAAI/bge-m3 via embedding-service API)               │
│         │                                                   │
│  [b] Cosine similarity vs ~450 chunk di Chroma             │
│         │                                                   │
│  [c] Ambil 10 chunk paling relevan                          │
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
│  [g] Model menghasilkan jawaban                              │
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