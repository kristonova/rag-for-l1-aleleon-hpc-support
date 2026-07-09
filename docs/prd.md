# Product Requirements Document (PRD)

# RAG-ALELEON: RAG for L1 ALELEON HPC Support

| | |
|---|---|
| **Author** | Krisostomus Nova RAHMANTO (krisostomus.nova.r@efisonlt.com) |
| **Institution** | EURECOM, Sophia Antipolis, France |
| **Type** | Master Thesis Project |
| **Version** | 1.3.0 |
| **Last Updated** | 2026-07-09 |
| **Status** | Active Development |

---

## 1. Executive Summary

RAG-ALELEON adalah sistem asisten AI berbasis Retrieval-Augmented Generation (RAG) yang membantu tim L1 HPC Support menjawab pertanyaan teknis dan kebijakan Supercomputer ALELEON secara cepat, akurat, dan berbasis dokumen. Sistem ini mengingesti halaman wiki resmi HPC, lalu menggunakan pipeline hybrid retrieval dan inferensi LLM lokal untuk menghasilkan jawaban yang selalu bersumber dari dokumentasi — mengurangi waktu resolusi, meminimalkan eskalasi, dan menjaga konsistensi jawaban antar tim support.

**Product Vision:** Menjadi asisten AI yang andal, akurat, dan selalu mutakhir untuk mendukung tim L1 HPC Support dalam menjawab pertanyaan teknis dan kebijakan Supercomputer ALELEON — mengurangi ketergantungan pada eskalasi manual serta mempercepat waktu resolusi.

---

## 2. Problem Statement & Target Users

### Problem Statement

Tim L1 HPC Support ALELEON menghadapi volume pertanyaan berulang yang tinggi dari pengguna supercomputer (peneliti, akademisi, engineer) mengenai environment Conda, job Slurm, troubleshooting, kebijakan partisi, kuota, dan prosedur administrasi lainnya.

Jawaban yang konsisten dan akurat bergantung pada dokumentasi wiki yang ekstensif, membutuhkan waktu untuk mencari dan memverifikasi informasi. Tidak adanya sistem berbasis pengetahuan yang terotomatisasi menyebabkan:

- **Waktu respons lambat** — pertanyaan umum tetap memerlukan pencarian manual di wiki
- **Inkonsistensi jawaban** — antar anggota tim support memberikan jawaban yang berbeda
- **Eskalasi tidak perlu** — pertanyaan yang sudah terdokumentasi tetap di-eskalasi ke L2/L3
- **Beban kognitif tinggi** — L1 Support harus mengingat detail teknis dari 69+ topik dokumentasi

### Target Users

| Persona | Deskripsi | Kebutuhan Utama |
|---|---|---|
| **L1 HPC Support Engineer** | Tim front-line support yang menangani pertanyaan pengguna sehari-hari | Jawaban cepat, akurat, berbasis dokumen untuk pertanyaan teknis HPC; review script Slurm |
| **End User / Peneliti ALELEON** | Pengguna supercomputer yang mengakses cluster untuk komputasi | Panduan self-service untuk troubleshooting, pembuatan environment, job submission |
| **Developer / Maintainer Sistem** | Engineer yang memelihara dan mengembangkan sistem RAG | API yang terdokumentasi, benchmark untuk evaluasi performa, incremental sync untuk update knowledge base |

---

## 3. User Stories

| ID | Persona | Story | Prioritas |
|---|---|---|---|
| US-01 | L1 Support | Sebagai L1 Support, saya ingin bertanya tentang kebijakan HPC via Telegram, sehingga saya bisa menjawab user tanpa membuka wiki manual | P0 |
| US-02 | L1 Support | Sebagai L1 Support, saya ingin me-review script Slurm user sebelum submission, sehingga saya bisa mengidentifikasi parameter yang melanggar kebijakan partisi | P0 |
| US-03 | End User | Sebagai peneliti, saya ingin bertanya cara membuat conda environment, sehingga saya bisa mulai bekerja tanpa menunggu respons support | P0 |
| US-04 | End User | Sebagai peneliti, saya ingin mengupload file .sh/.slurm ke Telegram Bot untuk direview, sehingga saya bisa memperbaiki script sebelum submit ke Slurm | P1 |
| US-05 | Developer | Sebagai developer, saya ingin mengakses RAG via REST API, sehingga saya bisa mengintegrasikan dengan sistem lain | P1 |
| US-06 | Developer | Sebagai developer, saya ingin menjalankan benchmark retrieval, sehingga saya bisa mengukur dan membandingkan kualitas retrieval berbagai strategi | P1 |
| US-07 | Operator | Sebagai operator, saya ingin men-trigger sync knowledge base tanpa restart, sehingga informasi terbaru dari wiki langsung tersedia | P1 |
| US-08 | End User | Sebagai peneliti, saya ingin mendapat jawaban dengan sumber dokumen yang jelas, sehingga saya bisa memverifikasi informasi yang diberikan | P0 |

---

## 4. System Scope & User Roles

### In Scope

| Area | Detail |
|---|---|
| **Knowledge Ingestion** | Scraping otomatis halaman wiki HPC dari sitemap XML, filtering non-webpage, extraction konten HTML, structure-based splitting (HTMLSectionSplitter per h1/h2/h3) dengan fallback RecursiveCharacterTextSplitter (chunk_size: 4500, overlap: 900) |
| **Embedding & Vector Storage** | Multi-mode embedding (dense 1024-dim + sparse lexical weights) via BAAI/bge-m3; hybrid collection (dense + text-sparse) di Qdrant; batched ingestion (32 texts/batch) |
| **Hybrid Retrieval (4-Tahap)** | (1) Dense cosine similarity → (2) Sparse keyword match → (3) RRF Fusion (Reciprocal Rank Fusion) dengan over-fetch 2× → (4) ColBERT reranking via `/rerank` endpoint → top-10 chunks |
| **LLM Question Answering** | Inferensi LLM lokal via vLLM pada AMD ROCm GPU dengan model Qwen3.5-35B-A3B-GPTQ-Int4, sistem prompt 11 aturan anti-halusinasi dalam Bahasa Indonesia |
| **Script Review (3-Tahap)** | (1) Ekstraksi parameter #SBATCH via LLM → (2) Retrieval kebijakan HPC dari knowledge base → (3) Review teknis + validasi kebijakan |
| **Relevance Pre-Filter** | LLM-based filter (`is_question_relevant`) untuk membuang pertanyaan off-topic sebelum proses embedding/retrieval; menghemat siklus inferensi |
| **Source Justification & Filtering** | Post-generation LLM untuk menjelaskan relevansi setiap sumber dokumen; filter otomatis membuang sumber yang dinilai "TIDAK RELEVAN" dari output |
| **Incremental Sync** | Sinkronisasi cerdas berbasis `lastmod` sitemap — hanya scrape/embed halaman baru/berubah, hapus halaman yang dihapus; **auto-sync saat startup** + manual via `/refresh` |
| **Concurrency Management** | Asyncio Semaphore (max 2 concurrent) untuk graceful degradation under load — request yang melebihi kapasitas di-queue, bukan ditolak |
| **User Interfaces** | REST API (FastAPI v1.3.0, port 8080), CLI interaktif, Telegram Bot (`/ask`, `/askscript`, `/start`, `/help`, `/status`, file upload `.sh/.slurm/.sbatch/.bash`, progress animation, HTML message splitting) |
| **Benchmarking** | Retrieval benchmark (Dense vs Sparse vs Multi-Vector vs Hybrid) dan TTFT/latency benchmark dengan configurable concurrency levels |
| **Logging & Observability** | Question logging persisten (UTC timestamp → `user_questions.logs`), Promtail log scraping untuk Grafana Loki |
| **Deployment** | Podman multi-container orchestration dengan 10 compose profiles untuk selective startup; per-service healthchecks; restart policies |
| **Testing** | Integration test suite (`test_services.py`) untuk verifikasi konektivitas dan fungsionalitas antar service |
| **Backward Compatibility** | Migrasi otomatis (`_migrate_add_lastmod`) untuk koleksi Qdrant lama tanpa metadata lastmod |

### Out of Scope

- Antarmuka web dashboard (hanya CLI, REST API, Telegram Bot)
- Multi-bahasa selain Bahasa Indonesia
- Fine-tuning model LLM atau embedding
- Integrasi dengan sistem ticketing eksternal
- Autentikasi dan otorisasi pengguna tingkat lanjut (RBAC)
- Dukungan multi-tenant

### User Roles

| Role | Interface | Kemampuan |
|---|---|---|
| **End User** | Telegram Bot | `/ask` untuk Q&A, `/askscript` untuk review script, upload file `.sh/.slurm/.sbatch/.bash`, `/start`, `/help`, `/status` |
| **Operator / Admin** | REST API / CLI | Semua kemampuan End User + `POST /refresh` untuk sync manual, `GET /refresh/status`, akses logs |
| **Developer** | REST API | Semua endpoint, akses benchmark, akses logs, pengembangan dan eksperimen |
| **System** (background) | Internal | Auto-sync saat startup, question logging, periodic healthchecks per service |

---

## 5. Functional Requirements

### FR-01: RAG Question Answering (`POST /ask`)

Pipeline menjawab pertanyaan pengguna secara end-to-end:

1. **Relevance Check** — LLM mengevaluasi apakah pertanyaan relevan dengan domain HPC ALELEON (YA/TIDAK)
2. **Multi-Mode Embedding** — Query di-embed menjadi dense (1024-dim) + sparse (lexical weights) via BAAI/bge-m3
3. **Hybrid Retrieval** — Dual-path search (dense cosine + sparse keyword) → RRF Fusion → over-fetch 2× limit
4. **ColBERT Reranking** — Candidates di-rerank via ColBERT endpoint → ambil top-10
5. **LLM Generation** — Generate jawaban dengan 11 aturan anti-halusinasi (temperature 0.3, max_tokens 8192)
6. **Source Justification** — LLM menjelaskan mengapa setiap sumber relevan
7. **Irrelevant Source Filtering** — Sumber dengan justifikasi "TIDAK RELEVAN" otomatis dibuang
8. **Question Logging** — Setiap pertanyaan dicatat dengan UTC timestamp

### FR-02: Script Review (`POST /review-script`)

Review skrip Bash/Slurm dengan pendekatan hybrid 3-tahap:

1. **Relevance Check** — Cek apakah skrip relevan dengan domain HPC
2. **Parameter Extraction** — LLM mengekstrak parameter `#SBATCH` sebagai JSON (temperature 0.0)
3. **Policy Retrieval** — Jika ada resource params, retrieval kebijakan HPC dari knowledge base
4. **Hybrid Review** — Gabungan analisis teknis LLM (syntax, best practices) + validasi kebijakan HPC (batas partisi, kuota)

### FR-03: Incremental Knowledge Sync (`POST /refresh`)

- **Startup Auto-Sync** — Saat container start, otomatis cek perubahan sitemap dan sync
- **Manual Trigger** — `POST /refresh` menjalankan sync di background thread
- **Status Check** — `GET /refresh/status` untuk memonitor progress sync
- **Smart Sync** — Skip halaman unchanged, hapus halaman removed, scrape+embed hanya yang new/modified
- **Thread Safety** — `threading.Lock` mencegah concurrent sync

### FR-04: Telegram Bot Interface

| Command / Action | Deskripsi |
|---|---|
| `/ask <pertanyaan>` | Tanya jawab RAG |
| `/askscript` | Review skrip (kirim skrip setelah command) |
| `/start` | Welcome message dan instruksi penggunaan |
| `/help` | Daftar command yang tersedia |
| `/status` | Cek status koneksi ke RAG API |
| Plain text message | Otomatis diteruskan sebagai pertanyaan RAG |
| File upload (`.sh`/`.slurm`/`.sbatch`/`.bash`) | Otomatis di-review sebagai script |
| Progress animation | Animasi placeholder selama menunggu respons |
| HTML message splitting | Pesan panjang dipecah agar sesuai limit Telegram (4000 char) |

### FR-05: Benchmarking Suite

| Benchmark | Mode | Detail |
|---|---|---|
| **Retrieval Benchmark** | `--mode ingest`, `--mode query`, `--mode cleanup` | Membandingkan Dense vs Sparse vs Multi-Vector vs Hybrid dengan 69 pertanyaan; mengukur retrieval quality |
| **TTFT/Latency Benchmark** | Configurable concurrency levels (1,2,4,8,16) | Mengukur Time to First Token dan latency end-to-end pada berbagai tingkat concurrency; 49 requests per level |

---

## 6. Non-Functional Requirements

### Performance

| Metrik | Target | Justifikasi |
|---|---|---|
| Time to First Token (TTFT) | ≤ 5 detik (concurrency 1) | Pengalaman interaktif yang responsif |
| End-to-end response time | ≤ 60 detik (concurrency 1) | Toleransi wajar untuk jawaban RAG komprehensif |
| Concurrent request handling | 2 simultaneous (graceful queue beyond) | Optimum berdasarkan benchmark: 0% failure rate, throughput 0.051 req/s |
| Embedding batch throughput | 32 texts/batch | Menyeimbangkan memory usage dan speed |

### Availability & Reliability

| Aspek | Detail |
|---|---|
| Restart Policy | `unless-stopped` untuk semua infra services (embedding, vLLM, Qdrant, rag-api, telegram-bot) |
| Healthchecks | Per-service healthcheck di compose.yml (interval 30s, retries 3-10, start_period 60-300s) |
| Graceful Degradation | Semaphore-based concurrency limiter — request yang melebihi kapasitas di-antri, bukan ditolak/connection reset |
| Startup Resilience | `wait_for_vllm()` dengan timeout 600s dan retry interval 10s; auto-sync saat startup |

### Security & Data Privacy

| Aspek | Detail |
|---|---|
| Zero Cloud Dependency | Seluruh pipeline berjalan lokal — tidak ada data yang dikirim ke cloud |
| API Key Management | Qdrant API key via environment variable (`QDRANT__SERVICE__API_KEY`) |
| Telegram Token | Token bot disimpan di `.env`, tidak di-commit ke repository |
| Network Isolation | Service-to-service communication melalui Podman internal network |

### Scalability

| Aspek | Detail |
|---|---|
| Horizontal | Saat ini single-node, single-GPU — tidak dirancang untuk horizontal scaling |
| Vertical | `--gpu-memory-utilization 0.99` memaksimalkan penggunaan VRAM; model GPTQ 4-bit (35B total, ~3B aktif per forward) |
| Knowledge Base | Incremental sync memungkinkan knowledge base tumbuh tanpa full rebuild |

---

## 7. Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Podman Multi-Container Orchestration                │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Infrastructure Layer                      │    │
│  │  ┌──────────────────┐ ┌────────────────┐ ┌──────────────┐  │    │
│  │  │ embedding-service│ │   vllm-rocm    │ │    qdrant    │  │    │
│  │  │  (BAAI/bge-m3)   │ │ (Qwen3.5-35B)  │ │ (Vector DB)  │  │    │
│  │  │  Port 8001       │ │  Port 8000     │ │ Port 6333    │  │    │
│  │  │  /embed          │ │  OpenAI API    │ │ REST + gRPC  │  │    │
│  │  │  /embed/multi    │ │                │ │              │  │    │
│  │  │  /rerank         │ │                │ │              │  │    │
│  │  └──────────────────┘ └────────────────┘ └──────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Application Layer                         │    │
│  │  ┌──────────────┐ ┌────────────────┐ ┌───────────────────┐ │    │
│  │  │   rag-app    │ │    rag-api     │ │   telegram-bot    │ │    │
│  │  │  (CLI Mode)  │ │ (REST API)     │ │  (User Interface) │ │    │
│  │  │              │ │  Port 8080     │ │                   │ │    │
│  │  └──────────────┘ └────────────────┘ └───────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Tooling Layer                              │    │
│  │  ┌──────────────┐ ┌────────────────┐ ┌───────────────────┐ │    │
│  │  │  benchmark   │ │ benchmark-ttft │ │    promtail       │ │    │
│  │  │ (Retrieval)  │ │  (Latency)     │ │  (Log Scraping)   │ │    │
│  │  └──────────────┘ └────────────────┘ └───────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Question Answering

```
User Question
     │
     ▼
[1] Relevance Filter (LLM, temp=0.0, max_tokens=10)
     │ YA                          │ TIDAK
     ▼                             ▼
[2] Multi-Mode Embed          "Pertanyaan tidak relevan"
    (dense + sparse)
     │
     ▼
[3] Hybrid Search (Qdrant)
    ├─ Dense cosine similarity (limit: 2× TOP_K × 2)
    └─ Sparse keyword match   (limit: 2× TOP_K × 2)
     │
     ▼
[4] RRF Fusion (limit: 2× TOP_K = 20 candidates)
     │
     ▼
[5] ColBERT Reranking → Top-10 chunks
     │
     ▼
[6] LLM Generation (temp=0.3, max_tokens=8192)
    + 11 aturan anti-halusinasi
     │
     ▼
[7] Source Justification (temp=0.1, max_tokens=1024)
     │
     ▼
[8] Filter "TIDAK RELEVAN" sources
     │
     ▼
Response: { answer, sources[] }
```

### Key Technical Decisions

| Keputusan | Pilihan | Alasan |
|---|---|---|
| Embedding Model | BAAI/bge-m3 | Multi-modal (dense + sparse + ColBERT) dalam satu model; mendukung hybrid retrieval tanpa model terpisah |
| LLM Model | Qwen3.5-35B-A3B-GPTQ-Int4 | Mixture-of-Experts (~3B aktif per forward); GPTQ 4-bit fit di single GPU 32GB; 256K context window |
| Vector DB | Qdrant | Native support untuk hybrid dense-sparse collections; persistent storage; REST + gRPC API |
| Retrieval Strategy | Hybrid (Dense + Sparse + RRF + ColBERT) | Over-fetch 2× lalu rerank — menangkap baik semantic similarity maupun exact keyword match |
| Container Runtime | Podman | Rootless, daemonless; cocok untuk HPC environment |
| Concurrency Limit | Semaphore(2) | Sweet spot berdasarkan benchmark: 0% failure, throughput optimal |

---

## 8. API Specification

### Endpoints

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/ask` | RAG Question Answering | None |
| `POST` | `/review-script` | Script Review (Hybrid) | None |
| `POST` | `/refresh` | Trigger Incremental Sync (background) | None |
| `GET` | `/refresh/status` | Check Sync Status | None |
| `GET` | `/health` | Health Check | None |
| `GET` | `/` | Service Info & Endpoint List | None |

### Request/Response Schemas

#### `POST /ask`

```json
// Request
{ "question": "Bagaimana cara membuat conda environment?" }

// Response (200)
{
  "answer": "Untuk membuat conda environment di ALELEON...",
  "sources": [
    {
      "title": "Conda Environment Setup",
      "source_url": "https://wiki.efisonlt.com/...",
      "section": "Langkah Pembuatan",
      "justification": "Dokumen ini berisi panduan langkah-demi-langkah..."
    }
  ]
}

// Response (503) — Service belum siap
{ "detail": "RAG chain belum siap, coba lagi nanti." }
```

#### `POST /review-script`

```json
// Request
{ "script": "#!/bin/bash\n#SBATCH --mem=64G\nsrun gmx_mpi mdrun" }

// Response (200)
{
  "review": "## Analisis Script\n...",
  "issues_found": 2,
  "policy_sources": [
    {
      "title": "Kebijakan Partisi",
      "source_url": "https://wiki.efisonlt.com/...",
      "section": "Batas Memory",
      "justification": "..."
    }
  ]
}
```

#### `POST /refresh`

```json
// Response (200) — Sync started
{ "status": "started", "message": "Sync dimulai di background..." }

// Response (200) — Already running
{ "status": "already_running", "message": "Sync sedang berjalan..." }
```

#### `GET /refresh/status`

```json
{
  "running": false,
  "last_result": { "added": 2, "updated": 1, "deleted": 0, "unchanged": 66 },
  "last_sync_time": "2026-07-09T12:00:00+00:00"
}
```

### LLM Inference Parameters

| Endpoint / Function | `max_tokens` | `temperature` | `top_p` | Lainnya |
|---|---|---|---|---|
| `/ask` — `generate_response` | 8192 | 0.3 | 0.9 | `top_k=20`, `presence_penalty=1.5`, `enable_thinking=False` |
| `/review-script` — `review_script_hybrid` | 4096 | 0.2 | 0.9 | `enable_thinking=False` |
| `is_question_relevant` | 10 | 0.0 | — | Relevance filter (YA/TIDAK) |
| `generate_source_justifications` | 1024 | 0.1 | — | Per-source justification |
| `extract_resource_params` | 512 | 0.0 | — | Parse #SBATCH params → JSON |

### vLLM Server Configuration

| Parameter | Value | Reason |
|---|---|---|
| `--dtype` | `float16` | Inference precision untuk quantized model |
| `--enforce-eager` | True | Menghindari CUDAGraph issues pada ROCm/RDNA4 |
| `--gpu-memory-utilization` | 0.99 | Gunakan 99% VRAM yang tersedia |
| `--max-model-len` | 262144 | Full 256K context window |
| `--max-num-seqs` | 16 | Max concurrent sequences |
| `--tensor-parallel-size` | 1 | Single GPU inference |
| `--enable-auto-tool-choice` | True | Tool/function calling support |
| `--tool-call-parser` | `qwen3_coder` | Tool call parser untuk Qwen3 Coder |
| `--reasoning-parser` | `qwen3` | Reasoning parser untuk Qwen3 |
| `--enable-prefix-caching` | True | Cache common prefixes untuk faster inference |

---

## 9. Success Metrics & KPIs

### Retrieval Quality

| Metrik | Target | Cara Ukur |
|---|---|---|
| Retrieval Accuracy (Top-10) | ≥ 85% relevant chunks in top-10 | Benchmark suite 69 pertanyaan (5 level kesulitan) |
| Hybrid vs Dense Improvement | Hybrid ≥ Dense dalam retrieval quality | Retrieval benchmark: Dense vs Sparse vs Multi-Vector vs Hybrid |
| ColBERT Rerank Lift | Reranked results ≥ RRF-only results | A/B comparison dalam benchmark |

### Answer Quality

| Metrik | Target | Cara Ukur |
|---|---|---|
| Anti-Hallucination Rate (Level 4) | 100% jawab "tidak ada informasi" untuk 15 pertanyaan out-of-context | Level 4 benchmark (15 pertanyaan yang jawabannya TIDAK ada di dokumen) |
| Source Attribution Rate | 100% jawaban memiliki minimal 1 sumber yang valid | Cek output `sources[]` pada setiap response |
| Factual Consistency | Angka, versi, dan perintah sesuai dokumen sumber | Manual review sample answers vs wiki |

### System Performance

| Metrik | Target | Cara Ukur |
|---|---|---|
| TTFT (concurrency=1) | ≤ 5 detik | TTFT benchmark |
| End-to-end Latency (concurrency=1) | ≤ 60 detik | TTFT benchmark |
| Failure Rate (concurrency=2) | 0% | TTFT benchmark dengan concurrency levels |
| Throughput (concurrency=2) | ≥ 0.05 req/s | TTFT benchmark |

### Operational

| Metrik | Target | Cara Ukur |
|---|---|---|
| Incremental Sync Speed | ≤ 5 menit untuk full sync | Timer pada `/refresh` |
| Knowledge Base Coverage | 100% halaman wiki yang aktif | Bandingkan sitemap entries vs Qdrant collection points |
| Uptime | ≥ 99% selama jam kerja | Healthcheck monitoring via Promtail/Grafana Loki |

### Benchmark Suite

69 test questions tersebar dalam 5 level kesulitan:

| Level | Count | Description | Tujuan Evaluasi |
|---|---|---|---|
| **Level 1** — Direct Facts | 20 | RAM, walltime, GPU, commands, quotas | Retrieval presisi + factual accuracy |
| **Level 2** — Multi-Chunk | 10 | Perbandingan partisi, prosedur multi-langkah | Cross-document retrieval |
| **Level 3** — Reasoning | 10 | Kalkulasi memory, troubleshooting, deduksi | Reasoning + retrieval gabungan |
| **Level 4** — Anti-Hallucination | 15 | Pertanyaan yang jawabannya TIDAK ada di wiki | Jujur menjawab "tidak tahu" |
| **Level 5** — Additional | 14 | Pertanyaan umum tambahan | Coverage dan konsistensi |

---

## 10. Key Value Propositions

### Core (Diferensiator Utama)

| Value | Deskripsi | Dampak |
|---|---|---|
| **Akurasi Berbasis Dokumen** | 11 aturan anti-halusinasi yang ketat; LLM dipaksa menjawab HANYA dari dokumen referensi dengan kutipan persis, termasuk angka/versi presisi | Mengurangi misinformasi; informasi terverifikasi dari wiki resmi |
| **Hybrid Retrieval 4-Tahap** | Dense → Sparse → RRF Fusion → ColBERT Reranking dengan over-fetch 2× memberikan presisi retrieval tertinggi | Menemukan dokumen relevan bahkan dengan sinonim/terminologi berbeda |
| **Anti-Halusinasi Berlapis** | Pre-filter relevansi + 11 aturan sistem prompt + source justification + filtering sumber "TIDAK RELEVAN" | Kepercayaan tinggi terhadap output; sumber selalu dapat ditelusuri |

### Differentiator (Keunggulan Kompetitif)

| Value | Deskripsi | Dampak |
|---|---|---|
| **Zero Cloud Dependency** | Seluruh pipeline berjalan lokal (AMD ROCm GPU, Podman containers) | Privasi data terjaga, tidak ada biaya API cloud, latency rendah |
| **Script Review Terintegrasi** | Review Slurm 3-tahap: parse parameter → retrieval kebijakan → validasi otomatis | Mencegah job failure akibat parameter melampaui batas partisi/kuota |
| **Evaluasi Kuantitatif** | Benchmark 69 pertanyaan (5 level) + retrieval benchmark + TTFT benchmark | Pengukuran objektif kualitas; reproducible evaluation |

### Enabler (Pendukung Operasional)

| Value | Deskripsi | Dampak |
|---|---|---|
| **Knowledge Base Selalu Mutakhir** | Incremental sync + startup auto-sync + endpoint `/refresh` manual | Tidak perlu rebuild; hanya halaman baru/berubah diproses |
| **Multi-Interface Akses** | CLI (debug), REST API (integrasi), Telegram Bot (mobile/desktop) | Fleksibilitas akses sesuai kebutuhan |
| **Biaya Operasional Minimal** | GPTQ 4-bit MoE (~3B aktif) pada single AMD GPU 32GB | Tidak perlu cluster GPU mahal |

---

## 11. Dependencies & Constraints

### Hardware Requirements

| Komponen | Minimum | Tested On |
|---|---|---|
| **GPU** | AMD GPU dengan ROCm support, ≥ 24GB VRAM | Radeon AI PRO R9700 (gfx1201, 32GB VRAM) |
| **RAM** | 16GB system RAM | 48GB DDR5 |
| **CPU** | Any x86_64 | Intel i7-12700K / Ryzen 7 9800X3D |
| **ROCm** | 6.0+ | 7.0 (HIP 7.0.51831) |
| **Disk** | ~40GB total | ~15GB container image + ~25GB model weights |

### Software Dependencies

| Dependency | Version/Detail | Purpose |
|---|---|---|
| Podman (+ podman-compose) | Latest | Container orchestration |
| ROCm drivers | 6.0+ (tested: 7.0) | GPU compute |
| vLLM | Nightly (rocm/vllm-dev:nightly) | LLM inference engine |
| BAAI/bge-m3 | Latest | Embedding model (dense + sparse + ColBERT) |
| Qwen3.5-35B-A3B-GPTQ-Int4 | Latest | LLM model (auto-downloaded) |
| Qdrant | Latest | Vector database |
| FastAPI + Uvicorn | Latest | REST API framework |
| LangChain (core, community, text-splitters, qdrant) | Latest | Document processing & retriever utilities |
| python-telegram-bot | Latest | Telegram Bot integration |
| BeautifulSoup4 + lxml | Latest | HTML parsing |

### External Dependencies

| Dependency | Detail | Risiko jika Unavailable |
|---|---|---|
| `wiki.efisonlt.com` | Sumber knowledge base (sitemap + halaman wiki) | Sync gagal; knowledge base tetap menggunakan data terakhir di Qdrant |
| Hugging Face Hub | Download model weights (satu kali, lalu cached) | Initial setup gagal; setelah cached, tidak diperlukan |
| `docker.io` | Pull container images (Qdrant, vLLM, Promtail) | Build/deploy gagal; gunakan cached images |

### Constraints

- **Single GPU** — Sistem dirancang untuk single-GPU inference (`tensor-parallel-size=1`); tidak mendukung multi-GPU
- **Bahasa Indonesia only** — Seluruh prompt, jawaban, dan antarmuka dalam Bahasa Indonesia
- **AMD ROCm only** — GPU compute terikat pada ekosistem AMD ROCm; tidak kompatibel dengan NVIDIA CUDA tanpa modifikasi
- **No Authentication** — API tidak memiliki authentication layer; diasumsikan berjalan di internal network
- **No Persistent Conversation** — Setiap pertanyaan independen; tidak ada conversation memory/history

---

## 12. Risks & Mitigations

| ID | Risiko | Severity | Probabilitas | Mitigasi |
|---|---|---|---|---|
| R-01 | **LLM Hallucination** — Model menjawab informasi yang tidak ada di dokumen | Tinggi | Sedang | 11 aturan anti-halusinasi; relevance pre-filter; source justification + filtering "TIDAK RELEVAN"; Level 4 benchmark (15 anti-hallucination test questions) |
| R-02 | **Wiki Downtime** — `wiki.efisonlt.com` tidak bisa diakses saat sync | Sedang | Rendah | Qdrant persistent storage menyimpan data terakhir; sync gagal tidak menghapus data existing; retry mechanism |
| R-03 | **GPU Memory OOM** — Model terlalu besar untuk VRAM | Tinggi | Rendah | GPTQ 4-bit quantization; `--gpu-memory-utilization 0.99`; MoE architecture (~3B aktif per forward dari 35B total) |
| R-04 | **ROCm Compatibility** — vLLM nightly build tidak kompatibel dengan GPU/ROCm versi tertentu | Sedang | Sedang | `--enforce-eager` menghindari CUDAGraph issues; `HSA_OVERRIDE_GFX_VERSION` untuk GPU compatibility; pin tested vLLM image |
| R-05 | **Stale Knowledge Base** — Wiki berubah tapi sync tidak berjalan | Sedang | Rendah | Auto-sync saat startup; manual `/refresh` endpoint; `lastmod`-based incremental sync |
| R-06 | **Concurrent Overload** — Terlalu banyak request bersamaan menyebabkan GPU contention | Sedang | Sedang | Semaphore(2) concurrency limiter; queue-based graceful degradation; healthcheck monitoring |
| R-07 | **Embedding Service Failure** — embedding-service down saat ingestion/query | Tinggi | Rendah | Healthcheck dengan start_period 120s; restart policy `unless-stopped`; API timeout 600s |
| R-08 | **Large Document Chunks** — Beberapa halaman wiki sangat panjang | Rendah | Sedang | HTMLSectionSplitter + fallback RecursiveCharacterTextSplitter (chunk_size=4500, overlap=900) |

---

## 13. Milestones & Deliverables

| Fase | Milestone | Deliverables | Status |
|---|---|---|---|
| **Fase 1 — Foundation** | Core RAG Pipeline | Ingestion pipeline, embedding service, Qdrant vector store, basic LLM Q&A, CLI interface | ✅ Selesai |
| **Fase 2 — Hybrid Retrieval** | Advanced Retrieval | Dense + Sparse hybrid collection, RRF Fusion, ColBERT reranking, retrieval benchmark | ✅ Selesai |
| **Fase 3 — Anti-Hallucination** | Accuracy & Trust | 11 aturan anti-halusinasi, relevance pre-filter, source justification, irrelevant source filtering | ✅ Selesai |
| **Fase 4 — Microservices** | Production Architecture | FastAPI REST API, Podman compose orchestration, healthchecks, concurrency management, auto-sync | ✅ Selesai |
| **Fase 5 — User Interfaces** | Multi-Channel Access | Telegram Bot (full features), CLI interactive mode, API documentation | ✅ Selesai |
| **Fase 6 — Script Review** | Extended Capabilities | 3-tahap hybrid script review, parameter extraction, policy retrieval & validation | ✅ Selesai |
| **Fase 7 — Benchmarking** | Evaluation & Metrics | Retrieval benchmark (4 strategies), TTFT/latency benchmark, 69 test questions (5 levels), chart generators | ✅ Selesai |
| **Fase 8 — Observability** | Monitoring & Logging | Question logging, Promtail integration, Grafana Loki pipeline | ✅ Selesai |
| **Fase 9 — Thesis Writeup** | Documentation | PRD, HOW_IT_WORKS.md, MICROSERVICES_ARCHITECTURE.md, thesis document | 🔄 In Progress |

---

## 14. Acceptance Criteria

### AC-01: Question Answering

- [x] Sistem menjawab pertanyaan dalam Bahasa Indonesia
- [x] Setiap jawaban menyertakan minimal 1 sumber dokumen yang valid
- [x] Pertanyaan off-topic ditolak dengan pesan yang jelas
- [x] Jawaban Level 4 (anti-hallucination) menjawab "tidak menemukan informasi" bukan mengarang
- [x] Response time ≤ 60 detik pada concurrency 1

### AC-02: Script Review

- [x] Sistem mampu meng-extract parameter `#SBATCH` dari script
- [x] Review mencakup analisis teknis DAN validasi kebijakan HPC
- [x] Script yang tidak relevan ditolak
- [x] Policy sources ditampilkan dengan justifikasi

### AC-03: Knowledge Sync

- [x] Sync hanya memproses halaman baru/berubah (bukan full rebuild)
- [x] Halaman yang dihapus dari sitemap juga dihapus dari Qdrant
- [x] Auto-sync berjalan saat startup container
- [x] Manual sync via API tidak memblokir request lain

### AC-04: Telegram Bot

- [x] Bot merespons `/ask` dan `/askscript` commands
- [x] Bot menerima file upload `.sh/.slurm` untuk review
- [x] Progress animation ditampilkan selama menunggu
- [x] Pesan panjang dipecah agar tidak error Telegram API limit

### AC-05: Deployment

- [x] Seluruh sistem bisa di-deploy dengan satu perintah compose
- [x] Semua service memiliki healthcheck
- [x] Service yang crash otomatis restart
- [x] Selective startup via compose profiles berfungsi

---

## 15. Stakeholders

| Role | Nama / Tim | Tanggung Jawab |
|---|---|---|
| **Author / Developer** | Krisostomus Nova RAHMANTO | Desain, implementasi, testing, dokumentasi |
| **Thesis Advisor (EURECOM)** | [Thesis Advisor] | Review akademis, approval thesis |
| **Industry Supervisor (EFISON)** | [EFISON Supervisor] | Validasi kebutuhan bisnis, domain expertise HPC |
| **End Users** | Tim L1 HPC Support EFISON | User acceptance testing, feedback |

---

## 16. Revision History

| Versi | Tanggal | Perubahan |
|---|---|---|
| 1.0.0 | — | PRD awal: Executive Summary, Problem Statement, System Scope, Key Value Propositions |
| 1.3.0 | 2026-07-09 | Restrukturisasi lengkap: tambah User Stories, Functional Requirements, Non-Functional Requirements, Technical Architecture, API Specification, Success Metrics, Dependencies & Constraints, Risks & Mitigations, Milestones, Acceptance Criteria, Stakeholders, Revision History; update konten dari codebase aktual (concurrency limiter, auto-sync, Telegram Bot features, source filtering, testing); simplifikasi Executive Summary; prioritisasi Value Propositions |
