# 📊 Interpretasi Benchmark RAG API (20260420-071631) — Prefix Cache

## Informasi Benchmark

| Parameter | Nilai |
|---|---|
| **Target URL** | `http://rag-api:8080/ask` |
| **GPU** | AMD Radeon AI Pro R9700 |
| **Jumlah Pertanyaan Unik** | 48 |
| **Request per Level** | 49 |
| **Concurrency Levels** | 1, 2, 4, 8, 16 |
| **Fitur Aktif** | Prefix Cache |
| **Timestamp** | 20 April 2026, 07:16:31 |

---

## Ringkasan Data

| Concurrency | Sukses | Gagal | Avg Latency (s) | P50 (s) | P99 (s) | Stdev (s) | Throughput (req/s) | Wall Clock (s) |
|:-----------:|:------:|:-----:|:----------------:|:-------:|:-------:|:---------:|:------------------:|:--------------:|
| **1** | 49/49 | 0 | 20.38 | 18.60 | 47.84 | 7.78 | **0.0491** | 998.4 |
| **2** | 49/49 | 0 | 39.20 | 37.45 | 57.71 | 7.88 | **0.0505** | 969.6 |
| **4** | 37/49 | 12 | 72.71 | 70.06 | 132.50 | 27.33 | **0.0478** | 773.7 |
| **8** | 35/49 | 14 | 129.64 | 138.34 | 274.53 | 69.53 | **0.0521** | 672.2 |
| **16** | 35/49 | 14 | 230.56 | 311.72 | 327.06 | 108.47 | **0.0526** | 664.9 |

---

## Visualisasi

````carousel
![RAG E2E Benchmark - AMD Radeon AI Pro R9700](output/benchmark_ttft/results-rag-20260420-071631 prefix cache/rag_benchmark_chart_20260420-071631_AMD_Radeon_AI_Pro_R9700.png)
<!-- slide -->
![RAG E2E Benchmark - AMD GPU (simplified)](/output/benchmark_ttft/results-rag-20260420-071631 prefix cache/rag_benchmark_chart_20260420-071631_AMD_GPU.png)
````

---

## Interpretasi Detail

### 1. 🟢 Concurrency 1 & 2 — Zona Stabil & Optimal

Pada concurrency **1** dan **2**, sistem berjalan **100% stabil** — seluruh 49 request berhasil tanpa satupun kegagalan.

| Metrik | Conc. 1 | Conc. 2 | Perubahan |
|---|---|---|---|
| Avg Latency | 20.38s | 39.20s | +92% (≈ 2×) |
| P50 Latency | 18.60s | 37.45s | +101% (≈ 2×) |
| Throughput | 0.0491 | 0.0505 | **+2.9%** ✅ |
| Wall Clock | 998.4s | 969.6s | **-2.9%** ✅ |

**Analisis:**
- Latency naik **hampir 2×** dari concurrency 1 ke 2 — ini menunjukkan bahwa LLM inference pada GPU **diproses secara serial** (antrian), bukan benar-benar paralel. Request kedua harus menunggu giliran, sehingga waktu rata-rata per request menjadi dua kali lipat.
- Namun **throughput sedikit naik** (0.049 → 0.051 req/s) karena overhead antrian diminimalkan dan total wall-clock turun ~30 detik. Ini berarti ada **sedikit keuntungan dari overlapping** (misalnya: saat satu request menunggu retrieval, request lain bisa mulai diproses).
- **Standard deviasi tetap rendah** (~7.8s), menunjukkan **konsistensi respons** yang baik.

> [!TIP]
> **Concurrency 2 adalah sweet spot operasional**: throughput sedikit lebih baik dari concurrency 1, tanpa satu pun kegagalan, dan wall-clock total lebih pendek.

---

### 2. 🟡 Concurrency 4 — Awal Degradasi

| Metrik | Nilai | Catatan |
|---|---|---|
| Sukses/Total | **37/49** | ⚠️ 12 request gagal (24.5% failure rate) |
| Avg Latency | 72.71s | 3.6× lebih lambat dari conc. 1 |
| P99 Latency | 132.50s | >2 menit untuk worst case |
| Stdev | 27.33s | 3.5× stdev dibanding conc. 1 |
| Error Type | `[Errno 104] Connection reset by peer` | Server menolak koneksi |

**Analisis:**
- Mulai terjadi **connection reset by peer** — server (atau reverse proxy di depannya) **secara aktif memutus koneksi** saat terlalu banyak request antri. Ini bukan timeout, melainkan server yang kewalahan dan menolak koneksi baru.
- Latency melonjak signifikan karena 4 request bersaing untuk satu resource inference GPU.
- Variabilitas (stdev) meningkat tajam — pengalaman pengguna menjadi **tidak terprediksi**.

---

### 3. 🔴 Concurrency 8 & 16 — Zona Saturasi Penuh

| Metrik | Conc. 8 | Conc. 16 |
|---|---|---|
| Sukses/Total | 35/49 (28.6% fail) | 35/49 (28.6% fail) |
| Avg Latency | **129.6s** (~2 min) | **230.6s** (~4 min) |
| P50 Latency | 138.3s | **311.7s** (~5 min) |
| P99 Latency | 274.5s (~4.5 min) | **327.1s** (~5.5 min) |
| Stdev | 69.5s | 108.5s |

**Analisis:**
- **Jumlah kegagalan stabil di 14** pada conc. 8 dan 16 — ini mengindikasikan ada **hard limit** (kemungkinan connection pool atau max worker pada server) yang membatasi ~14 koneksi simultan untuk ditolak.
- Pada concurrency 16, **P50 = 311.7 detik** — artinya **lebih dari separuh request menunggu lebih dari 5 menit**. Ini tidak dapat diterima untuk pengalaman pengguna interaktif.
- Throughput tetap naik tipis (0.052 → 0.053 req/s) karena lebih banyak request dijejalkan dalam waktu yang sama, tapi ini datang dengan **biaya latency dan reliability** yang sangat tinggi.

> [!CAUTION]
> Throughput yang sedikit naik pada concurrency tinggi **bukan** tanda positif — ini hanya efek matematis dari lebih banyak request selesai dibanding wall-time yang lebih pendek, sementara kualitas layanan (latency & error rate) anjlok drastis.

---

### 4. 📈 Tren Throughput — Plateau yang Menyesatkan

```
Concurrency:  1     2     4     8     16
Throughput:  0.049  0.051  0.048  0.052  0.053 req/s
```

Throughput **hampir datar** di sekitar **~0.05 req/s** (sekitar 1 request setiap 20 detik). Ini mengungkap bahwa:

1. **Bottleneck utama adalah LLM inference pada GPU** — tidak peduli berapa banyak request dikirim paralel, GPU hanya mampu memproses ~0.05 req/s.
2. **Prefix cache tidak memberikan peningkatan throughput** yang dramatis pada skenario concurrency tinggi, karena bottleneck bukan pada tahap KV cache computation, melainkan pada **total decode time**.
3. Sistem berperilaku seperti **single-server queue** — kapasitas riilnya ~3 request/menit.

---

### 5. 🔧 Error Pattern: Connection Reset by Peer

Semua kegagalan pada concurrency ≥ 4 disebabkan oleh error yang sama:

```
[Errno 104] Connection reset by peer
```

Ini mengindikasikan:
- Server (atau layer di depannya: Nginx, Gunicorn, dsb.) memiliki **batas koneksi/worker aktif**
- Ketika kapasitas antrian habis, koneksi baru **langsung ditolak** alih-alih di-queue
- Ini terjadi secara **konsisten** (~12-14 request gagal) terlepas dari apakah concurrency 4, 8, atau 16

---

## Kesimpulan & Rekomendasi

### Temuan Utama

| # | Temuan |
|---|---|
| 1 | **Kapasitas real GPU** adalah **~0.05 req/s** (~3 request/menit), terlepas dari concurrency |
| 2 | **Concurrency 2 adalah titik operasional terbaik** — throughput optimal tanpa kegagalan |
| 3 | **Concurrency ≥ 4 menyebabkan kegagalan** karena server menolak koneksi berlebih |
| 4 | **Latency meningkat linear** dengan concurrency — sistem memproses request secara serial |
| 5 | Prefix cache memberikan **stabilitas** (stdev rendah pada conc. rendah) tapi **tidak menghilangkan bottleneck inference** |

### Rekomendasi

> [!IMPORTANT]
> **Untuk production deployment:**
> 1. **Batasi concurrency maksimum ke 2** pada level load balancer/API gateway
> 2. **Implementasi request queuing** di application layer agar request yang melebihi kapasitas di-antri, bukan ditolak
> 3. **Pertimbangkan horizontal scaling** (tambah GPU/instance) jika kebutuhan throughput > 3 req/menit
> 4. **Set timeout client yang realistis** — minimal 60 detik untuk concurrency 1, minimal 120 detik untuk concurrency 2

---

## Kesimpulan: Error `[Errno 104]` **bukan disebabkan oleh bug di `run-rag-bench.py`**

Kode benchmark-nya sendiri **sudah benar** dari sisi connection handling. Berikut penjelasan detail:

---

### ✅ Yang Sudah Benar di `run-rag-bench.py`

1. **Semaphore sudah benar** ([baris 168](file:///home/efison-kristo/rag-for-l1-aleleon-hpc-support/services/benchmark_ttft/run-rag-bench.py#L168)) — membatasi concurrency sesuai level yang diminta.

2. **Satu `ClientSession` untuk semua request** ([baris 176](file:///home/efison-kristo/rag-for-l1-aleleon-hpc-support/services/benchmark_ttft/run-rag-bench.py#L176)) — ini *best practice* aiohttp karena reuse connection pool.

3. **Timeout 10 menit** ([baris 170](file:///home/efison-kristo/rag-for-l1-aleleon-hpc-support/services/benchmark_ttft/run-rag-bench.py#L170)) — cukup longgar.

4. **Error handling proper** ([baris 144-155](file:///home/efison-kristo/rag-for-l1-aleleon-hpc-support/services/benchmark_ttft/run-rag-bench.py#L144-L155)) — exception ditangkap dan dilaporkan.

---

### 🔍 Akar Masalah Sebenarnya: **Server-side bottleneck**

Error `[Errno 104] Connection reset by peer` berasal dari **sisi server**, bukan client. Buktinya terletak di pola data:

#### Bukti 1: Latency gagal pada concurrency 16 **identik**

```
Concurrency 16 — semua 14 request gagal pada ~19.982s (identik hingga milidetik):
  ID=18  latency=19.982s
  ID=19  latency=19.983s
  ID=20  latency=19.983s
  ... (semua 14 request gagal di waktu yang persis sama)
```

Ini menunjukkan **server yang sedang memproses request batch** lalu **crash/reset semua koneksi yang antri** secara bersamaan.

#### Bukti 2: Concurrency 8 — cluster gagal pada `17.343s`

```
  ID=10-16 → semua gagal di 17.343s (6 request, persis sama)
```

Ini bukan perilaku timeout — ini adalah **server yang secara aktif menutup koneksi** saat resource habis.

#### Bukti 3: Arsitektur RAG chain **blocking + synchronous**

Alur satu request `/ask` di [rag_api.py baris 156](file:///home/efison-kristo/rag-for-l1-aleleon-hpc-support/services/rag-app/rag_api.py#L156) memanggil `rag_chain(req.question)` yang sebenarnya adalah fungsi **synchronous** `retrieve_and_answer()` di [rag_app.py baris 1307](file:///home/efison-kristo/rag-for-l1-aleleon-hpc-support/services/rag-app/rag_app.py#L1307). Di dalamnya:

```
1 request /ask = 3 panggilan LLM synchronous berurutan:
  ① is_question_relevant()     → requests.post ke vLLM  (blocking)
  ② generate_response()        → OpenAI client ke vLLM  (blocking)
  ③ generate_source_justifications() → OpenAI client ke vLLM (blocking)
```

Semua panggilan ke vLLM dan embedding service menggunakan **`requests.post` (synchronous)** — bukan `aiohttp`. Ini **memblokir event loop uvicorn**.

#### Bukti 4: vLLM `--max-num-seqs 16`

Di [compose.yml baris 115](file:///home/efison-kristo/rag-for-l1-aleleon-hpc-support/compose.yml#L115):
```
--max-num-seqs 16
```

Tapi satu request RAG membuat **3 panggilan LLM berurutan**. Jadi 4 concurrent RAG requests = 4× aktif di vLLM (belum lagi embedding dan reranking request). Saat vLLM saturated → koneksi baru di-reset.

---

### 📋 Ringkasan Penyebab

| Layer | Masalah |
|---|---|
| **`rag_api.py`** | Endpoint `async def ask()` memanggil fungsi **synchronous blocking** (`rag_chain()`) → memblokir event loop uvicorn, mengurangi kemampuan menerima koneksi baru |
| **`rag_app.py`** | Semua HTTP calls ke vLLM & embedding menggunakan `requests.post` (sync), bukan async client |
| **vLLM** | `--max-num-seqs 16` limitasi, dan setiap RAG request = 3 LLM calls + 1 embedding + 1 rerank = **5 downstream calls** |
| **uvicorn** | Dijalankan **single worker** (default, [compose.yml L40](file:///home/efison-kristo/rag-for-l1-aleleon-hpc-support/services/rag-app/Dockerfile.rag-app#L40)) — 1 event loop, mudah terblokir oleh sync calls |

**Jadi masalahnya bukan di `run-rag-bench.py`, melainkan di arsitektur server RAG API yang menjalankan blocking synchronous calls di dalam async endpoint.**
