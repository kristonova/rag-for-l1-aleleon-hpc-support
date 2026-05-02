#!/usr/bin/env python3
"""
benchmark_retrieval.py — Benchmark Dense vs Sparse vs Multi-Vector vs Hybrid
=============================================================================
Membandingkan 4 metode retrieval di Qdrant menggunakan model BAAI/bge-m3:
  1. Dense   — cosine similarity pada vektor 1024-dim
  2. Sparse  — lexical weights (BM25-like dari bge-m3)
  3. Multi   — ColBERT late interaction (MaxSim)
  4. Hybrid  — Dense + Sparse digabung via RRF (Reciprocal Rank Fusion)

Metrik yang diukur:
  • Retrieval time (ms)
  • End-to-end time (retrieval + LLM generation) (ms)
  • Chunk overlap antar metode (Jaccard similarity)
  • Ingestion time (s)

Usage:
  python benchmark_retrieval.py --mode ingest     # Ingest dulu
  python benchmark_retrieval.py --mode query      # Benchmark query
  python benchmark_retrieval.py --mode all        # Ingest + query
  python benchmark_retrieval.py --mode cleanup    # Hapus collection benchmark
=============================================================================
"""

import os
import sys
import time
import json
import csv
import argparse
import requests
from typing import List, Dict, Any, Tuple
from xml.etree import ElementTree
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    PointStruct,
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
    MultiVectorConfig,
    MultiVectorComparator,
)
from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "your-secret-key")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8001")
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8000/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")

DENSE_DIM = 1024  # bge-m3 dense dimension

COLLECTION_DENSE = "bench_dense"
COLLECTION_SPARSE = "bench_sparse"
COLLECTION_MULTI = "bench_multivec"
COLLECTION_HYBRID = "bench_hybrid"

TOP_K = 10

SITEMAP_URL = "https://wiki.efisonlt.com/sitemap/sitemap-wiki.efisonlt.com-0.xml"

# Output directory (fixed path inside container, bind-mounted to host)
OUTPUT_DIR = os.getenv("BENCHMARK_OUTPUT_DIR", "/app/output/benchmark")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Questions (dari rag_app.py)
# ═══════════════════════════════════════════════════════════════════════

QUESTIONS = [
    # LEVEL 1
    "Berapa kapasitas RAM efektif per node di partisi epyc-jumbo?",
    "Berapa batas maksimal walltime (waktu komputasi) per job untuk golongan akun perseorangan?",
    "Saya ingin pakai partisi GPU. GPU jenis apa yang terpasang di partisi ampere?",
    "Apa alamat website portal EFIRO Web Service (EWS) untuk login?",
    "Perintah apa yang harus saya ketik di terminal untuk melihat daftar environment python/pyload yang saya buat?",
    "Berapa harga 1 GPU Hour (GH) untuk pengguna golongan perseorangan non-akademia?",
    "Saya mau cek sisa kuota core hour saya. Perintah sausage apa yang harus diketik?",
    "Apa itu PKSPIAS dalam pendaftaran akun ALELEON?",
    "Jika saya pakai aplikasi SFTP seperti FileZilla, apakah ada limit ukuran file yang bisa diupload?",
    "OS (Sistem Operasi) apa yang digunakan oleh ALELEON Mk.V?",
    "Versi SLURM berapa yang terpasang di sistem ALELEON saat ini?",
    "Bagaimana cara membatalkan/menghentikan job yang berstatus PENDING di terminal?",
    "Berapa kapasitas limit storage HOME untuk akun perseorangan?",
    "Apakah sistem ALELEON memiliki backup jika saya tidak sengaja menghapus data di HOME?",
    "Email resmi apa yang harus saya hubungi jika ingin submit support ticket?",
    "Saya mau menjalankan simulasi GROMACS, apa nama binary MPI yang dipakai? Apakah gmx atau yang lain?",
    "Di EFIRO Account Manager, aplikasi authenticator apa saja yang didukung untuk fitur 2FA?",
    "Apa perintah terminal untuk mengecek status antrian job saya di Slurm?",
    "Apakah ALELEON mendukung instalasi package Python menggunakan pip?",
    "Modul Lmod apa yang harus saya load jika ingin menggunakan compiler GCC versi 15.2.0?",
    # LEVEL 2
    "Saya ingin buka sesi interaktif JupyterLab menggunakan GPU. Apa bedanya partisi torti dan tilla, dan mana yang harus saya pilih?",
    "Saya punya file simulasi.ipynb. Bagaimana urutan langkah menjalankannya sebagai batch job di Job Composer EWS menggunakan conda environment saya sendiri?",
    "Jelaskan perbedaan arti status job 'PD' dan 'CG' saat saya mengecek squeue. Lalu sebutkan satu contoh Reason kenapa job bisa berstatus PD!",
    "Sebagai pengguna dari Akun Institusi, apakah job saya dibatasi maksimal 128 core CPU seperti akun perseorangan, dan apakah saya menggunakan sistem kuota (beli di awal)?",
    "Saya mau ganti password akun ALELEON saya. Di portal web mana saya harus login, dan menu apa yang harus diklik?",
    "File upload saya ukurannya 500 MB. Kenapa saya selalu gagal upload lewat menu Files di EWS, dan apa solusi spesifik serta alamat host yang harus saya gunakan?",
    "Saya ingin mengkompilasi code C++ menggunakan compiler AMD target Zen 2 dan OpenMPI terbaru. Modul apa saja yang harus saya module load secara berurutan?",
    "Saya mau pre-processing data GROMACS menggunakan binary gmx_mpi. Boleh tidak saya menjalankannya di Login Node? Jika boleh, apa syaratnya agar tidak di-kill admin?",
    "Jika saya menjalankan batch job lalu tiba-tiba koneksi internet rumah saya mati dan laptop saya disconnect dari VPN ALELEON, apakah job saya di Slurm ikut berhenti?",
    "Apa bedanya Effective Core Hour dengan Actual Core Hour di dalam sistem ALELEON?",
    # LEVEL 3
    "Saya menjalankan simulasi FLACS-CFD dengan 192 proses MPI murni. Di partisi epyc, otomatis job ini butuh lebih dari 1 node. Berapa angka yang harus saya tulis persisnya di #SBATCH --mem= jika total RAM yang saya butuhkan untuk seluruh job adalah 400GB?",
    "(Troubleshooting) Saya submit job GROMACS tapi selalu gagal dengan pesan error Invalid syntax. Di script saya menulis #SBATCH --mem= 64 GB. Apa yang salah dari tulisan saya?",
    "Saya submit 3 batch job berturut-turut. Job 1 pakai 64 CPU. Job 2 pakai 32 CPU. Kenapa saat saya submit Job 3 yang butuh 64 CPU, statusnya malah PENDING dengan tulisan QOSMaxCpuPerUserLimit, padahal node epyc masih banyak yang kosong?",
    "(Troubleshooting) Job saya berstatus PD dengan alasan AssocMaxWallDurationPerJobLimit. Di script saya menulis #SBATCH --time=4-00:00:00. Akun saya adalah akun perseorangan biasa. Mengapa tertahan?",
    "Saya mau menjalankan 10 simulasi FLACS-CFD sekaligus menggunakan fitur Slurm Array. Setiap simulasi butuh 4 core CPU dan 8GB RAM. Di script, apakah saya harus menulis --cpus-per-task=40 atau --cpus-per-task=4?",
    "Saya menjalankan script dengan #SBATCH --ntasks=4 dan #SBATCH --cpus-per-task=8 untuk OpenMX hibrida. Berapa total core thread CPU yang saya konsumsi, dan berapa Actual Core Hour yang terpotong jika job ini jalan 2 jam?",
    "Saya butuh komputasi memori raksasa sebesar 350 GB untuk satu aplikasi yang non-MPI (tidak bisa dibagi ke banyak node). Partisi apa yang WAJIB saya gunakan agar tidak error kehabisan memori?",
    "Saya mencoba mengisi form Sesi JupyterLab di EFIRO. Saya set waktu 3 hari (72 jam) dan minta 1 GPU. Namun tombol Launch ditolak karena saldo kurang. Jika sisa kuota GPU Hour (GH) saya tinggal 50 GH, berapa maksimal hari/jam yang bisa saya ajukan?",
    "Kenapa saat saya meminta alokasi #SBATCH --ntasks=7, sistem Slurm ALELEON akan membulatkannya menjadi 8 dan saya ditagih biaya untuk 8 core?",
    "Apakah ada gunanya saya upload file Slaster-Koster (SK) ke setiap ruang Job Composer DFTB+? Ataukah ada cara yang lebih hemat storage?",
    # LEVEL 4
    "Berapa kapasitas ukuran penyimpanan (storage) SSD untuk satu node Login di ALELEON?",
    "Bagaimana langkah-langkah submit job menggunakan aplikasi MATLAB di ALELEON?",
    "Berapa biaya denda yang harus dibayar jika file di HOME saya melebihi kuota 150GB?",
    "Apakah saya bisa menginstal package R menggunakan perintah conda install r-seurat di ALELEON?",
    "Berapa kecepatan internet/bandwidth VPN jika saya akses dari luar pulau Jawa?",
    "Bagaimana cara mereset environment Python bawaan sistem (python 3.9) ke kondisi pabrik jika saya merusaknya?",
    "Apakah tersedia modul aplikasi ANSYS Fluent di ALELEON?",
    "Bagaimana cara menyambungkan ekstensi Remote-SSH dari aplikasi Visual Studio Code (VSCode) ke compute node ALELEON?",
    "Saya adalah user dari Singapura (WNA). Berapa tarif konversi Core Hour ke dalam US Dollar (USD)?",
    "Apa password standar/bawaan dari admin sebelum saya menggantinya di awal?",
    "Siapa nama Chief Technology Officer (CTO) dari EFISON yang membangun ALELEON ini?",
    "Bagaimana cara menghapus halaman Wiki ALELEON jika saya menemukan typo?",
    "Bagaimana cara membatalkan/mengakhiri perjanjian PKSPIAS untuk akun Institusi sebelum waktunya habis?",
    "Jika server ALELEON mati lampu, berapa jam daya tahan baterai UPS yang dimiliki EFISON?",
    "Bagaimana cara menggunakan AutoGluon untuk machine learning di sistem ini?",
    # LEVEL 5
    "Bagaimana cara membuat conda environment di aleleon?",
    "bagaimana cara menjalankan jupyter dengan conda environment sendiri?",
    "Versi Python default dari Anaconda3 2025.06-1 apa?",
    "Perintah apa untuk mengaktifkan Mamba 23.11.0-0?",
    "Bagaimana cara membuat modul pyload setelah conda env aktif?",
    "Perintah apa untuk melihat daftar modul pyload yang tersedia?",
    "Di partisi GPU mana batch job conda berjalan?",
    "Apa email support admin ALELEON?",
    "Jam kerja support EFISON kapan?",
    "Apa saja pilihan cara menjalankan komputasi Python dengan conda env di ALELEON?",
    "Apa perbedaan antara menjalankan batch job via Job Composer EWS dan via terminal Slurm?",
    "Bagaimana langkah lengkap membuat conda env baru dan modul pyload dari awal?",
    "Apa saja status job di squeue dan artinya masing-masing?",
    "Bagaimana cara mengisi formulir Jupyter di EWS untuk conda env user?",
]


# ═══════════════════════════════════════════════════════════════════════
# Helper: Load wiki documents (reused from rag_app.py)
# ═══════════════════════════════════════════════════════════════════════

def load_wiki_documents(sitemap_url: str, requests_per_second: int = 2):
    """Fetch wiki pages from sitemap and split into chunks."""
    from langchain_core.documents import Document

    print("    Mengambil sitemap...")
    resp = requests.get(sitemap_url)
    root = ElementTree.fromstring(resp.content)
    ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text for loc in root.findall(".//ns:loc", ns)]
    print(f"    → {len(urls)} URL ditemukan")

    headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]
    html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4500, chunk_overlap=900, separators=["\n---", "\n\n", "\n", " "]
    )

    all_splits = []
    for i, url in enumerate(urls):
        try:
            time.sleep(1.0 / requests_per_second)
            page_resp = requests.get(url, timeout=30)
            soup = BeautifulSoup(page_resp.content, "lxml")
            content_div = soup.find("div", {"id": "mw-content-text"})
            if not content_div:
                continue

            content_html = str(content_div)
            page_title = url.split("/wiki/")[-1].replace("_", " ") if "/wiki/" in url else url

            html_docs = html_splitter.split_text(content_html)
            for doc in html_docs:
                doc.metadata["source"] = url
                doc.metadata["title"] = page_title
                if len(doc.page_content) > 4500:
                    all_splits.extend(text_splitter.split_documents([doc]))
                else:
                    all_splits.append(doc)

            print(f"    [{i+1}/{len(urls)}] {page_title}: {len(html_docs)} sections")
        except Exception as e:
            print(f"    [{i+1}/{len(urls)}] ERROR {url}: {e}")
            continue

    # Add source prefix like rag_app.py
    for s in all_splits:
        title = s.metadata.get("title", "Unknown")
        header = s.metadata.get("Header 2", s.metadata.get("Header 3", ""))
        prefix = f"[Sumber: {title}]"
        if header:
            prefix += f" [Section: {header}]"
        s.page_content = f"{prefix}\n{s.page_content}"

    return all_splits


# ═══════════════════════════════════════════════════════════════════════
# Helper: Embedding API calls
# ═══════════════════════════════════════════════════════════════════════

def embed_dense(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Call embedding service for dense-only embeddings."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"      Dense batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
        resp = requests.post(f"{EMBEDDING_API_URL}/embed", json={"texts": batch}, timeout=600)
        resp.raise_for_status()
        all_embeddings.extend(resp.json()["embeddings"])
    return all_embeddings


def embed_multi(texts: List[str], batch_size: int = 16) -> Dict[str, Any]:
    """Call embedding service for multi-mode embeddings."""
    all_dense, all_sparse, all_colbert = [], [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        n_batches = (len(texts) + batch_size - 1) // batch_size
        print(f"      Multi batch {i // batch_size + 1}/{n_batches} ({len(batch)} texts)")
        resp = requests.post(
            f"{EMBEDDING_API_URL}/embed/multi",
            json={
                "texts": batch,
                "return_dense": True,
                "return_sparse": True,
                "return_colbert": True,
            },
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("dense"):
            all_dense.extend(data["dense"])
        if data.get("sparse"):
            all_sparse.extend(data["sparse"])
        if data.get("colbert"):
            all_colbert.extend(data["colbert"])
    return {"dense": all_dense, "sparse": all_sparse, "colbert": all_colbert}


def embed_query_multi(text: str) -> Dict[str, Any]:
    """Get multi-mode embedding for a single query."""
    resp = requests.post(
        f"{EMBEDDING_API_URL}/embed/multi",
        json={
            "texts": [text],
            "return_dense": True,
            "return_sparse": True,
            "return_colbert": False,  # ColBERT query doesn't need colbert doc vecs
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    result = {}
    if data.get("dense"):
        result["dense"] = data["dense"][0]
    if data.get("sparse"):
        result["sparse"] = data["sparse"][0]
    return result


def embed_query_colbert(text: str) -> Dict[str, Any]:
    """Get colbert query embedding."""
    resp = requests.post(
        f"{EMBEDDING_API_URL}/embed/multi",
        json={
            "texts": [text],
            "return_dense": False,
            "return_sparse": False,
            "return_colbert": True,
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    return {"colbert": data["colbert"][0]}


# ═══════════════════════════════════════════════════════════════════════
# Helper: LLM Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_response(question: str, context: str) -> str:
    """Generate LLM response using vLLM (OpenAI-compatible API)."""
    from openai import OpenAI

    client = OpenAI(base_url=LLM_API_URL, api_key=os.getenv("LLM_API_KEY", ""))

    messages = [
        {
            "role": "system",
            "content": (
                "Kamu adalah agen AI asisten admin HPC Slurm yang ahli. "
                "Tugasmu adalah membantu user berdasarkan dokumen referensi yang diberikan. "
                "Gunakan Bahasa Indonesia yang jelas. Jawab HANYA berdasarkan dokumen referensi."
            ),
        },
        {
            "role": "user",
            "content": f"Dokumen Referensi:\n{context}\n\nPertanyaan: {question}",
        },
    ]

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=messages,
        max_tokens=2048,
        temperature=0.3,
        top_p=0.9,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════════
# Qdrant Collection Management
# ═══════════════════════════════════════════════════════════════════════

def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=300)


def create_collections(client: QdrantClient):
    """Create all 4 benchmark collections in Qdrant."""
    existing = [c.name for c in client.get_collections().collections]

    # 1. Dense
    if COLLECTION_DENSE not in existing:
        client.create_collection(
            COLLECTION_DENSE,
            vectors_config=VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        )
        print(f"    ✅ Created collection: {COLLECTION_DENSE}")
    else:
        print(f"    ⏩ Collection exists: {COLLECTION_DENSE}")

    # 2. Sparse
    if COLLECTION_SPARSE not in existing:
        client.create_collection(
            COLLECTION_SPARSE,
            vectors_config={},
            sparse_vectors_config={"text-sparse": SparseVectorParams()},
        )
        print(f"    ✅ Created collection: {COLLECTION_SPARSE}")
    else:
        print(f"    ⏩ Collection exists: {COLLECTION_SPARSE}")

    # 3. Multi-vector (ColBERT)
    if COLLECTION_MULTI not in existing:
        client.create_collection(
            COLLECTION_MULTI,
            vectors_config={
                "colbert": VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                )
            },
        )
        print(f"    ✅ Created collection: {COLLECTION_MULTI}")
    else:
        print(f"    ⏩ Collection exists: {COLLECTION_MULTI}")

    # 4. Hybrid (dense + sparse in single collection)
    if COLLECTION_HYBRID not in existing:
        client.create_collection(
            COLLECTION_HYBRID,
            vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={"text-sparse": SparseVectorParams()},
        )
        print(f"    ✅ Created collection: {COLLECTION_HYBRID}")
    else:
        print(f"    ⏩ Collection exists: {COLLECTION_HYBRID}")


def cleanup_collections(client: QdrantClient):
    """Delete all benchmark collections."""
    for name in [COLLECTION_DENSE, COLLECTION_SPARSE, COLLECTION_MULTI, COLLECTION_HYBRID]:
        try:
            client.delete_collection(name)
            print(f"    🗑️  Deleted: {name}")
        except Exception:
            print(f"    ⏩ Not found: {name}")


# ═══════════════════════════════════════════════════════════════════════
# Ingestion
# ═══════════════════════════════════════════════════════════════════════

def ingest(client: QdrantClient, documents):
    """Ingest documents into all 4 collections."""
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    # Get multi-mode embeddings for all texts
    print("\n  [Embed] Generating multi-mode embeddings...")
    t0 = time.time()
    multi = embed_multi(texts, batch_size=16)
    embed_time = time.time() - t0
    print(f"    ⏱️  Embedding time: {embed_time:.1f}s")

    n = len(texts)
    batch_size = 64

    # ── Dense ───────────────────────────────────────────────────────
    print(f"\n  [Ingest] {COLLECTION_DENSE} ({n} points)...")
    t0 = time.time()
    for i in range(0, n, batch_size):
        points = []
        for j in range(i, min(i + batch_size, n)):
            points.append(
                PointStruct(
                    id=j,
                    vector=multi["dense"][j],
                    payload={"text": texts[j], **metadatas[j]},
                )
            )
        client.upsert(COLLECTION_DENSE, points)
    dense_time = time.time() - t0
    print(f"    ⏱️  Dense ingest: {dense_time:.1f}s")

    # ── Sparse ──────────────────────────────────────────────────────
    print(f"\n  [Ingest] {COLLECTION_SPARSE} ({n} points)...")
    t0 = time.time()
    for i in range(0, n, batch_size):
        points = []
        for j in range(i, min(i + batch_size, n)):
            sp = multi["sparse"][j]
            points.append(
                PointStruct(
                    id=j,
                    vector={"text-sparse": SparseVector(indices=sp["indices"], values=sp["values"])},
                    payload={"text": texts[j], **metadatas[j]},
                )
            )
        client.upsert(COLLECTION_SPARSE, points)
    sparse_time = time.time() - t0
    print(f"    ⏱️  Sparse ingest: {sparse_time:.1f}s")

    # ── Multi-Vector (ColBERT) ──────────────────────────────────────
    # ColBERT vectors are huge (~N_tokens × 1024 per doc), so use small batches
    # to stay under Qdrant's 33MB payload limit.
    colbert_batch = 4
    print(f"\n  [Ingest] {COLLECTION_MULTI} ({n} points, batch={colbert_batch})...")
    t0 = time.time()
    for i in range(0, n, colbert_batch):
        points = []
        for j in range(i, min(i + colbert_batch, n)):
            points.append(
                PointStruct(
                    id=j,
                    vector={"colbert": multi["colbert"][j]},
                    payload={"text": texts[j], **metadatas[j]},
                )
            )
        client.upsert(COLLECTION_MULTI, points)
    multi_time = time.time() - t0
    print(f"    ⏱️  Multi-vector ingest: {multi_time:.1f}s")

    # ── Hybrid (Dense + Sparse) ─────────────────────────────────────
    hybrid_batch = 32
    print(f"\n  [Ingest] {COLLECTION_HYBRID} ({n} points, batch={hybrid_batch})...")
    t0 = time.time()
    for i in range(0, n, hybrid_batch):
        points = []
        for j in range(i, min(i + hybrid_batch, n)):
            sp = multi["sparse"][j]
            points.append(
                PointStruct(
                    id=j,
                    vector={
                        "dense": multi["dense"][j],
                        "text-sparse": SparseVector(indices=sp["indices"], values=sp["values"]),
                    },
                    payload={"text": texts[j], **metadatas[j]},
                )
            )
        client.upsert(COLLECTION_HYBRID, points)
    hybrid_time = time.time() - t0
    print(f"    ⏱️  Hybrid ingest: {hybrid_time:.1f}s")

    # ── Summary ─────────────────────────────────────────────────────
    ingest_summary = {
        "num_documents": n,
        "embedding_time_s": round(embed_time, 1),
        "dense_ingest_s": round(dense_time, 1),
        "sparse_ingest_s": round(sparse_time, 1),
        "multivec_ingest_s": round(multi_time, 1),
        "hybrid_ingest_s": round(hybrid_time, 1),
    }

    # Get collection sizes
    for name in [COLLECTION_DENSE, COLLECTION_SPARSE, COLLECTION_MULTI, COLLECTION_HYBRID]:
        info = client.get_collection(name)
        ingest_summary[f"{name}_points"] = info.points_count

    summary_path = os.path.join(OUTPUT_DIR, "ingest_summary.json")
    with open(summary_path, "w") as f:
        json.dump(ingest_summary, f, indent=2)
    print(f"\n  📄 Ingest summary saved to {summary_path}")

    return ingest_summary


# ═══════════════════════════════════════════════════════════════════════
# Query / Search Methods
# ═══════════════════════════════════════════════════════════════════════

def search_dense(client: QdrantClient, query_vec: List[float]) -> List[int]:
    """Search dense collection, return point IDs."""
    results = client.query_points(
        COLLECTION_DENSE,
        query=query_vec,
        limit=TOP_K,
    )
    return [r.id for r in results.points]


def search_sparse(client: QdrantClient, sparse_vec: Dict) -> List[int]:
    """Search sparse collection, return point IDs."""
    results = client.query_points(
        COLLECTION_SPARSE,
        query=SparseVector(indices=sparse_vec["indices"], values=sparse_vec["values"]),
        using="text-sparse",
        limit=TOP_K,
    )
    return [r.id for r in results.points]


def search_multivec(client: QdrantClient, colbert_vec: List[List[float]]) -> List[int]:
    """Search multi-vector collection (ColBERT), return point IDs."""
    results = client.query_points(
        COLLECTION_MULTI,
        query=colbert_vec,
        using="colbert",
        limit=TOP_K,
    )
    return [r.id for r in results.points]


def search_hybrid(client: QdrantClient, dense_vec: List[float], sparse_vec: Dict) -> List[int]:
    """Search hybrid collection with RRF fusion, return point IDs."""
    results = client.query_points(
        COLLECTION_HYBRID,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=TOP_K * 2,
            ),
            Prefetch(
                query=SparseVector(indices=sparse_vec["indices"], values=sparse_vec["values"]),
                using="text-sparse",
                limit=TOP_K * 2,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=TOP_K,
    )
    return [r.id for r in results.points]


def get_chunks_by_ids(client: QdrantClient, collection: str, ids: List[int]) -> List[str]:
    """Retrieve chunk texts by point IDs."""
    if not ids:
        return []
    points = client.retrieve(collection, ids=ids, with_payload=True)
    # Return in order of input ids
    id_to_text = {p.id: p.payload.get("text", "") for p in points}
    return [id_to_text.get(pid, "") for pid in ids]


# ═══════════════════════════════════════════════════════════════════════
# Overlap Metrics
# ═══════════════════════════════════════════════════════════════════════

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def compute_overlap(results: Dict[str, List[int]]) -> Dict[str, float]:
    """Compute pairwise Jaccard overlap between methods."""
    methods = list(results.keys())
    overlaps = {}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            score = jaccard(set(results[a]), set(results[b]))
            overlaps[f"{a}_vs_{b}"] = round(score, 3)
    return overlaps


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark(client: QdrantClient, questions: List[str], do_llm: bool = True):
    """Run benchmark on all questions across all 4 methods."""
    all_results = []

    total = len(questions)
    print(f"\n{'='*70}")
    print(f"  BENCHMARK: {total} questions × 4 methods (LLM={'ON' if do_llm else 'OFF'})")
    print(f"{'='*70}")

    for qi, question in enumerate(questions):
        print(f"\n  [{qi+1}/{total}] {question[:80]}...")

        # ── Embed query ──
        query_multi = embed_query_multi(question)
        query_colbert = embed_query_colbert(question)

        row = {"question_id": qi, "question": question}

        # ── Dense ──
        t0 = time.time()
        dense_ids = search_dense(client, query_multi["dense"])
        retrieval_dense_ms = (time.time() - t0) * 1000

        # ── Sparse ──
        t0 = time.time()
        sparse_ids = search_sparse(client, query_multi["sparse"])
        retrieval_sparse_ms = (time.time() - t0) * 1000

        # ── Multi-vector ──
        t0 = time.time()
        multi_ids = search_multivec(client, query_colbert["colbert"])
        retrieval_multi_ms = (time.time() - t0) * 1000

        # ── Hybrid ──
        t0 = time.time()
        hybrid_ids = search_hybrid(client, query_multi["dense"], query_multi["sparse"])
        retrieval_hybrid_ms = (time.time() - t0) * 1000

        row["retrieval_dense_ms"] = round(retrieval_dense_ms, 1)
        row["retrieval_sparse_ms"] = round(retrieval_sparse_ms, 1)
        row["retrieval_multi_ms"] = round(retrieval_multi_ms, 1)
        row["retrieval_hybrid_ms"] = round(retrieval_hybrid_ms, 1)

        # ── Overlap ──
        results_map = {
            "dense": dense_ids,
            "sparse": sparse_ids,
            "multi": multi_ids,
            "hybrid": hybrid_ids,
        }
        overlaps = compute_overlap(results_map)
        row.update(overlaps)

        # ── Top-K IDs for reference ──
        row["dense_ids"] = dense_ids
        row["sparse_ids"] = sparse_ids
        row["multi_ids"] = multi_ids
        row["hybrid_ids"] = hybrid_ids

        # ── LLM generation (end-to-end) ──
        if do_llm:
            for method_name, method_ids, collection in [
                ("dense", dense_ids, COLLECTION_DENSE),
                ("sparse", sparse_ids, COLLECTION_SPARSE),
                ("multi", multi_ids, COLLECTION_MULTI),
                ("hybrid", hybrid_ids, COLLECTION_HYBRID),
            ]:
                chunks = get_chunks_by_ids(client, collection, method_ids)
                context = "\n\n".join(chunks)
                t0 = time.time()
                try:
                    answer = generate_response(question, context)
                    e2e_ms = (time.time() - t0) * 1000
                    row[f"e2e_{method_name}_ms"] = round(e2e_ms, 1)
                    row[f"answer_{method_name}"] = answer.strip()[:500]  # Truncate for CSV
                except Exception as e:
                    row[f"e2e_{method_name}_ms"] = -1
                    row[f"answer_{method_name}"] = f"ERROR: {e}"
                    print(f"      ⚠️  LLM error ({method_name}): {e}")

        all_results.append(row)

        # Print quick summary for this question
        print(f"      Retrieval: D={retrieval_dense_ms:.0f}ms  S={retrieval_sparse_ms:.0f}ms  "
              f"M={retrieval_multi_ms:.0f}ms  H={retrieval_hybrid_ms:.0f}ms")
        print(f"      Overlap: D-S={overlaps.get('dense_vs_sparse','-'):.2f}  "
              f"D-M={overlaps.get('dense_vs_multi','-'):.2f}  "
              f"D-H={overlaps.get('dense_vs_hybrid','-'):.2f}  "
              f"S-H={overlaps.get('sparse_vs_hybrid','-'):.2f}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_report(results: List[Dict], ingest_summary: Dict = None):
    """Generate summary report and CSV output."""

    # ── Save detailed CSV ──
    csv_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    if results:
        # Exclude list columns from CSV
        csv_keys = [k for k in results[0].keys() if not isinstance(results[0][k], list)]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                csv_row = {k: v for k, v in row.items() if k in csv_keys}
                writer.writerow(csv_row)
        print(f"\n  📄 Detailed results: {csv_path}")

    # ── Compute aggregated stats ──
    methods = ["dense", "sparse", "multi", "hybrid"]

    print(f"\n{'='*70}")
    print("  📊 BENCHMARK RESULTS SUMMARY")
    print(f"{'='*70}")

    # Retrieval time
    print(f"\n  {'Method':<12} {'Avg Retrieval (ms)':<20} {'P50 (ms)':<12} {'P95 (ms)':<12}")
    print(f"  {'-'*56}")
    for m in methods:
        key = f"retrieval_{m}_ms"
        vals = [r[key] for r in results if key in r and r[key] >= 0]
        if vals:
            avg = sum(vals) / len(vals)
            vals_sorted = sorted(vals)
            p50 = vals_sorted[len(vals_sorted) // 2]
            p95 = vals_sorted[int(len(vals_sorted) * 0.95)]
            print(f"  {m:<12} {avg:<20.1f} {p50:<12.1f} {p95:<12.1f}")

    # E2E time (if available)
    if any(f"e2e_dense_ms" in r for r in results):
        print(f"\n  {'Method':<12} {'Avg E2E (ms)':<20} {'P50 (ms)':<12} {'P95 (ms)':<12}")
        print(f"  {'-'*56}")
        for m in methods:
            key = f"e2e_{m}_ms"
            vals = [r[key] for r in results if key in r and r[key] >= 0]
            if vals:
                avg = sum(vals) / len(vals)
                vals_sorted = sorted(vals)
                p50 = vals_sorted[len(vals_sorted) // 2]
                p95 = vals_sorted[int(len(vals_sorted) * 0.95)]
                print(f"  {m:<12} {avg:<20.1f} {p50:<12.1f} {p95:<12.1f}")

    # Overlap stats
    overlap_keys = [
        "dense_vs_sparse", "dense_vs_multi", "dense_vs_hybrid",
        "sparse_vs_multi", "sparse_vs_hybrid", "multi_vs_hybrid",
    ]
    print(f"\n  {'Pair':<24} {'Avg Jaccard':<14} {'Min':<10} {'Max':<10}")
    print(f"  {'-'*58}")
    for ok in overlap_keys:
        vals = [r.get(ok, 0) for r in results if ok in r]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {ok:<24} {avg:<14.3f} {min(vals):<10.3f} {max(vals):<10.3f}")

    # Ingest summary
    if ingest_summary:
        print(f"\n  ── Ingestion Summary ──")
        print(f"  Documents: {ingest_summary.get('num_documents', '?')}")
        print(f"  Embedding time: {ingest_summary.get('embedding_time_s', '?')}s")
        print(f"  Dense ingest: {ingest_summary.get('dense_ingest_s', '?')}s")
        print(f"  Sparse ingest: {ingest_summary.get('sparse_ingest_s', '?')}s")
        print(f"  Multi-vec ingest: {ingest_summary.get('multivec_ingest_s', '?')}s")
        print(f"  Hybrid ingest: {ingest_summary.get('hybrid_ingest_s', '?')}s")

    # Save JSON summary
    summary = {
        "total_questions": len(results),
        "retrieval_avg_ms": {},
        "e2e_avg_ms": {},
        "overlap_avg": {},
    }
    for m in methods:
        key = f"retrieval_{m}_ms"
        vals = [r[key] for r in results if key in r and r[key] >= 0]
        if vals:
            summary["retrieval_avg_ms"][m] = round(sum(vals) / len(vals), 1)
        key = f"e2e_{m}_ms"
        vals = [r[key] for r in results if key in r and r[key] >= 0]
        if vals:
            summary["e2e_avg_ms"][m] = round(sum(vals) / len(vals), 1)
    for ok in overlap_keys:
        vals = [r.get(ok, 0) for r in results if ok in r]
        if vals:
            summary["overlap_avg"][ok] = round(sum(vals) / len(vals), 3)

    if ingest_summary:
        summary["ingest"] = ingest_summary

    json_path = os.path.join(OUTPUT_DIR, "benchmark_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  📄 Summary JSON: {json_path}")

    print(f"\n{'='*70}")
    print("  ✅ BENCHMARK COMPLETE")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Benchmark Dense vs Sparse vs Multi-Vec vs Hybrid")
    parser.add_argument(
        "--mode",
        choices=["ingest", "query", "all", "cleanup"],
        default="all",
        help="Mode: ingest, query, all, or cleanup",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation (retrieval-only benchmark)",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=0,
        help="Limit number of questions (0 = all)",
    )
    args = parser.parse_args()

    client = get_client()
    ingest_summary = None

    if args.mode in ("cleanup",):
        print("\n[CLEANUP] Deleting benchmark collections...")
        cleanup_collections(client)
        return

    if args.mode in ("ingest", "all"):
        print("\n[1] Creating Qdrant collections...")
        create_collections(client)

        print("\n[2] Loading wiki documents...")
        documents = load_wiki_documents(SITEMAP_URL)
        print(f"    → Total chunks: {len(documents)}")

        print("\n[3] Ingesting into all 4 collections...")
        ingest_summary = ingest(client, documents)

    if args.mode in ("query", "all"):
        # Load ingest summary if exists
        if ingest_summary is None:
            summary_path = os.path.join(OUTPUT_DIR, "ingest_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    ingest_summary = json.load(f)

        questions = QUESTIONS
        if args.questions > 0:
            questions = questions[: args.questions]

        print(f"\n[4] Running benchmark ({len(questions)} questions)...")
        results = run_benchmark(client, questions, do_llm=not args.no_llm)

        print("\n[5] Generating report...")
        generate_report(results, ingest_summary)


if __name__ == "__main__":
    main()
