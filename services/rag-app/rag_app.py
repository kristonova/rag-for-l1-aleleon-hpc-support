#!/usr/bin/env python3
"""
rag_app.py — RAG Application for Podman
=============================================================================
This application orchestrates the RAG chain using embedding-service, vllm-rocm,
and qdrant services. It replaces the local embedding computation with API
calls and uses OpenAI-compatible API for vLLM inference.
=============================================================================
"""

import os
import requests
import re
from xml.etree import ElementTree
from typing import List
from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from openai import OpenAI
from langchain_core.documents import Document
from bs4 import BeautifulSoup
import time


# === KONFIGURASI ===
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION_NAME = "wiki_aleleon"
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://embedding-service:8001")


class EmbeddingServiceClient(Embeddings):
    """
    LangChain-compatible wrapper yang memanggil embedding-service REST API.
    """

    def __init__(self, api_url: str = EMBEDDING_API_URL):
        self.api_url = api_url

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
            print(f"  Embedding batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size} ({len(batch)} texts)...")
            all_embeddings.extend(self._call_api(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._call_api([text])[0]


def load_wiki_documents(sitemap_url, requests_per_second=2):
    """
    Document Structure-Based Loading:
    1. Parse sitemap XML → ambil semua URL
    2. Fetch setiap halaman
    3. Ekstrak <div id="mw-content-text"> sebagai HTML (BUKAN plain text)
    4. Split berdasarkan heading HTML (h2, h3)
    5. Fallback ke RecursiveCharacterTextSplitter jika chunk masih terlalu besar
    """

    # --- Step 1: Parse sitemap ---
    print("    Mengambil sitemap...")
    resp = requests.get(sitemap_url)
    root = ElementTree.fromstring(resp.content)
    ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text for loc in root.findall(".//ns:loc", ns)]
    print(f"    → {len(urls)} URL ditemukan")

    # --- Step 2-3: Fetch & extract HTML content ---
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)

    # Fallback splitter untuk chunk yang masih terlalu besar
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4500,
        chunk_overlap=900,
        separators=["\n---", "\n\n", "\n", " "],
    )

    all_splits = []

    for i, url in enumerate(urls):
        try:
            time.sleep(1.0 / requests_per_second)
            page_resp = requests.get(url, timeout=30)
            soup = BeautifulSoup(page_resp.content, "lxml")

            # Ekstrak konten utama wiki (masih HTML!)
            content_div = soup.find("div", {"id": "mw-content-text"})
            if not content_div:
                continue

            content_html = str(content_div)
            page_title = url.split("/wiki/")[-1].replace("_", " ") if "/wiki/" in url else url

            # --- Step 4: Split berdasarkan heading HTML ---
            html_docs = html_splitter.split_text(content_html)

            for doc in html_docs:
                # Tambahkan metadata
                doc.metadata["source"] = url
                doc.metadata["title"] = page_title

                # --- Step 5: Fallback split jika chunk terlalu besar ---
                if len(doc.page_content) > 4500:
                    sub_splits = text_splitter.split_documents([doc])
                    all_splits.extend(sub_splits)
                else:
                    all_splits.append(doc)

            print(f"    [{i+1}/{len(urls)}] {page_title}: {len(html_docs)} sections")

        except Exception as e:
            print(f"    [{i+1}/{len(urls)}] ERROR {url}: {e}")
            continue

    return all_splits


def qdrant_collection_exists() -> bool:
    """Cek apakah collection Qdrant sudah ada dan berisi data."""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections().collections
        return any(c.name == QDRANT_COLLECTION_NAME for c in collections)
    except Exception:
        return False


def build_vectorstore(embeddings) -> QdrantVectorStore:
    """
    Scraping wiki → splitting → simpan ke Qdrant permanen.
    Hanya dijalankan SEKALI saat pertama kali.
    """
    print("[1] Membaca & splitting halaman wiki berdasarkan struktur HTML...")
    splits = load_wiki_documents(
        sitemap_url="https://wiki.efisonlt.com/sitemap/sitemap-wiki.efisonlt.com-NS_0-0.xml",
        requests_per_second=2,
    )

    # Tambahkan label sumber ke setiap chunk
    for s in splits:
        title = s.metadata.get("title", "Unknown")
        header = s.metadata.get("Header 2", s.metadata.get("Header 3", ""))
        prefix = f"[Sumber: {title}]"
        if header:
            prefix += f" [Section: {header}]"
        s.page_content = f"{prefix}\n{s.page_content}"

    print(f"    → Total chunks: {len(splits)}")

    # DEBUG: Tampilkan isi setiap chunk
    for i, s in enumerate(splits):
        print(f"\n    [Chunk {i}] ({len(s.page_content)} chars):")
        print(f"    {s.page_content[:120]}...")

    # Simpan ke Qdrant
    print(f"[2] Menyimpan vektor ke database Qdrant di '{QDRANT_URL}'...")
    vectorstore = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_NAME,
    )
    print(f"    ✅ Qdrant tersimpan permanen di '{QDRANT_URL}'")

    return vectorstore


## Jika ingin re-scrape (misal wiki berubah):
## Hapus collection di Qdrant lalu jalankan ulang rag_app.py
def load_vectorstore(embeddings) -> QdrantVectorStore:
    """
    Load Qdrant collection yang sudah ada.
    Tidak perlu scraping ulang.
    """
    print(f"[1] ⚡ Memuat Qdrant dari '{QDRANT_URL}' (tanpa scraping)...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION_NAME,
    )
    count = client.get_collection(QDRANT_COLLECTION_NAME).points_count
    print(f"    ✅ Berhasil memuat {count} chunks dari Qdrant")

    return vectorstore


def wait_for_vllm(api_url, timeout=600, interval=10):
    """Tunggu sampai vLLM service ready (model selesai loading)."""
    # Base URL: strip /v1 suffix for health check
    base_url = api_url.rstrip("/").removesuffix("/v1")
    health_url = f"{base_url}/health"
    print(f"    Menunggu vLLM ready di {health_url} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                print("    ✅ vLLM ready!")
                return True
        except requests.ConnectionError:
            pass
        print(f"    ⏳ vLLM belum ready, retry dalam {interval}s ...")
        time.sleep(interval)
    raise RuntimeError(f"vLLM tidak ready setelah {timeout}s")


def generate_response(question, context, api_url=None):
    """Generate response using vLLM with non-thinking mode."""
    if api_url is None:
        api_url = os.getenv("LLM_API_URL", "http://vllm-rocm:8000/v1")
    
    # Configure OpenAI client to use vLLM endpoint
    client = OpenAI(
        base_url=api_url,
        api_key=os.getenv("LLM_API_KEY", "")  # If authentication is required
    )
    
    messages = [
        {
            "role": "system",
            "content": """Kamu adalah agen AI asisten admin HPC Slurm yang ahli. Tugasmu adalah membantu user berdasarkan dokumen referensi yang diberikan. Gunakan Bahasa Indonesia yang jelas.

Aturan:
0. Tidak perlu bilang kalo berdasarkan dokumen referensi yang diberikan, langsung saja menyapa klien dengan sopan. Jangan outputkan chain of thought atau proses berpikirmu, langsung saja jawab dengan ringkas dan jelas.
1. Jawab HANYA berdasarkan dokumen referensi. KUTIP langkah-langkah dan perintah PERSIS seperti di dokumen. Jangan menambahkan langkah atau perintah yang tidak ada di dokumen. Anda adalah L1 Support bot ALELEON. JANGAN PERNAH menyarankan solusi atau tool di luar dokumen yang diberikan. Jika di dokumen tidak ada, katakan Anda tidak tahu.
2a. Sertakan angka, nama, versi, dan spesifikasi PERSIS seperti tertulis di dokumen. Jangan membulatkan atau menambah presisi. Contoh: jika dokumen bilang ">=11", jawab ">=11", BUKAN "11.0" atau "11.2".
2b. Gunakan penomoran (1, 2, 3) untuk langkah-langkah, JANGAN gunakan bullet points/titik.
3. Jika informasi bisa DISIMPULKAN dari dokumen, berikan kesimpulan tersebut.
4. Jika informasi benar-benar TIDAK ADA di dokumen, katakan "Saya tidak menemukan informasi tersebut di sistem."
5. Jangan mengarang angka, rumus, perintah, URL, nama partisi, atau prosedur yang tidak ada di dokumen. KHUSUSNYA jangan mengarang nama partisi seperti "bigmem" jika tidak disebutkan di dokumen.
6. JANGAN mengganti perintah dari dokumen dengan perintah alternatif. Contoh: jika dokumen menulis "source activate", JANGAN ganti dengan "conda activate".
7. Bedakan "minimal" dan "maksimal". Jika dokumen hanya menyebutkan "minimal X" TANPA batas maksimal, jawab bahwa informasi batas maksimal tidak tersedia di dokumen.
8. Perhatikan label LEGACY. Jika halaman bertanda LEGACY untuk versi lama (misal Mk.III), JANGAN terapkan info tersebut untuk versi baru (Mk.V).
9. Jawab dengan LENGKAP termasuk contoh perintah dan kode jika ada di dokumen. Jangan hanya menjawab kalimat pembuka lalu berhenti.
10. WAJIB menjawab minimal 2 kalimat. Jangan mengeluarkan jawaban kosong."""
        },
        {
            "role": "user",
            "content": f"""Dokumen Referensi:
{context}

Pertanyaan: {question}"""
        }
    ]

    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"),
        messages=messages,
        max_tokens=32768,
        temperature=0.3,
        top_p=0.9,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},  # Non-thinking mode
        }
    )
    
    return response.choices[0].message.content


def generate_source_justifications(question, answer, docs, api_url=None):
    """Generate 'Why This Source' justification for each unique retrieved source."""
    if api_url is None:
        api_url = os.getenv("LLM_API_URL", "http://vllm-rocm:8000/v1")

    # Deduplicate sources by (title, header)
    unique_sources = []
    seen_keys = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        header = doc.metadata.get("Header 2", doc.metadata.get("Header 3", ""))
        key = (title, header)
        if key not in seen_keys:
            seen_keys.append(key)
            snippet = doc.page_content[:200].replace("\n", " ")
            unique_sources.append({"title": title, "header": header, "snippet": snippet})

    if not unique_sources:
        return []

    # Build source list for prompt
    source_list = ""
    for i, src in enumerate(unique_sources, 1):
        label = src["title"]
        if src["header"]:
            label += f" → {src['header']}"
        source_list += f"{i}. [{label}]: {src['snippet']}\n"

    client = OpenAI(
        base_url=api_url,
        api_key=os.getenv("LLM_API_KEY", "")
    )

    messages = [
        {
            "role": "system",
            "content": "Kamu adalah asisten yang menjelaskan relevansi sumber dokumen. Berikan justifikasi singkat (1 kalimat) untuk setiap sumber. Jangan outputkan chain of thought."
        },
        {
            "role": "user",
            "content": f"""Untuk setiap sumber berikut, berikan 1 kalimat singkat mengapa sumber tersebut relevan untuk menjawab pertanyaan user.

Pertanyaan: {question}
Jawaban: {answer[:500]}

Sumber:
{source_list}
Format output HARUS persis (hanya nomor dan alasan, tanpa label sumber):
1. [alasan]
2. [alasan]
..."""
        }
    ]

    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"),
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        raw = response.choices[0].message.content.strip()

        # Parse numbered list → list of reason strings
        justifications = []
        for line in raw.split("\n"):
            # Hapus markdown bold/italic agar angka maju ke paling depan
            line_clean = line.strip().replace("*", "") 
            if line_clean and line_clean[0].isdigit():
                parts = line_clean.split(".", 1)
                if len(parts) == 2:
                    justifications.append(parts[1].strip())
                else:
                    justifications.append(line_clean)

        return justifications

    except Exception as e:
        print(f"    ⚠️  Gagal generate justifikasi sumber: {e}")
        return []


def create_rag_chain(vectorstore, llm_api_url=None):
    """Create RAG chain using embedding service and vLLM API."""

    def retrieve_and_answer(question):
        # Retrieve relevant documents from Qdrant
        docs = vectorstore.similarity_search(question, k=10)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate response using vLLM API
        answer = generate_response(question, context, llm_api_url)

        # Generate "Why This Source" justifications
        justifications = generate_source_justifications(
            question, answer, docs, llm_api_url
        )

        return {
            "answer": answer,
            "context": docs,
            "justifications": justifications
        }

    return retrieve_and_answer


def main():
    print("Memulai proses RAG dengan mesin vLLM...\n")

    # --- FASE 1: MEMASUKKAN DATA (INGESTION) ---

    # 1. Setup Embedding Client (API-backed, model di embedding-service)
    print("[0] Menghubungi embedding-service API...")
    embeddings = EmbeddingServiceClient()

    # 2. Cek apakah Qdrant collection sudah ada → skip scraping jika sudah
    if qdrant_collection_exists():
        vectorstore = load_vectorstore(embeddings)
    else:
        vectorstore = build_vectorstore(embeddings)

    # --- FASE 2: SETUP RAG CHAIN ---

    print("\n[3] Membuat RAG chain...")
    llm_api_url = os.getenv("LLM_API_URL", "http://vllm-rocm:8000/v1")
    wait_for_vllm(llm_api_url)
    rag_chain = create_rag_chain(vectorstore, llm_api_url=llm_api_url)

    # --- FASE 3: TANYA JAWAB (RETRIEVAL & GENERATION) ---

    pertanyaan_list = [
        # =============================================================
        # LEVEL 1: Fakta Langsung / Direct Facts (20 pertanyaan)
        # Jawaban bisa ditemukan langsung di satu chunk/paragraf
        # =============================================================

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

        # =============================================================
        # LEVEL 2: Gabungan Info / Multi-Chunk (10 pertanyaan)
        # Butuh menggabungkan info dari beberapa bagian dokumen
        # =============================================================

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

        # =============================================================
        # LEVEL 3: Reasoning / Deduksi & Troubleshooting (10 pertanyaan)
        # Butuh menyimpulkan dari informasi yang tersedia
        # =============================================================

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

        # =============================================================
        # LEVEL 4: Anti-Hallucination / Out-of-Context (15 pertanyaan)
        # Jawaban TIDAK ada di dokumen, model harus jujur
        # =============================================================

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

        # =============================================================
        # LEVEL 5: Pertanyaan Tambahan (dari rag_app.py)
        # =============================================================

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

    print(f"\n[4] Menguji RAG chain dengan {len(pertanyaan_list)} pertanyaan...")
    for i, pertanyaan in enumerate(pertanyaan_list, 1):
        print(f"\n{'='*60}")
        print(f"[Q{i}/{len(pertanyaan_list)}] {pertanyaan}")
        print("-" * 60)
        try:
            result = rag_chain(pertanyaan)
            print(result['answer'].strip())

            # Tampilkan sumber dokumen yang digunakan
            if result.get('context'):
                print(f"\n    📚 Sumber ({len(result['context'])} chunks):")
                justifications = result.get('justifications', [])
                seen = []
                justification_idx = 0
                for doc in result['context']:
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
                        # Print justification if available
                        if justification_idx < len(justifications):
                            print(f"      💡 Why: {justifications[justification_idx]}")
                        justification_idx += 1
        except Exception as e:
            print(f"    → ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Selesai — {len(pertanyaan_list)} pertanyaan dijawab.")


if __name__ == "__main__":
    main()