import os
import gc
import torch
import requests
from xml.etree import ElementTree
from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import VLLM
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from bs4 import BeautifulSoup
import time
from typing import List


# === KONFIGURASI PATH PENYIMPANAN ===
CHROMA_PERSIST_DIR = "./chroma_db"  # Folder untuk menyimpan ChromaDB secara permanen
CHROMA_COLLECTION_NAME = "wiki_aleleon"


class E5Embeddings(HuggingFaceEmbeddings):
    """
    Wrapper untuk multilingual-e5-large yang otomatis menambahkan
    prefix 'passage: ' saat embed dokumen dan 'query: ' saat embed pertanyaan.
    
    Ini penting karena model E5 dilatih dengan format:
    - "query: <pertanyaan>"   → untuk teks pencarian user
    - "passage: <dokumen>"    → untuk teks dokumen/konteks
    
    Tanpa prefix ini, akurasi retrieval turun signifikan.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Tambahkan prefix 'passage: ' untuk setiap dokumen."""
        prefixed = [f"passage: {t}" for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        """Tambahkan prefix 'query: ' untuk pertanyaan user."""
        return super().embed_query(f"query: {text}")


def load_wiki_documents(sitemap_url, requests_per_second=2):
    """
    Load seluruh artikel wiki sebagai 1 chunk per halaman (tanpa splitting).
    1. Parse sitemap XML → ambil semua URL
    2. Fetch setiap halaman
    3. Ekstrak <div id="mw-content-text"> → ambil plain text
    4. 1 artikel = 1 Document (tanpa split heading)
    """

    # --- Step 1: Parse sitemap ---
    print("    Mengambil sitemap...")
    resp = requests.get(sitemap_url)
    root = ElementTree.fromstring(resp.content)
    ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text for loc in root.findall(".//ns:loc", ns)]
    print(f"    → {len(urls)} URL ditemukan")

    # --- Step 2-3: Fetch & extract sebagai 1 chunk per artikel ---
    all_docs = []

    for i, url in enumerate(urls):
        try:
            time.sleep(1.0 / requests_per_second)
            page_resp = requests.get(url, timeout=30)
            soup = BeautifulSoup(page_resp.content, "lxml")

            # Ekstrak konten utama wiki
            content_div = soup.find("div", {"id": "mw-content-text"})
            if not content_div:
                continue

            # Ambil plain text dari HTML (1 artikel penuh = 1 chunk)
            page_text = content_div.get_text(separator="\n", strip=True)
            if not page_text.strip():
                continue

            page_title = url.split("/wiki/")[-1].replace("_", " ") if "/wiki/" in url else url

            doc = Document(
                page_content=page_text,
                metadata={
                    "source": url,
                    "title": page_title,
                }
            )
            all_docs.append(doc)

            print(f"    [{i+1}/{len(urls)}] {page_title}: {len(page_text)} chars")

        except Exception as e:
            print(f"    [{i+1}/{len(urls)}] ERROR {url}: {e}")
            continue

    return all_docs


def chroma_db_exists() -> bool:
    """Cek apakah ChromaDB sudah ada di disk dan berisi data."""
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return False
    # Cek apakah folder tidak kosong (minimal ada file chroma.sqlite3)
    contents = os.listdir(CHROMA_PERSIST_DIR)
    return len(contents) > 0


def build_vectorstore(embeddings) -> Chroma:
    """
    Scraping wiki → splitting → simpan ke ChromaDB permanen.
    Hanya dijalankan SEKALI saat pertama kali.
    """
    print("[1] Membaca & splitting halaman wiki berdasarkan struktur HTML...")
    splits = load_wiki_documents(
        sitemap_url="https://wiki.efisonlt.com/sitemap/sitemap-wiki.efisonlt.com-NS_0-0.xml",
        requests_per_second=2,
    )

    # Split chunk besar menjadi lebih kecil
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )

    final_splits = []
    for doc in splits:
        if len(doc.page_content) > 2500:
            sub_docs = text_splitter.split_documents([doc])
            final_splits.extend(sub_docs)
        else:
            final_splits.append(doc)

    splits = final_splits
    print(f"    → Setelah splitting: {len(splits)} chunks")

    # DEBUG: Tampilkan isi setiap chunk
    for i, s in enumerate(splits):
        print(f"\n    [Chunk {i}] ({len(s.page_content)} chars):")
        print(f"    {s.page_content[:120]}...")

    # Simpan ke ChromaDB dengan persist_directory (PERMANEN di disk)
    print(f"[4] Menyimpan vektor ke database Chroma di '{CHROMA_PERSIST_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    print(f"    ✅ ChromaDB tersimpan permanen di '{CHROMA_PERSIST_DIR}'")

    return vectorstore


## Jika ingin re-scrape (misal wiki berubah): 
##rm -rf ./chroma_db && python rag_slurm_vllm.py
def load_vectorstore(embeddings) -> Chroma:
    """
    Load ChromaDB yang sudah ada dari disk.
    Tidak perlu scraping ulang.
    """
    print(f"[1] ⚡ Memuat ChromaDB dari disk '{CHROMA_PERSIST_DIR}' (tanpa scraping)...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    # Cek jumlah dokumen di database
    count = vectorstore._collection.count()
    print(f"    ✅ Berhasil memuat {count} chunks dari ChromaDB")

    return vectorstore


def main():
    print("Memulai proses RAG dengan mesin vLLM...\n")

    # --- FASE 1: MEMASUKKAN DATA (INGESTION) ---

    # 1. Setup Model Embedding (harus load dulu sebelum cek DB)
    print("[3] Load model embedding lokal...")
    embeddings = E5Embeddings(
        model_name="intfloat/multilingual-e5-large",
    )

    # 2. Cek apakah ChromaDB sudah ada → skip scraping jika sudah
    if chroma_db_exists():
        vectorstore = load_vectorstore(embeddings)
    else:
        vectorstore = build_vectorstore(embeddings)


    # --- FASE 2: SETUP vLLM (ENGINE INFERENCE) ---

    print("\n[5] Memuat model Qwen ke GPU menggunakan vLLM...")
    print("    (Ini akan memakan waktu untuk alokasi KV Cache di VRAM)")

    # Konfigurasi vLLM
    llm = VLLM(
        model="lovedheart/Qwen3.5-9B-FP8",
        trust_remote_code=True,
        max_new_tokens=2048,
        temperature=0.3,                           
        top_p=0.9,
        tensor_parallel_size=1,
        #dtype="float16",
        vllm_kwargs={
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
            "max_model_len": 200000,
        }
    )


    # --- FASE 3: TANYA JAWAB (RETRIEVAL & GENERATION) ---

    # Setup Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Buat Prompt dengan format ChatML (untuk Qwen)
    template_qwen = """<|im_start|>system
Kamu adalah agen AI asisten admin HPC Slurm yang ahli. Tugasmu adalah membantu user berdasarkan dokumen referensi yang diberikan. Gunakan Bahasa Indonesia yang jelas.

/no_think 

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
10. WAJIB menjawab minimal 2 kalimat. Jangan mengeluarkan jawaban kosong.<|im_end|>
    <|im_start|>user
    Dokumen Referensi:
    {context}
    
    Pertanyaan: {input}<|im_end|>
    <|im_start|>assistant
    """
    
    prompt = PromptTemplate(
            template=template_qwen,
            input_variables=["context", "input"]
        )
    
    # ...existing code...

    # Rangkai rantai RAG (Chain)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # --- UJI COBA: BATCH SEMUA PERTANYAAN ---
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
    ]


    # Batch invoke: kumpulkan semua input sekaligus lalu proses satu loop
    inputs = [{"input": q} for q in pertanyaan_list]

    for i, inp in enumerate(inputs, 1):
        print(f"\n{'='*60}")
        print(f"[Q{i}/{len(inputs)}] {inp['input']}")
        print("-" * 60)
        hasil = rag_chain.invoke(inp)
        print(hasil['answer'].strip())

        # Tampilkan sumber dokumen yang digunakan
        if 'context' in hasil and hasil['context']:
            print(f"\n    📚 Sumber ({len(hasil['context'])} chunks):")
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

    print(f"\n{'='*60}")
    print(f"Selesai — {len(inputs)} pertanyaan dijawab.")


# ============================================================
# INI KUNCINYA: Mencegah spawn menjalankan ulang seluruh script
# ============================================================
if __name__ == '__main__':
    main()