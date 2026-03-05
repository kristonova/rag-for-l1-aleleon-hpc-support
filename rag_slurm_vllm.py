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


def main():
    print("Memulai proses RAG dengan mesin vLLM...\n")

    # --- FASE 1: MEMASUKKAN DATA (INGESTION) ---

    # 1. Load + Split dokumen dari wiki (Document Structure-Based)
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

    print(f"\n    → Total chunks: {len(splits)}")

    # DEBUG: Tampilkan isi setiap chunk
    for i, s in enumerate(splits):
        print(f"\n    [Chunk {i}] ({len(s.page_content)} chars):")
        print(f"    {s.page_content[:120]}...")

    # 2. Setup Model Embedding (Lokal via HuggingFace - CPU/GPU)
    print("[3] Load model embedding lokal...")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # 4. Simpan ke Vector Database (Chroma)
    print("[4] Menyimpan vektor ke database Chroma...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)


    # --- FASE 2: SETUP vLLM (ENGINE INFERENCE) ---

    print("\n[5] Memuat model Qwen ke GPU menggunakan vLLM...")
    print("    (Ini akan memakan waktu untuk alokasi KV Cache di VRAM)")

    # Konfigurasi vLLM
    llm = VLLM(
        model="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",  # ← Ganti kembali
        trust_remote_code=True,
        max_new_tokens=1024,
        temperature=0.6,                           
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        tensor_parallel_size=1,
        dtype="float16",
        vllm_kwargs={
            "gpu_memory_utilization": 0.80,
            "max_model_len": 32768,
            "enforce_eager": True
        }
    )


    # --- FASE 3: TANYA JAWAB (RETRIEVAL & GENERATION) ---

    # Setup Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ...existing code...
    # Buat Prompt dengan format ChatML (untuk Qwen)
    template_qwen = """<|im_start|>system
Kamu adalah agen AI asisten admin HPC Slurm yang ahli. Tugasmu adalah membantu user berdasarkan dokumen referensi yang diberikan. Gunakan Bahasa Indonesia yang jelas.

Aturan:
0. Tidak perlu bilang kalo berdasarkan dokumen referensi yang diberikan, langsung saja menyapa klien dengan sopan. Jangan outputkan chain of thought atau proses berpikirmu, langsung saja jawab dengan ringkas dan jelas.
1. Jawab HANYA berdasarkan dokumen referensi. KUTIP langkah-langkah dan perintah PERSIS seperti di dokumen. Jangan menambahkan langkah atau perintah yang tidak ada di dokumen.
2. Sertakan angka, nama, versi, dan spesifikasi PERSIS seperti tertulis di dokumen. Jangan membulatkan atau menambah presisi. Contoh: jika dokumen bilang ">=11", jawab ">=11", BUKAN "11.0" atau "11.2".
3. Jika informasi bisa DISIMPULKAN dari dokumen, berikan kesimpulan tersebut.
4. Jika informasi benar-benar TIDAK ADA di dokumen, katakan "Saya tidak menemukan informasi tersebut di sistem."
5. Jangan mengarang angka, rumus, perintah, URL, atau prosedur yang tidak ada di dokumen.
6. JANGAN mengganti perintah dari dokumen dengan perintah alternatif. Contoh: jika dokumen menulis "source activate", JANGAN ganti dengan "conda activate".
7. Bedakan "minimal" dan "maksimal". Jika dokumen hanya menyebutkan "minimal X" TANPA batas maksimal, jawab bahwa informasi batas maksimal tidak tersedia di dokumen.<im_end|>

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
        # LEVEL 1: Fakta Langsung (50 pertanyaan)
        # Jawaban bisa ditemukan langsung di satu chunk/paragraf
        # =============================================================

        # --- Umum ALELEON ---
        "Apa itu ALELEON Supercomputer?",
        "Apa email support admin ALELEON?",
        "Jam kerja support EFISON kapan?",
        "ALELEON Mk.V menggunakan platform apa?",
        "Sistem operasi apa yang digunakan ALELEON?",
        "Versi Slurm yang digunakan ALELEON berapa?",
        "Apa nama portal web untuk menggunakan ALELEON?",
        "Apa URL untuk login ke EFIRO Web Service?",
        "Apa itu EFIRO Account Manager?",
        "Bagaimana cara mengganti password akun ALELEON?",

        # --- Spesifikasi ---
        "Berapa jumlah node di partisi epyc?",
        "CPU apa yang digunakan di partisi epyc?",
        "Berapa RAM efektif per node di partisi epyc?",
        "Berapa RAM efektif di partisi epyc-jumbo?",
        "GPU apa yang digunakan di partisi ampere?",
        "Berapa jumlah GPU di compute node ampere?",
        "Apa interkoneksi yang digunakan di partisi epyc?",
        "Berapa kapasitas hot storage SSD di ALELEON?",
        "Berapa kapasitas cold storage HDD di ALELEON?",
        "Apakah ALELEON mendukung CUDA 11 ke bawah di partisi ampere?",

        # --- Core Hour & Biaya ---
        "Apa itu CCH?",
        "Apa itu GH?",
        "Berapa harga CCH untuk golongan perseorangan akademia?",
        "Berapa harga GH untuk golongan perseorangan non-akademia?",
        "Berapa storage HOME yang didapatkan pengguna perseorangan?",
        "Berapa masa aktif akun perseorangan?",
        "Apa rumus menghitung CPU Core Hour?",

        # --- Software ---
        "Apa nama modul GROMACS versi CPU 2025.3?",
        "Apa nama modul GROMACS versi GPU 2025.3?",
        "Versi ORCA berapa yang tersedia di ALELEON?",
        "Apa nama modul Quantum ESPRESSO 7.4.1?",
        "Apa nama modul CP2K versi GPU?",
        "Versi LAMMPS berapa yang tersedia di ALELEON?",
        "Apa nama modul SIESTA yang tersedia?",
        "Apa nama modul OpenMX 3.9?",
        "Apa nama modul PHASE/0 2025?",
        "Versi COMCOT berapa yang tersedia?",
        "Apa nama modul DFTB+ yang tersedia?",
        "Apa nama modul NAMD versi GPU?",
        "Apa nama modul NWChem yang tersedia?",

        # --- Python & Conda ---
        "Versi Python default dari Anaconda3 2025.06-1 apa?",
        "Perintah apa untuk mengaktifkan Mamba 23.11.0-0?",
        "Bagaimana cara membuat modul pyload setelah conda env aktif?",
        "Perintah apa untuk melihat daftar modul pyload yang tersedia?",
        "Apa perintah untuk mengaktifkan conda env dengan pyload?",
        "Di mana conda env dibuat secara default?",

        # --- Lmod & Module ---
        "Apa perintah untuk mencari modul software berdasarkan keyword?",
        "Apa perintah untuk menonaktifkan semua modul software aktif?",
        "Apa perintah untuk mengganti modul software aktif?",
        "Versi GCC berapa saja yang tersedia di ALELEON?",

        # =============================================================
        # LEVEL 2: Gabungan Info / Multi-Chunk (60 pertanyaan)
        # Butuh menggabungkan info dari beberapa bagian dokumen
        # =============================================================

        # --- Alur Kerja Umum ---
        "Apa saja pilihan cara menjalankan komputasi Python dengan conda env di ALELEON?",
        "Apa perbedaan antara menjalankan batch job via Job Composer EWS dan via terminal Slurm?",
        "Bagaimana langkah lengkap membuat conda env baru dan modul pyload dari awal?",
        "Apa saja status job di squeue dan artinya masing-masing?",
        "Bagaimana cara mengisi formulir Jupyter di EWS untuk conda env user?",
        "Apa saja opsi login yang tersedia untuk mengakses ALELEON?",
        "Bagaimana langkah login pertama kali di ALELEON?",
        "Apa saja menu dan aplikasi yang tersedia di EFIRO Web Service?",
        "Bagaimana cara upload file ke ALELEON?",
        "Apa saja opsi transfer data di ALELEON?",

        # --- Submit Script ---
        "Apa saja parameter SBATCH esensial yang harus diisi dalam submit script?",
        "Bagaimana cara menulis SBATCH time untuk job 2 hari?",
        "Apa perbedaan SBATCH output dan SBATCH error?",
        "Bagaimana cara mengaktifkan notifikasi email untuk status job?",
        "Apa saja pilihan mail-type yang tersedia di SBATCH?",
        "Bagaimana format penulisan SBATCH yang benar?",
        "Apa itu sausage slimit dan bagaimana cara menggunakannya?",
        "Bagaimana cara mengisi template submit script yang ditandai 4 garing?",

        # --- MPI ---
        "Apa perbedaan Pure MPI dan Hybrid MPI/OpenMP?",
        "Bagaimana sintaks menjalankan MPI di ALELEON?",
        "Apa fungsi flag --use-hwthread-cpus pada mpirun?",
        "Bagaimana Slurm ALELEON menyebar proses MPI ke node?",
        "Berapa proses MPI maksimal yang bisa berjalan di 1 node partisi epyc?",
        "Bagaimana cara menjalankan Pure MPI pada core fisik?",
        "Apa itu CUDA-aware MPI dan modul apa yang tersedia?",
        "Bagaimana cara menjalankan MPI GPU dengan proses MPI langsung ke GPU?",

        # --- Software Spesifik ---
        "Bagaimana langkah menjalankan GROMACS di ALELEON via terminal?",
        "Apa perbedaan GROMACS versi CPU dan GPU dalam hal submit script?",
        "Bagaimana cara pre-processing GROMACS di Login Node?",
        "Apa itu binary gmx_mpi dan kenapa bukan gmx?",
        "Bagaimana langkah menjalankan Quantum ESPRESSO via Job Composer?",
        "Bagaimana langkah menjalankan ORCA via terminal Slurm?",
        "Apa perbedaan parameter !PAL dan %PAL NPROCS di ORCA?",
        "Bagaimana langkah menjalankan CP2K versi GPU?",
        "Bagaimana cara menjalankan LAMMPS versi GPU dengan Kokkos?",
        "Bagaimana cara menjalankan OpenMX secara hybrid MPI/OMP?",
        "Apa itu DATA.PATH di OpenMX dan apa nilainya untuk versi 3.9?",
        "Bagaimana langkah menjalankan PHASE/0 via terminal?",
        "Bagaimana langkah menjalankan NAMD versi GPU?",
        "Bagaimana cara menjalankan FLACS-CFD dengan array di ALELEON?",
        "Apa perbedaan menjalankan FLACS-CFD satu simulasi dan array?",
        "Bagaimana langkah menjalankan SIESTA di ALELEON?",

        # --- Monitoring & Troubleshooting ---
        "Bagaimana cara melihat utilisasi CPU dan memori job di Grafana?",
        "Apa saja menu sausage yang tersedia untuk monitoring?",
        "Bagaimana cara membatalkan job yang sedang berjalan?",
        "Apa arti NODELIST REASON 'Resources' pada squeue?",
        "Apa arti NODELIST REASON 'QOSMaxCpuPerUserLimit'?",
        "Apa arti NODELIST REASON 'AssocMaxWallDurationPerJobLimit'?",
        "Bagaimana langkah troubleshooting ketika job tertahan lama?",
        "Bagaimana cara download file output dari ALELEON?",

        # --- Limitasi & Fair Usage ---
        "Berapa limit maksimal CPU untuk akun perseorangan di LFU?",
        "Berapa limit walltime maksimal untuk akun perseorangan?",
        "Berapa limit GPU untuk akun uji coba?",
        "Bagaimana cara mengajukan pembukaan LFU sementara?",
        "Apa itu sausage sfair?",

        # --- Pendaftaran & Akun ---
        "Apa saja golongan pengguna ALELEON?",
        "Apa perbedaan golongan perseorangan dan institusi?",
        "Apa itu PKSPIAS?",
        "Bagaimana alur transaksi top-up kuota core hour?",
        "Apa saja hak pengguna ALELEON selain layanan komputasi?",

        # =============================================================
        # LEVEL 3: Reasoning / Deduksi (60 pertanyaan)
        # Butuh menyimpulkan dari informasi yang tersedia
        # =============================================================

        # --- Python & GPU ---
        "Saya ingin pakai TensorFlow GPU di conda env. Package CUDA versi berapa yang harus saya instal?",
        "Kenapa Anaconda3 2024.06-1 tidak direkomendasikan? Apa yang harus dilakukan user yang sudah terpasang?",
        "Saya upload file Notebook (.ipynb) untuk batch job. Apa yang harus saya lakukan sebelum submit?",
        "Saya ingin menggunakan multi-GPU di ALELEON untuk deep learning. Package apa yang perlu diinstal?",
        "Storage HOME saya hampir penuh setelah banyak instal package conda. Bagaimana cara membersihkannya?",
        "Saya ingin menjalankan PyTorch dengan GPU di conda env. Apa saja yang perlu diperhatikan?",
        "Apa perintah pycheck dan kapan saya harus menggunakannya?",
        "Kenapa submit script Python menggunakan pyl load dan pyl unload?",

        # --- Job Scheduling & Resource ---
        "Job saya status PENDING dengan reason QOSMaxMemoryPerUserLimit. Apa yang harus saya lakukan?",
        "Saya menjalankan 3 job masing-masing 64 core CPU. Kenapa job ke-3 tertahan?",
        "Saya menulis #SBATCH --time = 01:00:00 tapi job tidak jalan. Kenapa?",
        "Saya ingin menjalankan job lebih dari 72 jam. Apakah bisa?",
        "Job saya berhenti tiba-tiba sebelum selesai. Apa yang mungkin terjadi?",
        "Saya menulis #SBATCH --mem= 64 GB tapi job gagal submit. Kenapa?",
        "Bagaimana cara memperkirakan alokasi memori yang tepat untuk job saya?",
        "Apa yang terjadi jika job berjalan melebihi SBATCH time?",
        "Saya akun institusi, user A dan B menjalankan job dan habiskan LFU. Kenapa job saya (user C) tertahan?",
        "Reason MaxCPUPerAccount muncul di job saya. Apa artinya dan bagaimana solusinya?",

        # --- MPI & Paralel ---
        "Saya ingin menjalankan 192 proses MPI di partisi epyc. Berapa SBATCH mem yang harus saya isi jika butuh total 400GB RAM?",
        "Kenapa ALELEON memilih Open MPI dan bukan MPICH?",
        "Saya ingin menjalankan GROMACS dengan 2 GPU. Bagaimana submit scriptnya?",
        "Program saya menggunakan OpenBLAS threading. Bagaimana cara set OMP thread yang benar?",
        "Apa bedanya menjalankan MPI di core thread vs core fisik? Kapan saya pakai yang mana?",
        "Saya menjalankan program hybrid MPI/OMP dengan ntasks=4 dan cpus-per-task=8. Berapa total CPU yang dialokasikan?",
        "Bagaimana cara menjalankan CP2K secara multi-node di ALELEON?",
        "Saya ingin run GROMACS 1 GPU saja. Apa perintahnya berbeda dengan multi-GPU?",

        # --- Software Spesifik ---
        "ORCA saya berjalan single core saja padahal saya minta 32 core. Apa yang mungkin salah?",
        "Saya ingin menjalankan DFTB+ dengan Pure OMP tanpa MPI. Bagaimana SBATCH-nya?",
        "Kenapa GROMACS ALELEON menggunakan gmx_mpi bukan gmx? Apa yang harus saya ubah di script saya?",
        "Saya ingin post-processing Quantum ESPRESSO dengan XCrySDen. Bagaimana caranya?",
        "Bagaimana cara menggunakan BoltzTraP2 setelah kalkulasi Quantum ESPRESSO?",
        "Saya ingin pre-processing GROMACS dengan ACPYPE. Bagaimana langkahnya?",
        "Saya ingin menjalankan FLACS-CFD 8 simulasi sekaligus. Berapa total CPU dan memori jika masing-masing 4 core dan 8GB?",
        "Program R saya gagal karena package belum terinstal. Bagaimana cara mengetahui package mana yang kurang?",
        "Saya ingin menjalankan R di Jupyter. Apa saja yang perlu disiapkan terlebih dahulu?",
        "Bagaimana cara instal library R secara mandiri di ALELEON?",
        "Saya ingin menggunakan OpenMX. Apa yang harus saya definisikan di file input selain parameter komputasi?",

        # --- Infrastruktur & Keamanan ---
        "Saya tidak sengaja menghapus data penting. Apakah bisa di-recovery?",
        "Apa itu SSH Key dan kenapa diperlukan?",
        "Bagaimana cara registrasi SSH Key di ALELEON?",
        "Saya ingin menjalankan software sendiri yang tidak ada di ALELEON. Apa opsi yang tersedia?",
        "Apa itu EasyBuild dan bagaimana cara instal software dengannya?",
        "Apa itu Apptainer dan bagaimana cara menjalankan container di ALELEON?",

        # --- Migrasi & Legacy ---
        "Saya user lama Mk.III. Bagaimana cara migrasi ke Mk.V?",
        "Apakah VPN masih diperlukan untuk ALELEON Mk.V?",

        # --- Deduksi dari Spesifikasi ---
        "Saya ingin menjalankan job yang butuh 400GB RAM. Partisi mana yang harus saya gunakan?",
        "Saya butuh lebih dari 128 core CPU. Apakah ALELEON mendukung multi-node?",
        "Apakah ALELEON mendukung Python 2?",
        "Saya menggunakan software yang butuh GCC 15. Apakah tersedia di ALELEON?",
        "Compiler Fortran apa saja yang tersedia di ALELEON?",
        "Apakah ALELEON mendukung Intel MKL?",
        "Saya butuh library FFTW untuk kompilasi software. Versi apa yang tersedia?",
        "Apakah ALELEON mendukung NVIDIA NCCL untuk multi-GPU training?",

        # --- Layanan ---
        "Saya ingin instalasi software baru di ALELEON. Apakah gratis?",
        "Apa saja persyaratan layanan instalasi software gratis?",
        "Bagaimana cara submit support ticket di EWS?",
        "Apakah tim admin bisa membalas support di luar jam kerja?",

        # =============================================================
        # LEVEL 4: Anti-Hallucination (30 pertanyaan)
        # Jawaban TIDAK ada di dokumen, model harus jujur
        # =============================================================

        "Berapa harga berlangganan conda env di ALELEON per bulan?",
        "Apakah ALELEON mendukung instalasi Docker di dalam conda env?",
        "Berapa jumlah maksimal GPU yang bisa diminta dalam satu batch job conda?",
        "Apakah ALELEON menyediakan layanan cloud storage seperti Google Drive?",
        "Berapa kecepatan internet di ALELEON Supercomputer?",
        "Apakah ALELEON mendukung GPU AMD Radeon?",
        "Berapa biaya maintenance bulanan ALELEON Supercomputer?",
        "Apakah ALELEON mendukung Windows Subsystem for Linux (WSL)?",
        "Berapa jumlah total pengguna aktif ALELEON saat ini?",
        "Apakah ALELEON menyediakan layanan dedicated node untuk user tertentu?",
        "Berapa latency jaringan antar node di ALELEON?",
        "Apakah ALELEON mendukung Kubernetes untuk orchestration container?",
        "Berapa versi CUDA tertinggi yang pernah diinstal di ALELEON?",
        "Apakah ALELEON menyediakan layanan backup otomatis ke cloud?",
        "Berapa bandwidth NVLink antara 2 GPU RTX 3090 di ALELEON?",
        "Apakah user bisa mengubah konfigurasi Slurm sendiri?",
        "Berapa waktu rata-rata antrian job di ALELEON?",
        "Apakah ALELEON support InfiniBand?",
        "Berapa jumlah core fisik per CPU AMD EPYC 7702P?",
        "Apakah ALELEON menyediakan layanan pelatihan HPC untuk pemula?",
        "Berapa uptime SLA yang dijamin oleh EFISON?",
        "Apakah ALELEON mendukung remote desktop via RDP?",
        "Berapa jumlah mahasiswa yang pernah menggunakan EUREKA?",
        "Apakah pengguna bisa mengakses BIOS atau firmware node?",
        "Apakah ALELEON menyediakan API untuk submit job secara programmatic?",
        "Berapa jumlah penelitian yang sedang berjalan di ALELEON saat ini?",
        "Apakah ALELEON mendukung Singularity selain Apptainer?",
        "Berapa total daya listrik yang dikonsumsi ALELEON Supercomputer?",
        "Apakah ada diskon untuk pembelian core hour dalam jumlah besar?",
        "Apakah ALELEON berencana upgrade ke GPU NVIDIA H100?",
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