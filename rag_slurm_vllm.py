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
        temperature=0.7,                           
        top_p=0.8,
        top_k=20,
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
        # ===== LEVEL 1: Fakta Langsung =====
        "Bagaimana cara membuat conda environment di aleleon?",
        "bagaimana cara menjalankan jupyter dengan conda environment sendiri?",
        "Versi Python default dari Anaconda3 2025.06-1 apa?",
        "Perintah apa untuk mengaktifkan Mamba 23.11.0-0?",
        "Bagaimana cara membuat modul pyload setelah conda env aktif?",
        "Perintah apa untuk melihat daftar modul pyload yang tersedia?",
        "Di partisi GPU mana batch job conda berjalan?",
        "Apa email support admin ALELEON?",
        "Jam kerja support EFISON kapan?",

        # ===== LEVEL 2: Gabungan Info (Multi-Chunk) =====
        "Apa saja pilihan cara menjalankan komputasi Python dengan conda env di ALELEON?",
        "Apa perbedaan antara menjalankan batch job via Job Composer EWS dan via terminal Slurm?",
        "Bagaimana langkah lengkap membuat conda env baru dan modul pyload dari awal?",
        "Apa saja status job di squeue dan artinya masing-masing?",
        "Bagaimana cara mengisi formulir Jupyter di EWS untuk conda env user?",

        # ===== LEVEL 3: Reasoning / Deduksi =====
        "Saya ingin pakai TensorFlow GPU di conda env. Package CUDA versi berapa yang harus saya instal?",
        "Kenapa Anaconda3 2024.06-1 tidak direkomendasikan? Apa yang harus dilakukan user yang sudah terpasang?",
        "Saya upload file Notebook (.ipynb) untuk batch job. Apa yang harus saya lakukan sebelum submit?",
        "Kenapa submit script menggunakan header #!/bin/bash -l dan perintah pyl load/pyl unload?",
        "Saya ingin menggunakan multi-GPU di ALELEON untuk deep learning. Package apa yang perlu diinstal?",
        "Storage HOME saya hampir penuh setelah banyak instal package conda. Bagaimana cara membersihkannya?",

        # ===== LEVEL 4: Anti-Hallucination (jawaban TIDAK ada di dokumen) =====
        "Berapa harga berlangganan conda env di ALELEON per bulan?",
        "Apakah ALELEON mendukung instalasi Docker di dalam conda env?",
        "Berapa jumlah maksimal GPU yang bisa diminta dalam satu batch job conda?",
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