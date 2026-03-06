#!/usr/bin/env python3
"""
rag_app.py — RAG Application for Podman
=============================================================================
This application orchestrates the RAG chain using embedding-service, vllm-rocm,
and chromadb services. It replaces the local embedding computation with API
calls and uses OpenAI-compatible API for vLLM inference.
=============================================================================
"""

import os
import gc
import requests
from xml.etree import ElementTree
from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from openai import OpenAI
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


def get_embeddings(texts, api_url=None):
    """Get embeddings from embedding service."""
    if api_url is None:
        api_url = os.getenv("EMBEDDING_API_URL", "http://embedding-service:8001")
    
    response = requests.post(
        f"{api_url}/embed",
        json={"texts": texts}
    )
    response.raise_for_status()
    return response.json()["embeddings"]


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
1. Jawab HANYA berdasarkan dokumen referensi. KUTIP langkah-langkah dan perintah PERSIS seperti di dokumen. Jangan menambahkan langkah atau perintah yang tidak ada di dokumen.
2. Sertakan angka, nama, versi, dan spesifikasi PERSIS seperti tertulis di dokumen. Jangan membulatkan atau menambah presisi. Contoh: jika dokumen bilang ">=11", jawab ">=11", BUKAN "11.0" atau "11.2".
3. Jika informasi bisa DISIMPULKAN dari dokumen, berikan kesimpulan tersebut.
4. Jika informasi benar-benar TIDAK ADA di dokumen, katakan "Saya tidak menemukan informasi tersebut di sistem."
5. Jangan mengarang angka, rumus, perintah, URL, atau prosedur yang tidak ada di dokumen.
6. JANGAN mengganti perintah dari dokumen dengan perintah alternatif. Contoh: jika dokumen menulis "source activate", JANGAN ganti dengan "conda activate".
7. Bedakan "minimal" dan "maksimal". Jika dokumen hanya menyebutkan "minimal X" TANPA batas maksimal, jawab bahwa informasi batas maksimal tidak tersedia di dokumen."""
        },
        {
            "role": "user",
            "content": f"""Dokumen Referensi:
{context}

Pertanyaan: {question}"""
        }
    ]

    response = client.chat.completions.create(
        model="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
        messages=messages,
        max_tokens=262144,
        temperature=0.6,
        top_p=0.95,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},  # Non-thinking mode
        }
    )
    
    return response.choices[0].message.content


def create_rag_chain(vectorstore, embedding_api_url=None, llm_api_url=None):
    """Create RAG chain using embedding service and vLLM API."""
    
    def retrieve_and_answer(question):
        # Retrieve relevant documents from ChromaDB
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response using vLLM API
        answer = generate_response(question, context, llm_api_url)
        
        return {
            "answer": answer,
            "context": docs
        }
    
    return retrieve_and_answer


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

    print("[2] Menampilkan isi setiap chunk...")
    # DEBUG: Tampilkan isi setiap chunk
    for i, s in enumerate(splits):
        print(f"\n    [Chunk {i}] ({len(s.page_content)} chars):")
        print(f"    {s.page_content[:120]}...")

    # 2. Get embeddings from embedding service
    print("[3] Mendapatkan embeddings dari embedding-service...")
    texts = [doc.page_content for doc in splits]
    embeddings = get_embeddings(texts)
    
    # DEBUG: Log embeddings type and structure
    print(f"\n    [DEBUG] embeddings type: {type(embeddings)}")
    print(f"    [DEBUG] embeddings length: {len(embeddings) if isinstance(embeddings, list) else 'N/A'}")
    if isinstance(embeddings, list) and len(embeddings) > 0:
        print(f"    [DEBUG] first embedding type: {type(embeddings[0])}")
        print(f"    [DEBUG] first embedding length: {len(embeddings[0]) if isinstance(embeddings[0], list) else 'N/A'}")
        print(f"    [DEBUG] first embedding sample: {embeddings[0][:5] if isinstance(embeddings[0], list) else embeddings[0]}")

    # 3. Simpan ke Vector Database (Chroma)
    print("[4] Menyimpan vektor ke database Chroma...")
    # Create Chroma vectorstore with embeddings
    # FIX: embeddings is a list, not an embedding function. Use EmbeddingFunction wrapper.
    from langchain_core.embeddings import Embeddings
    
    class EmbeddingList(Embeddings):
        """Wrapper for raw embedding list."""
        def __init__(self, embeddings):
            self._embeddings = embeddings
        
        def embed_documents(self, texts):
            return self._embeddings
        
        def embed_query(self, text):
            return self._embeddings[0] if self._embeddings else []
    
    embedding_function = EmbeddingList(embeddings)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)


    # --- FASE 2: SETUP RAG CHAIN ---

    print("\n[5] Membuat RAG chain...")

    # Create RAG chain
    rag_chain = create_rag_chain(
        vectorstore,
        embedding_api_url=os.getenv("EMBEDDING_API_URL", "http://embedding-service:8001"),
        llm_api_url=os.getenv("LLM_API_URL", "http://vllm-rocm:8000/v1")
    )

    # --- FASE 3: TANYA JAWAB (RETRIEVAL & GENERATION) ---

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
    ]

    print("\n[6] Menguji RAG chain dengan semua pertanyaan...")
    for i, pertanyaan in enumerate(pertanyaan_list, 1):
        print(f"\n    [{i}/{len(pertanyaan_list)}] Pertanyaan: {pertanyaan[:60]}...")
        try:
            result = rag_chain(pertanyaan)
            print(f"    → Jawaban: {result['answer'][:100]}...")
        except Exception as e:
            print(f"    → ERROR: {e}")

    print("\n\n=== RAG Process Complete ===")


if __name__ == "__main__":
    main()