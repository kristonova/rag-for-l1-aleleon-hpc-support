"""
rag_app.py — RAG Application

Orchestrates RAG chain using embedding service, ChromaDB, and vLLM.
Usage: podman run --rm --network rag-network rag-app
       python services/rag/rag_app.py
"""

import os
import gc
import requests
from typing import List, Dict, Any
from dataclasses import dataclass

# LangChain imports
from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import VLLM
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Local client imports
from services.chromadb.chromadb_client import ChromaDBClient, CollectionConfig


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AppConfig:
    """Application configuration."""
    # API endpoints
    EMBEDDING_API_URL: str = "http://embedding-service:8001"
    LLM_API_URL: str = "http://vllm-rocm:8000/v1"
    CHROMADB_URL: str = "http://chromadb:8000"
    
    # Model names
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    LLM_MODEL: str = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    
    # Collection name
    COLLECTION_NAME: str = "wiki-embeddings"
    
    # Retrieval parameters
    RETRIEVE_K: int = 3
    
    # LLM generation parameters
    MAX_NEW_TOKENS: int = 1024
    TEMPERATURE: float = 0.6
    TOP_P: float = 0.95
    TOP_K: int = 20


# =============================================================================
# Embedding Service Client
# =============================================================================

class EmbeddingClient:
    """Client for embedding service API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
    
    def embed(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of text strings
            normalize: Whether to normalize embeddings
            
        Returns:
            List of embedding vectors
        """
        response = requests.post(
            f"{self.base_url}/embed",
            json={"texts": texts, "normalize": normalize},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embeddings"]
    
    def health(self) -> bool:
        """Check if embedding service is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


# =============================================================================
# RAG Application
# =============================================================================

def load_wiki_documents(sitemap_url: str, requests_per_second: float = 2.0) -> List[Document]:
    """
    Load wiki documents from sitemap.
    
    Args:
        sitemap_url: URL of wiki sitemap XML
        requests_per_second: Rate limit for fetching pages
        
    Returns:
        List of Document objects
    """
    import requests
    from xml.etree import ElementTree
    from bs4 import BeautifulSoup
    import time
    
    print("    Loading wiki documents from sitemap...")
    
    # Parse sitemap
    resp = requests.get(sitemap_url, timeout=30)
    root = ElementTree.fromstring(resp.content)
    ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text for loc in root.findall(".//ns:loc", ns)]
    print(f"    → Found {len(urls)} wiki pages")
    
    # Setup splitters
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4500,
        chunk_overlap=900,
        separators=["\n---", "\n\n", "\n", " "],
    )
    
    all_splits = []
    
    # Fetch and split each page
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
                    sub_splits = text_splitter.split_documents([doc])
                    all_splits.extend(sub_splits)
                else:
                    all_splits.append(doc)
            
            print(f"    [{i+1}/{len(urls)}] {page_title}: {len(html_docs)} sections")
            
        except Exception as e:
            print(f"    [{i+1}/{len(urls)}] ERROR {url}: {e}")
            continue
    
    return all_splits


def enrich_documents(docs: List[Document]) -> List[Document]:
    """
    Enrich documents with source labels.
    
    Args:
        docs: List of Document objects
        
    Returns:
        Enriched list of Document objects
    """
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        header = doc.metadata.get("Header 2", doc.metadata.get("Header 3", ""))
        prefix = f"[Sumber: {title}]"
        if header:
            prefix += f" [Section: {header}]"
        doc.page_content = f"{prefix}\n{doc.page_content}"
    return docs


def create_rag_chain(config: AppConfig) -> Any:
    """
    Create RAG chain with all services.
    
    Args:
        config: Application configuration
        
    Returns:
        RAG chain object
    """
    print("\n[1] Setting up RAG chain...")
    
    # Initialize embedding model (for local use)
    print("    Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    # Initialize ChromaDB client
    print("    Connecting to ChromaDB...")
    chroma_client = ChromaDBClient(base_url=config.CHROMADB_URL)
    
    # Create/retrieve collection
    try:
        collection_info = chroma_client.get_collection(config.COLLECTION_NAME)
        print(f"    → Using existing collection: {config.COLLECTION_NAME}")
    except Exception:
        print(f"    → Creating new collection: {config.COLLECTION_NAME}")
        config_obj = CollectionConfig(
            name=config.COLLECTION_NAME,
            embedding_function="default",
            dimension=1024
        )
        chroma_client.create_collection(config_obj)
    
    # Initialize Chroma vector store
    print("    Initializing Chroma vector store...")
    vectorstore = Chroma(
        client=chroma_client,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME
    )
    
    # Create retriever
    print("    Creating retriever (k={})...".format(config.RETRIEVE_K))
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVE_K})
    
    # Create ChatML prompt
    print("    Creating ChatML prompt...")
    template_qwen = """system
Kamu adalah agen AI asisten admin HPC Slurm yang ahli. Tugasmu adalah membantu user berdasarkan dokumen referensi yang diberikan. Gunakan Bahasa Indonesia yang jelas.

Aturan:
1. Jawab HANYA berdasarkan dokumen referensi. KUTIP langkah-langkah dan perintah PERSIS seperti di dokumen. Jangan menambahkan langkah atau perintah yang tidak ada di dokumen.
2. Sertakan angka, nama, versi, dan spesifikasi PERSIS seperti tertulis di dokumen. Jangan membulatkan atau menambah presisi. Contoh: jika dokumen bilang ">=11", jawab ">=11", BUKAN "11.0" atau "11.2".
3. Jika informasi bisa DISIMPULKAN dari dokumen, berikan kesimpulan tersebut.
4. Jika informasi benar-benar TIDAK ADA di dokumen, katakan "Saya tidak menemukan informasi tersebut di sistem."
5. Jangan mengarang angka, rumus, perintah, URL, atau prosedur yang tidak ada di dokumen.
6. JANGAN mengganti perintah dari dokumen dengan perintah alternatif. Contoh: jika dokumen menulis "source activate", JANGAN ganti dengan "conda activate".
7. Bedakan "minimal" dan "maksimal". Jika dokumen hanya menyebutkan "minimal X" TANPA batas maksimal, jawab bahwa informasi batas maksimal tidak tersedia di dokumen.
</think>
user
Kamu adalah agen AI asisten admin HPC Slurm yang ahli. Tugasmu adalah membantu user berdasarkan dokumen referensi yang diberikan. Gunakan Bahasa Indonesia yang jelas.

Aturan:
1. Jawab HANYA berdasarkan dokumen referensi. KUTIP langkah-langkah dan perintah PERSIS seperti di dokumen. Jangan menambahkan langkah atau perintah yang tidak ada di dokumen.
2. Sertakan angka, nama, versi, dan spesifikasi PERSIS seperti tertulis di dokumen. Jangan membulatkan atau menambah presisi. Contoh: jika dokumen bilang ">=11", jawab ">=11", BUKAN "11.0" atau "11.2".
3. Jika informasi bisa DISIMPULKAN dari dokumen, berikan kesimpulan tersebut.
4. Jika informasi benar-benar TIDAK ADA di dokumen, katakan "Saya tidak menemukan informasi tersebut di sistem."
5. Jangan mengarang angka, rumus, perintah, URL, atau prosedur yang tidak ada di dokumen.
6. JANGAN mengganti perintah dari dokumen dengan perintah alternatif. Contoh: jika dokumen menulis "source activate", JANGAN ganti dengan "conda activate".
7. Bedakan "minimal" dan "maksimal". Jika dokumen hanya menyebutkan "minimal X" TANPA batas maksimal, jawab bahwa informasi batas maksimal tidak tersedia di dokumen.
</think>
user
Dokumen Referensi:
{context}

Pertanyaan: {input}
</think>
assistant
"""
    
    prompt = PromptTemplate(
        template=template_qwen,
        input_variables=["context", "input"]
    )
    
    # Create question-answer chain
    print("    Creating question-answer chain...")
    question_answer_chain = create_stuff_documents_chain(None, prompt)  # LLM will be injected later
    
    # Create RAG chain
    print("    Creating RAG chain...")
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain


def query_rag(rag_chain: Any, question: str) -> Dict[str, Any]:
    """
    Query RAG chain with a question.
    
    Args:
        rag_chain: RAG chain object
        question: User question
        
    Returns:
        Dict with answer and context
    """
    print(f"\nQuery: {question}")
    print("-" * 60)
    
    result = rag_chain.invoke({"input": question})
    
    print(result["answer"].strip())
    
    # Show sources
    if "context" in result and result["context"]:
        print(f"\n📚 Sumber ({len(result['context'])} chunks):")
        seen = []
        for doc in result["context"]:
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
    
    return result


def main():
    """Main RAG application entry point."""
    print("=" * 60)
    print("RAG Application — Retrieval-Augmented Generation")
    print("=" * 60)
    
    # Load configuration from environment
    config = AppConfig(
        EMBEDDING_API_URL=os.getenv("EMBEDDING_API_URL", "http://embedding-service:8001"),
        LLM_API_URL=os.getenv("LLM_API_URL", "http://vllm-rocm:8000/v1"),
        CHROMADB_URL=os.getenv("CHROMADB_URL", "http://chromadb:8000"),
        EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large"),
        LLM_MODEL=os.getenv("LLM_MODEL", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"),
        COLLECTION_NAME=os.getenv("COLLECTION_NAME", "wiki-embeddings"),
        RETRIEVE_K=int(os.getenv("RETRIEVE_K", "3")),
        MAX_NEW_TOKENS=int(os.getenv("MAX_NEW_TOKENS", "1024")),
        TEMPERATURE=float(os.getenv("TEMPERATURE", "0.6")),
        TOP_P=float(os.getenv("TOP_P", "0.95")),
        TOP_K=int(os.getenv("TOP_K", "20")),
    )
    
    # Test service connectivity
    print("\n[0] Testing service connectivity...")
    
    embedding_client = EmbeddingClient(config.EMBEDDING_API_URL)
    if embedding_client.health():
        print("    ✓ Embedding service is healthy")
    else:
        print("    ✗ Embedding service is NOT healthy")
        return
    
    # Create RAG chain
    rag_chain = create_rag_chain(config)
    
    # Test questions
    pertanyaan_list = [
        # Level 1: Direct Facts
        "Bagaimana cara membuat conda environment di aleleon?",
        "bagaimana cara menjalankan jupyter dengan conda environment sendiri?",
        "Versi Python default dari Anaconda3 2025.06-1 apa?",
        "Perintah apa untuk mengaktifkan Mamba 23.11.0-0?",
        "Bagaimana cara membuat modul pyload setelah conda env aktif?",
        "Perintah apa untuk melihat daftar modul pyload yang tersedia?",
        "Di partisi GPU mana batch job conda berjalan?",
        "Apa email support admin ALELEON?",
        "Jam kerja support EFISON kapan?",
        
        # Level 2: Multi-Chunk
        "Apa saja pilihan cara menjalankan komputasi Python dengan conda env di ALELEON?",
        "Apa perbedaan antara menjalankan batch job via Job Composer EWS dan via terminal Slurm?",
        "Bagaimana langkah lengkap membuat conda env baru dan modul pyload dari awal?",
        "Apa saja status job di squeue dan artinya masing-masing?",
        "Bagaimana cara mengisi formulir Jupyter di EWS untuk conda env user?",
        
        # Level 3: Reasoning / Deduction
        "Saya ingin pakai TensorFlow GPU di conda env. Package CUDA versi berapa yang harus saya instal?",
        "Kenapa Anaconda3 2024.06-1 tidak direkomendasikan? Apa yang harus dilakukan user yang sudah terpasang?",
        "Saya upload file Notebook (.ipynb) untuk batch job. Apa yang harus saya lakukan sebelum submit?",
        "Kenapa submit script menggunakan header #!/bin/bash -l dan perintah pyl load/pyl unload?",
        "Saya ingin menggunakan multi-GPU di ALELEON untuk deep learning. Package apa yang perlu diinstal?",
        "Storage HOME saya hampir penuh setelah banyak instal package conda. Bagaimana cara membersihkannya?",
        
        # Level 4: Anti-Hallucination
        "Berapa harga berlangganan conda env di ALELEON per bulan?",
        "Apakah ALELEON mendukung instalasi Docker di dalam conda env?",
        "Berapa jumlah maksimal GPU yang bisa diminta dalam satu batch job conda?",
    ]
    
    # Batch query
    print("\n" + "=" * 60)
    print("Processing {} questions".format(len(pertanyaan_list)))
    print("=" * 60)
    
    for i, q in enumerate(pertanyaan_list, 1):
        query_rag(rag_chain, q)
    
    print("\n" + "=" * 60)
    print(f"Selesai — {len(pertanyaan_list)} pertanyaan dijawab.")
    print("=" * 60)


if __name__ == '__main__':
    main()