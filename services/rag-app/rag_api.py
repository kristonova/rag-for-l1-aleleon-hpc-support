#!/usr/bin/env python3
"""
rag_api.py — FastAPI wrapper untuk RAG Application
Endpoint: POST /ask  →  { "question": "..." }  →  { "answer": "...", "sources": [...] }
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Import fungsi-fungsi dari rag_app.py yang sudah ada
from rag_app import (
    EmbeddingServiceClient,
    qdrant_collection_exists,
    load_vectorstore,
    build_vectorstore,
    wait_for_vllm,
    create_rag_chain,
)

# ── FastAPI App ──────────────────────────────────────────────
app = FastAPI(
    title="ALELEON HPC RAG API",
    description="REST API untuk RAG L1 Support ALELEON HPC",
    version="1.0.0",
)

# ── Global variables (diisi saat startup) ────────────────────
rag_chain = None


# ── Pydantic Models ──────────────────────────────────────────
class AskRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    title: str
    source_url: str
    section: Optional[str] = None
    justification: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]


# ── Startup Event ────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """
    Saat server mulai:
    1. Koneksi ke embedding service
    2. Load/build vector store dari Qdrant
    3. Tunggu vLLM ready
    4. Buat RAG chain
    """
    global rag_chain

    print("[API] Memulai inisialisasi RAG...")

    # 1. Embedding client
    embeddings = EmbeddingServiceClient()

    # 2. Vector store
    if qdrant_collection_exists():
        vectorstore = load_vectorstore(embeddings)
    else:
        vectorstore = build_vectorstore(embeddings)

    # 3. Tunggu LLM ready
    llm_api_url = os.getenv("LLM_API_URL", "http://vllm-rocm:8000/v1")
    wait_for_vllm(llm_api_url)

    # 4. RAG chain
    rag_chain = create_rag_chain(vectorstore, embeddings, llm_api_url=llm_api_url)

    print("[API] ✅ RAG chain siap menerima pertanyaan!")


# ── Endpoints ────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Kirim pertanyaan ke RAG dan dapatkan jawaban + sumber.

    Contoh request:
        curl -X POST http://localhost:8080/ask \
          -H "Content-Type: application/json" \
          -d '{"question": "Bagaimana cara membuat conda environment?"}'
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain belum siap, coba lagi nanti.")

    try:
        result = rag_chain(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat memproses: {str(e)}")

    # Kumpulkan sumber unik
    sources = []
    seen_keys = []
    justifications = result.get("justifications", [])
    j_idx = 0

    for doc in result.get("context", []):
        title = doc.metadata.get("title", "Unknown")
        source_url = doc.metadata.get("source", "")
        section = doc.metadata.get("Header 2", doc.metadata.get("Header 3", ""))
        key = (title, section)

        if key not in seen_keys:
            seen_keys.append(key)
            sources.append(SourceInfo(
                title=title,
                source_url=source_url,
                section=section or None,
                justification=justifications[j_idx] if j_idx < len(justifications) else None,
            ))
            j_idx += 1

    return AskResponse(answer=result["answer"], sources=sources)


@app.get("/health")
async def health():
    return {
        "status": "ready" if rag_chain is not None else "initializing",
        "service": "rag-api",
    }


@app.get("/")
async def root():
    return {
        "service": "ALELEON HPC RAG API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ask": "Kirim pertanyaan ke RAG",
            "GET /health": "Health check",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
