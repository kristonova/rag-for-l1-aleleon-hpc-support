#!/usr/bin/env python3
"""
rag_api.py — FastAPI wrapper untuk RAG Application
Endpoint: POST /ask  →  { "question": "..." }  →  { "answer": "...", "sources": [...] }
Endpoint: POST /review-script  →  { "script": "..." }  →  { "review": "...", "issues_found": N }
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
    review_script_hybrid,
)

# ── FastAPI App ──────────────────────────────────────────────
app = FastAPI(
    title="ALELEON HPC RAG API",
    description="REST API untuk RAG L1 Support ALELEON HPC",
    version="1.2.0",
)

# ── Global variables (diisi saat startup) ────────────────────
rag_chain = None
llm_api_url = None
qdrant_client_global = None
embeddings_global = None


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


class ReviewScriptRequest(BaseModel):
    script: str


class ReviewScriptResponse(BaseModel):
    review: str
    issues_found: int
    policy_sources: Optional[List[SourceInfo]] = None


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
    global rag_chain, llm_api_url, qdrant_client_global, embeddings_global

    print("[API] Memulai inisialisasi RAG...")

    # 1. Embedding client
    embeddings = EmbeddingServiceClient()
    embeddings_global = embeddings  # Simpan untuk hybrid script review

    # 2. Vector store
    if qdrant_collection_exists():
        vectorstore = load_vectorstore(embeddings)
    else:
        vectorstore = build_vectorstore(embeddings)
    qdrant_client_global = vectorstore  # Simpan untuk hybrid script review

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


@app.post("/review-script", response_model=ReviewScriptResponse)
async def review_script_endpoint(req: ReviewScriptRequest):
    """
    Review skrip Bash/Slurm dengan pendekatan HYBRID:
    - LLM analisis teknis skrip
    - Jika ada resource params → retrieval kebijakan HPC untuk validasi

    Contoh request:
        curl -X POST http://localhost:8080/review-script \
          -H "Content-Type: application/json" \
          -d '{"script": "#!/bin/bash\\n#SBATCH --mem= 64 GB\\nsrun gmx_mpi mdrun"}'
    """
    if llm_api_url is None:
        raise HTTPException(status_code=503, detail="LLM belum siap, coba lagi nanti.")

    try:
        result = review_script_hybrid(
            req.script,
            api_url=llm_api_url,
            qdrant_client=qdrant_client_global,
            embeddings=embeddings_global,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat review skrip: {str(e)}")

    # Build policy_sources for response
    policy_sources = None
    if result.get("policy_sources"):
        policy_sources = [
            SourceInfo(
                title=s["title"],
                source_url=s["source_url"],
                section=s.get("section"),
            )
            for s in result["policy_sources"]
        ]

    return ReviewScriptResponse(
        review=result["review"],
        issues_found=result["issues_found"],
        policy_sources=policy_sources,
    )


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
        "version": "1.2.0",
        "endpoints": {
            "POST /ask": "Kirim pertanyaan ke RAG (retrieval + LLM)",
            "POST /review-script": "Review skrip Bash/Slurm (hybrid: LLM teknis + RAG kebijakan HPC)",
            "GET /health": "Health check",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
