"""
embedding_api.py — FastAPI Embedding Service (Multi-Mode)

Serves BAAI/bge-m3 embedding model via REST API.
Supports dense, sparse (lexical), and ColBERT (multi-vector) embeddings.

Usage: podman run -d --name embedding-service -p 8001:8001 embedding-service
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Embedding Service",
    description="Serves BAAI/bge-m3 embedding model (dense + sparse + colbert)",
    version="2.0.0"
)

# Load embedding model at startup using FlagEmbedding for multi-mode support
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
print(f"Loading embedding model: {MODEL_NAME}")

# Monkey-patch: FlagEmbedding's reranker module imports is_torch_fx_available
# which was removed in transformers>=4.47. We don't use the reranker, but
# Python's import chain pulls it in at module level.
import transformers.utils.import_utils
if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
    transformers.utils.import_utils.is_torch_fx_available = lambda: False

from FlagEmbedding import BGEM3FlagModel
embedding_model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)
print("Embedding model loaded successfully (FlagEmbedding multi-mode)")


# ── Request / Response Models ──────────────────────────────────────────

class EmbedRequest(BaseModel):
    """Request model for embedding endpoint."""
    texts: List[str]
    normalize: bool = True


class EmbedResponse(BaseModel):
    """Response model for dense-only embedding endpoint."""
    embeddings: List[List[float]]
    model: str
    count: int


class EmbedMultiRequest(BaseModel):
    """Request model for multi-mode embedding endpoint."""
    texts: List[str]
    return_dense: bool = True
    return_sparse: bool = True
    return_colbert: bool = True


class SparseEntry(BaseModel):
    """One sparse vector: parallel lists of token-indices and weights."""
    indices: List[int]
    values: List[float]


class EmbedMultiResponse(BaseModel):
    """Response model for multi-mode embedding endpoint."""
    dense: Optional[List[List[float]]] = None
    sparse: Optional[List[SparseEntry]] = None
    colbert: Optional[List[List[List[float]]]] = None
    model: str
    count: int


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    model: str


# ── Endpoints ──────────────────────────────────────────────────────────

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    Generate **dense-only** embeddings (backward compatible).

    Example:
        curl -X POST http://localhost:8001/embed \\
          -H "Content-Type: application/json" \\
          -d '{"texts": ["Hello world", "Test embedding"]}'
    """
    try:
        output = embedding_model.encode(
            request.texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"]
        if isinstance(dense, np.ndarray):
            dense = dense.tolist()
        return EmbedResponse(
            embeddings=dense,
            model=MODEL_NAME,
            count=len(request.texts),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")


@app.post("/embed/multi", response_model=EmbedMultiResponse)
async def embed_multi(request: EmbedMultiRequest):
    """
    Generate **multi-mode** embeddings (dense + sparse + colbert).

    Returns whichever modes are requested. Sparse vectors are returned as
    parallel ``indices`` / ``values`` lists suitable for Qdrant SparseVector.

    Example:
        curl -X POST http://localhost:8001/embed/multi \\
          -H "Content-Type: application/json" \\
          -d '{"texts": ["Hello world"], "return_dense": true, "return_sparse": true, "return_colbert": true}'
    """
    try:
        output = embedding_model.encode(
            request.texts,
            return_dense=request.return_dense,
            return_sparse=request.return_sparse,
            return_colbert_vecs=request.return_colbert,
        )

        resp: dict = {"model": MODEL_NAME, "count": len(request.texts)}

        # Dense
        if request.return_dense:
            d = output["dense_vecs"]
            resp["dense"] = d.tolist() if isinstance(d, np.ndarray) else d

        # Sparse — convert list-of-dicts → list-of-SparseEntry
        if request.return_sparse:
            sparse_list = []
            for weight_dict in output["lexical_weights"]:
                # weight_dict: {token_id (str|int): weight, ...}
                indices = [int(k) for k in weight_dict.keys()]
                values = [float(v) for v in weight_dict.values()]
                sparse_list.append(SparseEntry(indices=indices, values=values))
            resp["sparse"] = sparse_list

        # ColBERT — list of 2-D arrays (tokens × dim)
        if request.return_colbert:
            colbert_list = []
            for arr in output["colbert_vecs"]:
                if isinstance(arr, np.ndarray):
                    colbert_list.append(arr.tolist())
                else:
                    colbert_list.append(arr)
            resp["colbert"] = colbert_list

        return EmbedMultiResponse(**resp)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", model=MODEL_NAME)


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Embedding Service",
        "version": "2.0.0",
        "model": MODEL_NAME,
        "endpoints": {
            "POST /embed": "Generate dense embeddings (backward compatible)",
            "POST /embed/multi": "Generate dense + sparse + colbert embeddings",
            "GET /health": "Health check",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)