"""
embedding_api.py — FastAPI Embedding Service

Serves intfloat/multilingual-e5-large embedding model via REST API.
Usage: podman run -d --name embedding-service -p 8001:8001 embedding-service
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import os

# Initialize FastAPI app
app = FastAPI(
    title="Embedding Service",
    description="Serves intfloat/multilingual-e5-large embedding model",
    version="1.0.0"
)

# Load embedding model at startup
MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-large")
print(f"Loading embedding model: {MODEL_NAME}")
embedding_model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded successfully")


class EmbedRequest(BaseModel):
    """Request model for embedding endpoint."""
    texts: List[str]
    normalize: bool = True
    prefix: str = ""
    # Shortcut: "query" → "query: ", "passage"/"document" → "passage: "
    text_type: str = ""


class EmbedResponse(BaseModel):
    """Response model for embedding endpoint."""
    embeddings: List[List[float]]
    model: str
    count: int


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    model: str


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    Generate embeddings for a list of texts.
    
    Args:
        request: EmbedRequest with texts list and normalize flag
        
    Returns:
        EmbedResponse with embeddings list, model name, and count
        
    Example:
        curl -X POST http://localhost:8001/embed \\
          -H "Content-Type: application/json" \\
          -d '{"texts": ["Hello world", "Test embedding"]}'
    """
    try:
        # Determine prefix: explicit prefix takes priority, then text_type shortcut
        prefix = request.prefix
        if not prefix and request.text_type:
            type_map = {
                "query": "query: ",
                "passage": "passage: ",
                "document": "passage: ",
            }
            prefix = type_map.get(request.text_type.lower(), "")

        texts = [f"{prefix}{t}" for t in request.texts] if prefix else request.texts

        embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=request.normalize,
            show_progress_bar=False
        )
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            model=MODEL_NAME,
            count=len(request.texts)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with status and model name
        
    Example:
        curl http://localhost:8001/health
    """
    return HealthResponse(
        status="ok",
        model=MODEL_NAME
    )


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Embedding Service",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "endpoints": {
            "POST /embed": "Generate embeddings for texts",
            "GET /health": "Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)