# rag_slurm_vllm.py Modification Requirements

## Overview

This document outlines the required modifications to [`rag_slurm_vllm.py`](rag_slurm_vllm.py) to integrate with the embedding service and use OpenAI API-compatible mode with non-thinking (instruct) mode for Qwen3.5.

---

## Required Modifications

### 1. Replace Local Embedding with Embedding Service

**Current Implementation** (Line 118-124):
```python
# 2. Setup Model Embedding (Lokal via HuggingFace - CPU/GPU)
print("[3] Load model embedding lokal...")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# 4. Simpan ke Vector Database (Chroma)
print("[4] Menyimpan vektor ke database Chroma...")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
```

**Required Change**:
- Remove `HuggingFaceEmbeddings` import
- Create API client to call `embedding-service` POST `/embed` endpoint
- Use returned embeddings for ChromaDB

**Example Implementation**:
```python
import requests

def get_embeddings(texts, api_url="http://embedding-service:8001"):
    """Get embeddings from embedding service."""
    response = requests.post(
        f"{api_url}/embed",
        json={"texts": texts}
    )
    response.raise_for_status()
    return response.json()["embeddings"]

# Usage
texts = [doc.page_content for doc in splits]
embeddings = get_embeddings(texts)
vectorstore = Chroma.from_documents(documents=splits, embedding=None)
# Manually set embeddings
vectorstore.add_embeddings(texts=texts, embeddings=embeddings, ids=[...])
```

---

### 2. Replace VLLM LLM with OpenAI API-Compatible Client

**Current Implementation** (Line 127-148):
```python
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
```

**Required Change**:
- Remove `langchain_community.llms.VLLM` import
- Use `openai` client to call vLLM API
- Configure `enable_thinking=False` for non-thinking (instruct) mode

**Example Implementation**:
```python
from openai import OpenAI

# Configure OpenAI client to use vLLM endpoint
client = OpenAI(
    base_url="http://vllm-rocm:8000/v1",  # Adjust URL as needed
    api_key="your-api-key"  # If authentication is required
)

def generate_response(question, context):
    """Generate response using vLLM with non-thinking mode."""
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
        max_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},  # Non-thinking mode
        }
    )
    
    return response.choices[0].message.content
```

---

### 3. Update RAG Chain to Use API Clients

**Current Implementation** (Line 151-187):
```python
# --- FASE 3: TANYA JAWAB (RETRIEVAL & GENERATION) ---

# Setup Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Buat Prompt dengan format ChatML (untuk Qwen)
template_qwen = """..."""

prompt = PromptTemplate(
    template=template_qwen,
    input_variables=["context", "input"]
)

# Rangkai rantai RAG (Chain)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

**Required Change**:
- Remove `langchain_classic` imports
- Create custom RAG chain using API clients
- Maintain same functionality with service-based approach

**Example Implementation**:
```python
from langchain_chroma import Chroma

def create_rag_chain(vectorstore, llm_api_url="http://vllm-rocm:8000/v1"):
    """Create RAG chain using embedding service and vLLM API."""
    
    def retrieve_and_answer(question):
        # Retrieve relevant documents
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response using vLLM API
        response = generate_response(question, context)
        
        return {
            "answer": response,
            "context": docs
        }
    
    return retrieve_and_answer

# Usage
rag_chain = create_rag_chain(vectorstore)
```

---

## Complete Modified Structure

```python
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

# ... (load_wiki_documents function remains the same) ...

def get_embeddings(texts, api_url="http://embedding-service:8001"):
    """Get embeddings from embedding service."""
    response = requests.post(
        f"{api_url}/embed",
        json={"texts": texts}
    )
    response.raise_for_status()
    return response.json()["embeddings"]

def generate_response(question, context, api_url="http://vllm-rocm:8000/v1"):
    """Generate response using vLLM with non-thinking mode."""
    client = OpenAI(base_url=api_url)
    
    messages = [
        {
            "role": "system",
            "content": """Kamu adalah agen AI asisten admin HPC Slurm yang ahli..."""
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
        max_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    
    return response.choices[0].message.content

def main():
    # ... (FASE 1: Load documents remains the same) ...
    
    # Get embeddings from service
    texts = [doc.page_content for doc in splits]
    embeddings = get_embeddings(texts)
    
    # Create vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=None)
    vectorstore.add_embeddings(texts=texts, embeddings=embeddings, ids=[...])
    
    # ... (FASE 2 & 3: Use API clients for generation) ...
    
    # Batch invoke
    pertanyaan_list = [...]
    inputs = [{"input": q} for q in pertanyaan_list]
    
    for inp in inputs:
        result = rag_chain(inp)
        print(result['answer'].strip())
```

---

## Benefits of This Approach

1. **Service-Based Architecture**: Embeddings computed by dedicated service
2. **API Compatibility**: Uses standard OpenAI-compatible API
3. **Non-Thinking Mode**: Qwen3.5 responds directly without thinking process
4. **Modularity**: Easy to update embedding service or vLLM independently
5. **Scalability**: Services can be scaled independently

---

## Testing Checklist

- [ ] Embedding service returns correct embeddings
- [ ] vLLM API accepts requests with `enable_thinking=False`
- [ ] RAG chain retrieves correct documents
- [ ] Responses are accurate and don't hallucinate
- [ ] All services communicate correctly via network

---

## References

- **Qwen3.5 Documentation**: [Qwen3.5 API Guide](https://qwen.readthedocs.io/)
- **OpenAI Python Client**: [OpenAI Python Docs](https://platform.openai.com/docs/guides/text-generation)
- **Embedding Service**: [`services/embedding/embedding_api.py`](services/embedding/embedding_api.py)