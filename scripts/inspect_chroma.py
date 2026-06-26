import os
import requests
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.embeddings import Embeddings
from typing import List

QDRANT_PERSIST_DIR = "./qdrant_db"
QDRANT_COLLECTION_NAME = "wiki_aleleon"
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8001")


class EmbeddingServiceClient(Embeddings):
    """Memanggil embedding-service REST API (BAAI/bge-m3)."""

    def __init__(self, api_url: str = EMBEDDING_API_URL):
        self.api_url = api_url

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            f"{self.api_url}/embed",
            json={"texts": texts},
            timeout=600,
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call_api(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._call_api([text])[0]


def main():
    print(f"Loading Qdrant dari '{QDRANT_PERSIST_DIR}'...")
    print(f"Embedding API: {EMBEDDING_API_URL}")
    embeddings = EmbeddingServiceClient()

    client = QdrantClient(path=QDRANT_PERSIST_DIR)
    vectorstore = QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION_NAME,
    )

    # === 1. Jumlah total chunks ===
    collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
    count = collection_info.points_count
    print(f"\n📊 Total chunks tersimpan: {count}")

    # === 2. Ambil semua data menggunakan scroll ===
    all_points = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(points)
        if offset is None:
            break

    # === 3. Tampilkan ringkasan setiap chunk ===
    print(f"\n{'='*70}")
    for i, point in enumerate(all_points):
        doc_id = point.id
        meta = point.payload.get("metadata", {})
        text = point.payload.get("page_content", "")
        title = meta.get("title", "N/A")
        source = meta.get("source", "N/A")

        print(f"\n[Chunk {i}] ID: {doc_id}")
        print(f"  📄 Title : {title}")
        print(f"  🔗 Source: {source}")
        print(f"  📏 Length: {len(text)} chars")
        print(f"  📝 Preview: {text}...")
        print("-" * 70)

    # === 4. Statistik per halaman ===
    print(f"\n{'='*70}")
    print("📊 Statistik per halaman:")
    titles = {}
    for point in all_points:
        meta = point.payload.get("metadata", {})
        t = meta.get("title", "N/A")
        titles[t] = titles.get(t, 0) + 1

    for title, chunk_count in sorted(titles.items(), key=lambda x: -x[1]):
        print(f"  {chunk_count:3d} chunks ← {title}")

    # === 5. Test similarity search (opsional) ===
    print(f"\n{'='*70}")
    query = "Bagaimana cara menjalankan GROMACS?"
    print(f"🔍 Test search: \"{query}\"")
    results = vectorstore.similarity_search_with_score(query, k=3)
    for j, (doc, score) in enumerate(results):
        print(f"\n  [{j+1}] Score: {score:.4f}")
        print(f"      Title: {doc.metadata.get('title', 'N/A')}")
        print(f"      Text:  {doc.page_content}...")


if __name__ == "__main__":
    main()