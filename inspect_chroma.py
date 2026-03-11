from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List

CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "wiki_aleleon"


class E5Embeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"query: {text}")


def main():
    print("Loading ChromaDB...")
    embeddings = E5Embeddings(model_name="intfloat/multilingual-e5-large")

    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    collection = vectorstore._collection

    # === 1. Jumlah total chunks ===
    count = collection.count()
    print(f"\n📊 Total chunks tersimpan: {count}")

    # === 2. Ambil semua data (tanpa embedding, biar cepat) ===
    data = collection.get(
        include=["documents", "metadatas"]
    )

    # === 3. Tampilkan ringkasan setiap chunk ===
    print(f"\n{'='*70}")
    for i in range(count):
        doc_id = data["ids"][i]
        text = data["documents"][i]
        meta = data["metadatas"][i]
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
    for meta in data["metadatas"]:
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