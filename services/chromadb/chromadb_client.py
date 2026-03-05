"""
chromadb_client.py — ChromaDB Client v2

Provides client functions for interacting with ChromaDB via REST API v2.
Usage: python services/chromadb/chromadb_client.py
"""

import os
import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class CollectionConfig:
    """Configuration for ChromaDB collection."""
    name: str
    embedding_function: str = "default"
    dimension: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ChromaDBClient:
    """Client for ChromaDB REST API v2."""
    
    def __init__(self, base_url: str = "http://chromadb:8000"):
        """
        Initialize ChromaDB client.
        
        Args:
            base_url: Base URL of ChromaDB server
        """
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/v2"
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request to ChromaDB API v2."""
        url = f"{self.api_url}/{endpoint}"
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def heartbeat(self) -> int:
        """
        Check ChromaDB server health.
        
        Returns:
            Heartbeat timestamp in nanoseconds
        """
        return self._request("GET", "heartbeat")["heartbeat"]
    
    def create_collection(self, name: str, embedding_function: str = "default", 
                          dimension: Optional[int] = None, 
                          metadata: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            embedding_function: Embedding function to use (default: "default")
            dimension: Embedding dimension (optional)
            metadata: Collection metadata (optional)
            
        Returns:
            Collection info dict
        """
        data = {
            "name": name,
            "embedding_function": embedding_function
        }
        if dimension:
            data["dimension"] = dimension
        if metadata:
            data["metadata"] = metadata
        
        return self._request("POST", "collections", json=data)
    
    def get_collection(self, name: str) -> Dict:
        """
        Get collection info.
        
        Args:
            name: Collection name
            
        Returns:
            Collection info dict
        """
        return self._request("GET", f"collections/{name}")
    
    def delete_collection(self, name: str) -> None:
        """
        Delete a collection.
        
        Args:
            name: Collection name
        """
        self._request("DELETE", f"collections/{name}")
    
    def list_collections(self) -> List[Dict]:
        """
        List all collections.
        
        Returns:
            List of collection info dicts
        """
        return self._request("GET", "collections")
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Add documents to a collection.
        
        Args:
            collection_name: Target collection name
            documents: List of document texts
            embeddings: Optional list of embedding vectors
            ids: Optional list of document IDs
            metadatas: Optional list of metadata dicts
            
        Returns:
            Operation result dict
        """
        data = {
            "documents": documents
        }
        if embeddings:
            data["embeddings"] = embeddings
        if ids:
            data["ids"] = ids
        if metadatas:
            data["metadatas"] = metadatas
        
        return self._request("POST", f"collections/{collection_name}/add", json=data)
    
    def get_documents(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None
    ) -> Dict:
        """
        Get documents from a collection.
        
        Args:
            collection_name: Target collection name
            ids: Optional list of document IDs to retrieve
            where: Optional filter dict
            limit: Optional limit on results
            offset: Optional offset for pagination
            include: Optional list of fields to include (e.g., ["embeddings", "metadatas"])
            
        Returns:
            Documents dict with ids, documents, embeddings, metadatas
        """
        params = {}
        if ids:
            params["ids"] = ids
        if where:
            params["where"] = where
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if include:
            params["include"] = include
        
        return self._request("GET", f"collections/{collection_name}/get", params=params)
    
    def update_documents(
        self,
        collection_name: str,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Update documents in a collection.
        
        Args:
            collection_name: Target collection name
            ids: List of document IDs to update
            documents: Optional new document texts
            embeddings: Optional new embedding vectors
            metadatas: Optional new metadata dicts
            
        Returns:
            Operation result dict
        """
        data = {"ids": ids}
        if documents:
            data["documents"] = documents
        if embeddings:
            data["embeddings"] = embeddings
        if metadatas:
            data["metadatas"] = metadatas
        
        return self._request("POST", f"collections/{collection_name}/update", json=data)
    
    def delete_documents(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Delete documents from a collection.
        
        Args:
            collection_name: Target collection name
            ids: Optional list of document IDs to delete
            where: Optional filter dict
            
        Returns:
            Operation result dict
        """
        data = {}
        if ids:
            data["ids"] = ids
        if where:
            data["where"] = where
        
        return self._request("POST", f"collections/{collection_name}/delete", json=data)
    
    def query(
        self,
        collection_name: str,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 3,
        where: Optional[Dict] = None,
        include: Optional[List[str]] = None
    ) -> Dict:
        """
        Query collection for similar documents.
        
        Args:
            collection_name: Target collection name
            query_texts: Optional list of query texts (auto-embeds)
            query_embeddings: Optional list of pre-computed query embeddings
            n_results: Number of results to return (default: 3)
            where: Optional filter dict
            include: Optional list of fields to include (e.g., ["embeddings", "metadatas"])
            
        Returns:
            Query result with ids, distances, documents, metadatas
        """
        data = {
            "n_results": n_results
        }
        if query_texts:
            data["query_texts"] = query_texts
        if query_embeddings:
            data["query_embeddings"] = query_embeddings
        if where:
            data["where"] = where
        if include:
            data["include"] = include
        
        return self._request("POST", f"collections/{collection_name}/query", json=data)
    
    def count(self, collection_name: str) -> int:
        """
        Get document count for a collection.
        
        Args:
            collection_name: Target collection name
            
        Returns:
            Document count
        """
        return self._request("GET", f"collections/{collection_name}/count")["count"]


# =============================================================================
# Main — Test Script
# =============================================================================

def main():
    """Test ChromaDB client functionality v2."""
    print("=" * 60)
    print("ChromaDB Client v2 Test")
    print("=" * 60)
    
    # Initialize client
    client = ChromaDBClient(base_url=os.getenv("CHROMADB_URL", "http://chromadb:8000"))
    
    # Test heartbeat
    print("\n[1] Testing heartbeat...")
    try:
        heartbeat = client.heartbeat()
        print(f"    ✓ ChromaDB is running (heartbeat: {heartbeat})")
    except Exception as e:
        print(f"    ✗ Heartbeat failed: {e}")
        return
    
    # Test collection creation
    print("\n[2] Creating collection 'wiki-embeddings'...")
    try:
        result = client.create_collection(
            name="wiki-embeddings",
            embedding_function="default",
            dimension=1024
        )
        print(f"    ✓ Collection created: {result['name']} (dimension: {result['dimension']})")
    except Exception as e:
        print(f"    ✗ Collection creation failed: {e}")
        return
    
    # Test adding documents
    print("\n[3] Adding test documents...")
    try:
        test_docs = [
            "Cara membuat conda environment di ALELEON",
            "Cara menjalankan Jupyter notebook dengan conda env",
            "Cara submit batch job di Slurm"
        ]
        test_embeddings = [
            [0.1] * 1024,  # Dummy embeddings
            [0.2] * 1024,
            [0.3] * 1024
        ]
        result = client.add_documents(
            collection_name="wiki-embeddings",
            documents=test_docs,
            embeddings=test_embeddings,
            ids=["doc1", "doc2", "doc3"]
        )
        print(f"    ✓ Added {result['count']} documents")
    except Exception as e:
        print(f"    ✗ Document addition failed: {e}")
        return
    
    # Test querying
    print("\n[4] Querying for 'conda environment'...")
    try:
        result = client.query(
            collection_name="wiki-embeddings",
            query_texts=["Bagaimana cara membuat conda environment?"],
            n_results=2
        )
        print(f"    ✓ Found {len(result['ids'][0])} results:")
        for i, doc_id in enumerate(result['ids'][0]):
            print(f"      [{i+1}] {doc_id}")
    except Exception as e:
        print(f"    ✗ Query failed: {e}")
        return
    
    # Test counting
    print("\n[5] Counting documents...")
    try:
        count = client.count("wiki-embeddings")
        print(f"    ✓ Collection has {count} documents")
    except Exception as e:
        print(f"    ✗ Count failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()