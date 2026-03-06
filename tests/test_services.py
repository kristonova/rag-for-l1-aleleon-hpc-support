"""
test_services.py — Test utilities for embedding, vLLM, and ChromaDB services

Usage:
    python tests/test_services.py                    # Run all tests
    python tests/test_services.py --embedding        # Test only embedding service
    python tests/test_services.py --chromadb         # Test only ChromaDB
    python tests/test_services.py --llm              # Test only vLLM service
    python tests/test_services.py --all              # Test all services
"""

import os
import sys
import argparse
import requests
from typing import Optional


# =============================================================================
# Configuration
# =============================================================================

class ServiceConfig:
    """Service endpoint configuration."""
    EMBEDDING_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8001")
    LLM_URL = os.getenv("LLM_API_URL", "http://0.0.0.0:8000")
    CHROMADB_URL = os.getenv("CHROMADB_URL", "http://localhost:8002")
    # ChromaDB authentication credentials (from compose.yml)
    CHROMADB_AUTH_TOKEN = os.getenv("CHROMADB_AUTH_TOKEN", "your-secret-key")


# =============================================================================
# Test Functions
# =============================================================================

def test_embedding_service() -> bool:
    """Test embedding service health and functionality."""
    print("\n" + "=" * 60)
    print("Testing Embedding Service")
    print("=" * 60)
    
    # Test health endpoint
    print("\n[1] Testing health endpoint...")
    try:
        response = requests.get(f"{ServiceConfig.EMBEDDING_URL}/health", timeout=10)
        if response.status_code == 200:
            print(f"    ✓ Health check passed: {response.json()}")
        else:
            print(f"    ✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Connection error: {e}")
        return False
    
    # Test embedding endpoint
    print("\n[2] Testing embedding endpoint...")
    try:
        test_texts = ["Hello world", "Test embedding service", "Podman container"]
        response = requests.post(
            f"{ServiceConfig.EMBEDDING_URL}/embed",
            json={"texts": test_texts, "normalize": True},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print(f"    ✓ Embedding passed: {result['count']} texts → {len(result['embeddings'][0])} dimensions")
            print(f"    Sample embedding: {result['embeddings'][0][:5]}...")
        else:
            print(f"    ✗ Embedding failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Connection error: {e}")
        return False
    
    # Test root endpoint
    print("\n[3] Testing root endpoint...")
    try:
        response = requests.get(f"{ServiceConfig.EMBEDDING_URL}/", timeout=10)
        if response.status_code == 200:
            print(f"    ✓ Root endpoint passed: {response.json()['service']}")
        else:
            print(f"    ✗ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Connection error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("Embedding Service: ALL TESTS PASSED ✓")
    print("=" * 60)
    return True


def test_chromadb_service() -> bool:
    """Test ChromaDB service health and functionality v2."""
    print("\n" + "=" * 60)
    print("Testing ChromaDB Service v2")
    print("=" * 60)
    
    # Set up authentication header
    headers = {"Authorization": f"Bearer {ServiceConfig.CHROMADB_AUTH_TOKEN}"}

    # Test heartbeat endpoint (public endpoint, no auth required)
    print("\n[1] Testing heartbeat endpoint...")
    try:
        response = requests.get(
            f"{ServiceConfig.CHROMADB_URL}/api/v2/heartbeat",
            timeout=10
        )
        if response.status_code == 200:
            print(f"    ✓ Heartbeat passed: {response.json()['nanosecond heartbeat']}")
        else:
            print(f"    ✗ Heartbeat failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Connection error: {e}")
        return False
    
    # Test version endpoint
    print("\n[2] Testing version endpoint...")
    try:
        response = requests.get(
            f"{ServiceConfig.CHROMADB_URL}/api/v2/version",
            timeout=10
        )
        if response.status_code == 200:
            print(f"    ✓ Version passed: {response.json()}")
        else:
            print(f"    ✗ Version failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Connection error: {e}")
        return False
    
    # Test healthcheck endpoint
    print("\n[3] Testing healthcheck endpoint...")
    try:
        response = requests.get(
            f"{ServiceConfig.CHROMADB_URL}/api/v2/healthcheck",
            timeout=10
        )
        if response.status_code == 200:
            print(f"    ✓ Healthcheck passed: {response.json()}")
        else:
            print(f"    ✗ Healthcheck failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Connection error: {e}")
        return False
    
    # Test pre-flight checks endpoint
    print("\n[4] Testing pre-flight checks endpoint...")
    try:
        response = requests.get(
            f"{ServiceConfig.CHROMADB_URL}/api/v2/pre-flight-checks",
            timeout=10
        )
        if response.status_code == 200:
            print(f"    ✓ Pre-flight checks passed: {response.json()}")
        else:
            print(f"    ✗ Pre-flight checks failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Connection error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("ChromaDB Service v2: ALL TESTS PASSED ✓")
    print("=" * 60)
    return True


def test_llm_service() -> bool:
    """Test vLLM LLM service health."""
    print("\n" + "=" * 60)
    print("Testing vLLM LLM Service")
    print("=" * 60)
    
    # Test health endpoint
    print("\n[1] Testing health endpoint...")
    try:
        response = requests.get(f"{ServiceConfig.LLM_URL}/version", timeout=10)
        if response.status_code == 200:
            print(f"    ✓ Health check passed: {response.json()}")
        else:
            print(f"    ✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Connection error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("vLLM LLM Service: ALL TESTS PASSED ✓")
    print("=" * 60)
    return True


def main():
    """Run all service tests."""
    parser = argparse.ArgumentParser(description="Test RAG services")
    parser.add_argument("--embedding", action="store_true", help="Test only embedding service")
    parser.add_argument("--chromadb", action="store_true", help="Test only ChromaDB service")
    parser.add_argument("--llm", action="store_true", help="Test only vLLM service")
    parser.add_argument("--all", action="store_true", help="Test all services")
    args = parser.parse_args()
    
    # Default to all tests if no specific flag is set
    if not (args.embedding or args.chromadb or args.llm or args.all):
        args.all = True
    
    results = []
    
    if args.all or args.embedding:
        results.append(("Embedding", test_embedding_service()))
    
    if args.all or args.chromadb:
        results.append(("ChromaDB", test_chromadb_service()))
    
    if args.all or args.llm:
        results.append(("vLLM", test_llm_service()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for service, passed in results:
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"  {service}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL SERVICES: HEALTHY ✓")
    else:
        print("SOME SERVICES: UNHEALTHY ✗")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())