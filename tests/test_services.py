"""
test_services.py — Test utilities for embedding, vLLM, and Qdrant services

Usage:
    python tests/test_services.py                    # Run all tests
    python tests/test_services.py --embedding        # Test only embedding service
    python tests/test_services.py --qdrant          # Test only Qdrant
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
    LLM_URL = os.getenv("LLM_API_URL", "http://localhost:8000")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "your-secret-key")


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


def test_qdrant_service() -> bool:
    """Test Qdrant service health and functionality."""
    print("\n" + "=" * 60)
    print("Testing Qdrant Service")
    print("=" * 60)
    
    # Set up authentication header
    headers = {"api-key": ServiceConfig.QDRANT_API_KEY}

    # Test healthz endpoint
    print("\n[1] Testing healthz endpoint...")
    try:
        response = requests.get(
            f"{ServiceConfig.QDRANT_URL}/healthz",
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            print(f"    ✓ Healthz passed")
        else:
            print(f"    ✗ Healthz failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Connection error: {e}")
        return False
    
    # Test collections endpoint
    print("\n[2] Testing collections endpoint...")
    try:
        response = requests.get(
            f"{ServiceConfig.QDRANT_URL}/collections",
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            collections = result.get("result", {}).get("collections", [])
            print(f"    ✓ Collections passed: {len(collections)} collections found")
            for col in collections:
                print(f"       - {col['name']}")
        else:
            print(f"    ✗ Collections failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Connection error: {e}")
        return False
    
    # Test cluster info
    print("\n[3] Testing cluster info endpoint...")
    try:
        response = requests.get(
            f"{ServiceConfig.QDRANT_URL}/cluster",
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            print(f"    ✓ Cluster info passed: {response.json().get('result', {}).get('status', 'unknown')}")
        else:
            print(f"    ✗ Cluster info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Connection error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("Qdrant Service: ALL TESTS PASSED ✓")
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
    parser.add_argument("--qdrant", action="store_true", help="Test only Qdrant service")
    parser.add_argument("--llm", action="store_true", help="Test only vLLM service")
    parser.add_argument("--all", action="store_true", help="Test all services")
    args = parser.parse_args()
    
    # Default to all tests if no specific flag is set
    if not (args.embedding or args.qdrant or args.llm or args.all):
        args.all = True
    
    results = []
    
    if args.all or args.embedding:
        results.append(("Embedding", test_embedding_service()))
    
    if args.all or args.qdrant:
        results.append(("Qdrant", test_qdrant_service()))
    
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