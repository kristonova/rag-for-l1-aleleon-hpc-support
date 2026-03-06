# ChromaDB API Documentation Summary

## Key Findings from Official Documentation

### 1. ChromaDB v2 API Endpoints

According to the official ChromaDB documentation and GitHub issues, ChromaDB v0.4.x+ uses v2 API endpoints:

**v2 Endpoints**:
- `GET /api/v2/heartbeat` - Server heartbeat (public endpoint)
- `GET /api/v2/version` - Server version
- `GET /api/v2/pre-flight-checks` - Pre-flight checks
- `GET /api/v2/healthcheck` - Health check
- `POST /api/v2/reset` - Reset database
- `GET /api/v2/auth/identity` - Get user identity
- `POST /api/v2/tenants` - Create tenant
- `GET /api/v2/collections` - List collections
- `POST /api/v2/collections` - Create collection
- `GET /api/v2/collections/{name}` - Get collection
- `DELETE /api/v2/collections/{name}` - Delete collection
- `POST /api/v2/collections/{name}/add` - Add documents
- `POST /api/v2/collections/{name}/get` - Get documents
- `POST /api/v2/collections/{name}/update` - Update documents
- `POST /api/v2/collections/{name}/upsert` - Upsert documents
- `POST /api/v2/collections/{name}/delete` - Delete documents
- `POST /api/v2/collections/{name}/query` - Query collection
- `POST /api/v2/collections/{name}/count` - Count documents
- `POST /api/v2/collections/{name}/modify` - Modify collection
- `POST /api/v2/collections/{name}/fork` - Fork collection
- `POST /api/v2/collections/{name}/create_index` - Create index
- `POST /api/v2/functions` - Create function
- `POST /api/v2/functions/{name}/attach` - Attach function
- `POST /api/v2/functions/{name}/detach` - Detach function

### 2. Version Compatibility Issues

According to GitHub issue #3073:
> **Bug**: `/api/v2/` routes missing in 0.5.13 docker image
> It breaks basic authentication at the very least. I can confirm that v.0.5.18 docker image has the v2 routes.

This means:
- **ChromaDB < 0.5.18**: Uses v1 API endpoints (`/api/v1/...`)
- **ChromaDB >= 0.5.18**: Uses v2 API endpoints (`/api/v2/...`)

### 3. Authentication Endpoints

For ChromaDB with authentication enabled:

```python
import chromadb

# Connect with authentication
client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=chromadb.config.Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials="your-api-key"
    )
)

# Heartbeat (public endpoint, no auth required)
print(client.heartbeat())

# List collections (protected endpoint, requires auth)
print(client.list_collections())
```

### 4. Environment Variables for ChromaDB

```yaml
environment:
  - CHROMA_SERVER_AUTHN_PROVIDER=chromadb.auth.token_authn.TokenAuthenticationServerProvider
  - CHROMA_SERVER_AUTHN_CREDENTIALS=your-secret-key
  - CHROMA_ADMIN_USER=admin
  - CHROMA_ADMIN_PASSWORD=admin-password
```

### 5. Docker Compose Configuration

```yaml
chromadb:
  image: chromadb/chroma:0.5.18  # Use version >= 0.5.18 for v2 API
  container_name: chromadb
  ports:
    - "8002:8000"
  volumes:
    - chromadb-data:/chroma/data
  environment:
    - CHROMA_SERVER_AUTHN_PROVIDER=chromadb.auth.token_authn.TokenAuthenticationServerProvider
    - CHROMA_SERVER_AUTHN_CREDENTIALS=your-secret-key
  restart: unless-stopped
```

### 6. API Usage Examples

```python
import chromadb

# Connect to ChromaDB server
client = chromadb.HttpClient(
    host="chromadb",
    port=8000,
    settings=chromadb.config.Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials="your-api-key"
    )
)

# Create collection
collection = client.create_collection(name="wiki-embeddings")

# Add documents
collection.add(
    documents=["Document 1", "Document 2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    ids=["id1", "id2"]
)

# Query collection
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=3
)

# Get documents
docs = collection.get(
    ids=["id1"],
    include=["embeddings", "metadatas"]
)
```

### 7. Testing Endpoints

```bash
# Test heartbeat (no auth required)
curl http://localhost:8002/api/v2/heartbeat

# Test version (no auth required)
curl http://localhost:8002/api/v2/version

# Test collections (requires auth)
curl -H "Authorization: Bearer your-api-key" \
  http://localhost:8002/api/v2/collections
```

## Recommendations for Your Project

1. **Use ChromaDB >= 0.5.18**: Ensure you're using a version that supports v2 API endpoints

2. **Update API Endpoints**: Change from `/api/v1/` to `/api/v2/` in your documentation

3. **Authentication**: Configure ChromaDB with token authentication for production use

4. **Test Before Deployment**: Verify the v2 endpoints are available in your ChromaDB version

## References

- [ChromaDB Official Documentation](https://docs.trychroma.com/)
- [ChromaDB Cookbook](https://cookbook.chromadb.dev/)
- [ChromaDB GitHub Repository](https://github.com/chroma-core/chroma)
- [ChromaDB API Layer](https://deepwiki.com/chroma-core/chroma/3-api-layer)