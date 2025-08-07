# Persistent Vector Database Implementation

## Overview

The gitbot application now includes a robust persistent vector database implementation using ChromaDB. This ensures that repository knowledge bases survive container restarts and provide consistent, performant retrieval-augmented generation (RAG) capabilities.

## Key Features

### 🔄 Persistent Storage
- **Container Restart Resilience**: Vector embeddings survive application restarts
- **Repository Isolation**: Each repository has its own isolated ChromaDB collection
- **Automatic Data Management**: Intelligent collection reuse and refresh capabilities

### 🏗️ Collection Management
- **Automated Collection Naming**: Repository names are automatically sanitized for ChromaDB compatibility
- **Collision Detection**: Hash-based uniqueness for long repository names
- **Collection Lifecycle**: Create, read, update, delete operations for collections

### ⚡ Performance Optimizations
- **Smart Reuse**: Existing collections are reused when possible
- **Incremental Updates**: Only re-index when necessary
- **Memory + Disk**: Hybrid in-memory/persistent storage strategy

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Repository    │    │  RAG Service     │    │   ChromaDB      │
│   Content       │───▶│  Management      │───▶│   Persistent    │
│                 │    │                  │    │   Collections   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌──────────────────┐             │
         │              │  Collection      │             │
         └─────────────▶│  Name            │◀────────────┘
                        │  Sanitization    │
                        └──────────────────┘
```

## Implementation Details

### Collection Naming Strategy

Repository names are automatically converted to valid ChromaDB collection names:

```python
# Examples:
"facebook/react" → "facebook_react"
"microsoft/vscode" → "microsoft_vscode"
"very-long-org/very-long-repo-name" → "very_long_org_very_long_repo_na_a1b2c3d4"
```

**Rules:**
- 3-63 characters (ChromaDB requirement)
- Alphanumeric + underscores only
- Must start with alphanumeric character
- Long names are truncated with hash suffix for uniqueness

### Directory Structure

```
chroma_db/
└── repositories/
    ├── facebook_react/        # Collection for facebook/react
    ├── microsoft_vscode/      # Collection for microsoft/vscode
    └── google_chromium/       # Collection for google/chromium
```

### Configuration

**Environment Variables:**
```bash
CHROMADB_PERSIST_DIRECTORY=./chroma_db  # Default persistent storage location
ADMIN_TOKEN=your_admin_token           # For collection management endpoints
```

**Application Settings:**
```python
# app/config.py
class Settings(BaseSettings):
    chromadb_persist_directory: str = "./chroma_db"
    admin_token: Optional[str] = None
```

## Usage

### Automatic Repository Indexing

When a webhook event is received, the system automatically:

1. **Sanitizes** the repository name for collection naming
2. **Checks** for existing collections
3. **Reuses** existing collections when available
4. **Fetches** repository content if collection is new/empty
5. **Indexes** content into ChromaDB with persistent storage

### Manual Collection Management

**List All Collections:**
```bash
curl "http://localhost:8050/admin/collections?admin_token=your_token"
```

**Get Collection Info:**
```bash
curl "http://localhost:8050/admin/collections/facebook/react?admin_token=your_token"
```

**Refresh Collection:**
```bash
curl -X POST "http://localhost:8050/admin/collections/facebook/react/refresh?installation_id=12345&admin_token=your_token"
```

**Delete Collection:**
```bash
curl -X DELETE "http://localhost:8050/admin/collections/facebook/react?admin_token=your_token"
```

### Programmatic Usage

```python
from app.services import rag_service

# Initialize knowledge base
rag_system = await rag_service.get_or_init_repo_knowledge_base(
    repo_full_name="facebook/react",
    installation_id=12345,
    force_refresh=False
)

# Get collection information
info = await rag_service.get_repository_collection_info("facebook/react")
print(f"Collection has {info['document_count']} documents")

# Refresh knowledge base
success = await rag_service.refresh_repository_knowledge_base("facebook/react", 12345)
```

## Data Persistence

### Storage Location

By default, ChromaDB data is stored in `./chroma_db/repositories/`. This can be configured via:
- Environment variable: `CHROMADB_PERSIST_DIRECTORY`
- Application setting: `settings.chromadb_persist_directory`

### Collection Metadata

Each collection stores:
- **Document Chunks**: Text segments with embeddings
- **Metadata**: Source file paths, content types, timestamps
- **Collection Info**: Document count, creation time
- **Embeddings**: High-dimensional vectors for semantic search

### Volume Mounting (Docker)

For production deployments:

```bash
# Mount persistent volume
docker run -v /host/chroma_data:/app/chroma_db gitbot:latest

# Docker Compose
services:
  gitbot:
    image: gitbot:latest
    volumes:
      - chroma_data:/app/chroma_db
volumes:
  chroma_data:
```

## Performance Characteristics

### Memory Usage
- **In-Memory Cache**: Active collections kept in memory for fast access
- **Lazy Loading**: Collections loaded on-demand
- **Memory Cleanup**: Automatic cleanup of unused collections

### Disk Usage
- **Incremental Growth**: Only new documents are added to existing collections
- **Compression**: ChromaDB handles internal compression
- **Estimated Size**: ~1-5MB per 1000 code files (varies by content)

### Query Performance
- **Sub-second Queries**: Typical query response time < 200ms
- **Concurrent Access**: Multiple repositories can be queried simultaneously
- **Caching**: Frequently accessed embeddings cached in memory

## Troubleshooting

### Common Issues

**Collection Not Found:**
```python
# Force refresh to recreate collection
await refresh_repository_knowledge_base(repo_name, installation_id)
```

**Disk Space Issues:**
```bash
# Check disk usage
du -sh ./chroma_db/

# Clean up unused collections
curl -X DELETE "http://localhost:8050/admin/collections/unused/repo?admin_token=token"
```

**Performance Issues:**
- Check `CHROMADB_PERSIST_DIRECTORY` is on fast storage (SSD)
- Ensure sufficient RAM for in-memory caching
- Monitor collection sizes and clean up if necessary

### Health Monitoring

**Health Check Endpoint:**
```bash
curl http://localhost:8050/health
```

**Collection Statistics:**
```bash
curl "http://localhost:8050/admin/collections?admin_token=token"
```

### Logging

ChromaDB operations are logged at INFO level:
```
INFO:rag_system:ChromaDB persistent client initialized at: /app/chroma_db/repositories
INFO:rag_system:Found existing collection 'facebook_react' with 1245 documents
INFO:rag_system:Using existing collection with 1245 documents
```

## Migration and Backup

### Backup Strategy

```bash
# Simple backup (stop application first)
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz chroma_db/

# Restore backup
tar -xzf chroma_backup_20240704.tar.gz
```

### Version Upgrades

ChromaDB data format is generally forward-compatible. For major version upgrades:

1. **Backup** existing data
2. **Test** with non-production collections
3. **Migrate** gradually by refreshing collections
4. **Verify** query results match expectations

## Security Considerations

### Access Control
- **Admin Token**: Required for collection management endpoints
- **File Permissions**: Ensure ChromaDB directory has appropriate permissions
- **Network Security**: Restrict admin endpoint access in production

### Data Privacy
- **Local Storage**: All embeddings stored locally (no external cloud dependencies)
- **Encryption**: Consider encrypting the ChromaDB directory in production
- **Data Retention**: Implement policies for collection cleanup

## Future Enhancements

### Planned Improvements
- **Automatic Collection Cleanup**: Remove stale collections
- **Advanced Metrics**: Query performance and usage analytics
- **Collection Versioning**: Track changes to repository content
- **Distributed Storage**: Support for multiple ChromaDB instances
- **Advanced Embedding Models**: Support for specialized embedding models

### Configuration Enhancements
- **Collection Size Limits**: Configurable limits per repository
- **Retention Policies**: Automatic cleanup of old collections
- **Performance Tuning**: Configurable chunk sizes and overlap settings

---

This persistent vector database implementation provides a solid foundation for production-grade RAG capabilities in gitbot, ensuring reliable, fast, and scalable retrieval of repository knowledge. 