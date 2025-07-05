# GitBot Auto-Indexing System

This document describes GitBot's production-level auto-indexing system that automatically creates and maintains knowledge bases for repositories when the GitHub App is installed.

## Overview

The auto-indexing system ensures that repository knowledge bases are ready **before** users start asking questions, providing immediate responses from the moment the app is installed.

### Key Features

- 🚀 **Automatic Indexing**: Starts indexing immediately when app is installed
- 📋 **Queue Management**: Handles multiple repositories with priority-based processing
- 🔄 **Retry Logic**: Automatically retries failed indexing with exponential backoff
- 📊 **Status Tracking**: Real-time monitoring of indexing progress
- 🛡️ **Production Ready**: Circuit breakers, error handling, and resource management
- 💾 **Persistent State**: Queue survives server restarts
- ⚖️ **Load Balancing**: Configurable concurrent workers

## How It Works

### 1. Installation Triggers

The system automatically starts indexing when:

```mermaid
graph TD
    A[GitHub App Installed] --> B{Installation Type}
    B -->|All Repositories| C[Queue All Repos for Indexing]
    B -->|Selected Repositories| D[Queue Selected Repos]
    B -->|Repository Selection| E[Wait for Repository Events]
    
    F[Repository Added Later] --> G[Queue New Repository]
    H[Repository Removed] --> I[Cancel Indexing/Cleanup]
    
    C --> J[Background Workers Process Queue]
    D --> J
    G --> J
```

### 2. Processing Pipeline

```mermaid
graph LR
    A[Repository Added to Queue] --> B[Priority Sorting]
    B --> C[Worker Picks Up Job]
    C --> D[Fetch Repository Content]
    D --> E[Create Embeddings]
    E --> F[Store in ChromaDB]
    F --> G[Mark Complete]
    
    G --> H{Success?}
    H -->|Yes| I[Reset Error Count]
    H -->|No| J[Increment Retry Count]
    J --> K{Max Retries?}
    K -->|No| L[Re-queue with Delay]
    K -->|Yes| M[Mark Failed]
```

### 3. Queue Management

- **Priority-based processing**: Lower numbers = higher priority
- **Deduplication**: Prevents duplicate jobs for same repository
- **Persistent storage**: Queue survives server restarts
- **Automatic cleanup**: Removes old completed/failed jobs

## Configuration

### Environment Variables

```bash
# Required
GITHUB_APP_ID=your_app_id
GITHUB_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----..."
GITHUB_WEBHOOK_SECRET=your_webhook_secret
GEMINI_API_KEY=your_gemini_api_key

# Admin access (recommended)
ADMIN_TOKEN=your_secure_admin_token

# Optional indexing configuration
INDEXING_MAX_WORKERS=3
INDEXING_QUEUE_FILE=data/indexing_queue.json
INDEXING_CLEANUP_DAYS=7

# Storage
DATA_DIR=./data
CHROMADB_PERSIST_DIR=./data/chroma
```

### GitHub App Configuration

Ensure your GitHub App is configured with these webhook events:

```yaml
Webhook Events:
  - installation (created, deleted, suspend, unsuspend)
  - installation_repositories (added, removed)
  - issues (opened)
  - issue_comment (created)
  - push
```

Required permissions:
- **Contents**: Read (to access repository files)
- **Issues**: Write (to post responses)
- **Metadata**: Read (repository information)

## API Endpoints

### Monitoring Endpoints

#### Get Overall Indexing Status
```bash
GET /api/admin/indexing/status?admin_token=your_token
```

Response:
```json
{
  "status": "success",
  "indexing": {
    "total_jobs": 15,
    "pending": 3,
    "in_progress": 2,
    "completed": 8,
    "failed": 2,
    "processing": ["owner/repo1", "owner/repo2"],
    "service_stats": {
      "total_processed": 50,
      "successful": 45,
      "failed": 5,
      "uptime_seconds": 3600
    },
    "jobs": [...]
  }
}
```

#### Get Repository-Specific Status
```bash
GET /api/admin/indexing/status/owner/repo?admin_token=your_token
```

Response:
```json
{
  "status": "success",
  "repository": "owner/repo",
  "indexing": {
    "repo_full_name": "owner/repo",
    "installation_id": 12345,
    "status": "completed",
    "created_at": "2024-01-15T10:30:00Z",
    "started_at": "2024-01-15T10:31:00Z",
    "completed_at": "2024-01-15T10:35:00Z",
    "retry_count": 0,
    "error_message": null,
    "priority": 1
  }
}
```

### Management Endpoints

#### Queue Repository for Indexing
```bash
POST /api/admin/indexing/queue/owner/repo?admin_token=your_token
Content-Type: application/json

{
  "installation_id": 12345,
  "priority": 1,
  "force_refresh": false
}
```

#### Cancel Repository Indexing
```bash
DELETE /api/admin/indexing/cancel/owner/repo?admin_token=your_token
```

## Monitoring and Observability

### Log Messages

The system provides comprehensive logging:

```bash
# Service startup
INFO:indexing_service:Starting indexing service with 3 workers
INFO:indexing_service:Started indexing worker: worker-1

# Job processing
INFO:indexing_service:Added indexing job for owner/repo with priority 0
INFO:indexing_service:worker-1: Processing owner/repo
INFO:indexing_service:Starting indexing for owner/repo
INFO:indexing_service:Successfully indexed owner/repo
INFO:indexing_service:Indexing completed for owner/repo

# Error handling
WARNING:indexing_service:Indexing failed for owner/repo, will retry (1/3): API quota exceeded
ERROR:indexing_service:Indexing failed permanently for owner/repo: Invalid API key
```

### Health Checks

Monitor the indexing system through:

1. **Application Health**: `/health` endpoint
2. **Indexing Status**: `/api/admin/indexing/status`
3. **Log Analysis**: Search for indexing-related log messages
4. **Queue Metrics**: Monitor queue depth and processing times

### Key Metrics to Monitor

```bash
# Queue metrics
- pending_jobs_count
- in_progress_jobs_count
- average_processing_time
- success_rate
- retry_rate

# Resource metrics
- worker_utilization
- api_quota_usage
- disk_space_usage (for ChromaDB)
- memory_usage

# Error metrics
- failed_indexing_rate
- circuit_breaker_openings
- api_error_rate
```

## Production Deployment

### Recommended Configuration

```yaml
# docker-compose.yml
services:
  gitbot:
    image: gitbot:latest
    environment:
      - INDEXING_MAX_WORKERS=5
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - ADMIN_TOKEN=${ADMIN_TOKEN}
    volumes:
      - gitbot_data:/app/data
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

volumes:
  gitbot_data:
```

### Scaling Considerations

#### Vertical Scaling
- **Memory**: 1-2GB for typical workloads, more for large repositories
- **CPU**: 1-2 cores sufficient for most installations
- **Storage**: 10-50MB per repository for embeddings

#### Horizontal Scaling
- Currently single-instance (queue in local file)
- For multi-instance: Replace file queue with Redis/database
- Use shared storage for ChromaDB persistence

### Resource Management

#### API Quota Management
```python
# Automatic quota checking before processing
if not await quota_manager.check_quota(repo_full_name):
    return False, "API quota exceeded"
```

#### Circuit Breaker Protection
```python
# Prevents overwhelming failing repositories
if is_circuit_breaker_open(repo_full_name):
    logger.info(f"Circuit breaker open for {repo_full_name}")
    return
```

## Troubleshooting

### Common Issues

#### 1. Indexing Stuck in Pending
```bash
# Check worker status
curl "http://localhost:8050/api/admin/indexing/status?admin_token=token"

# Possible causes:
- All workers busy with slow repositories
- API quota exceeded
- Configuration errors

# Solutions:
- Increase worker count
- Check API quota
- Restart service
```

#### 2. Repeated Indexing Failures
```bash
# Check specific repository status
curl "http://localhost:8050/api/admin/indexing/status/owner/repo?admin_token=token"

# Common error patterns:
- "API quota exceeded" → Wait or increase quota
- "Invalid API key" → Check GEMINI_API_KEY
- "IP address restriction" → Update API key settings
- "Failed to fetch repository" → Check GitHub permissions

# Force retry with manual queue
curl -X POST "http://localhost:8050/api/admin/indexing/queue/owner/repo?admin_token=token" \
  -H "Content-Type: application/json" \
  -d '{"installation_id": 12345, "priority": 0, "force_refresh": true}'
```

#### 3. High Memory Usage
```bash
# Check ChromaDB size
du -sh data/chroma/

# Solutions:
- Clean up old collections
- Implement collection archival
- Increase available memory
```

#### 4. Slow Indexing Performance
```bash
# Check processing times
grep "Processing" logs/gitbot.log | tail -20

# Optimization strategies:
- Increase worker count (if API quota allows)
- Optimize repository content filtering
- Use faster storage for ChromaDB
```

### Emergency Procedures

#### Stop All Indexing
```bash
# Graceful shutdown
curl -X POST "http://localhost:8050/admin/indexing/stop?admin_token=token"

# Or restart service
docker restart gitbot
```

#### Clear Queue
```bash
# Remove queue file (will lose pending jobs)
rm data/indexing_queue.json

# Or cancel all jobs via API
for repo in $(curl -s "http://localhost:8050/api/admin/indexing/status?admin_token=token" | jq -r '.indexing.jobs[].repo_full_name'); do
  curl -X DELETE "http://localhost:8050/api/admin/indexing/cancel/${repo}?admin_token=token"
done
```

#### Reset Failed Repository
```bash
# Cancel current job and re-queue with high priority
curl -X DELETE "http://localhost:8050/api/admin/indexing/cancel/owner/repo?admin_token=token"
curl -X POST "http://localhost:8050/api/admin/indexing/queue/owner/repo?admin_token=token" \
  -H "Content-Type: application/json" \
  -d '{"installation_id": 12345, "priority": 0, "force_refresh": true}'
```

## Performance Characteristics

### Typical Processing Times
- **Small repository** (< 100 files): 30-60 seconds
- **Medium repository** (100-1000 files): 2-5 minutes  
- **Large repository** (1000+ files): 5-15 minutes

### Resource Usage Per Repository
- **API calls**: 50-200 embedding requests
- **Memory**: 10-50MB during processing
- **Storage**: 5-25MB for embeddings
- **CPU**: Moderate during text processing

### Scaling Limits
- **Single instance**: 50-100 repositories efficiently
- **With optimization**: 200-500 repositories
- **Multi-instance setup**: 1000+ repositories

## Future Enhancements

### Planned Improvements
- **Incremental updates**: Only reindex changed files
- **Smart scheduling**: Index popular repos first
- **Multi-instance support**: Distributed queue management
- **Advanced filtering**: Skip binary/generated files
- **Compression**: Reduce storage requirements
- **Analytics**: Detailed performance metrics

### Integration Possibilities
- **GitHub Actions**: Trigger indexing on major releases
- **Slack/Discord**: Indexing status notifications
- **Prometheus**: Metrics export for monitoring
- **Webhook callbacks**: Notify external systems

This auto-indexing system ensures your GitBot installation provides immediate, intelligent responses to users from the moment it's installed, with production-grade reliability and monitoring capabilities. 