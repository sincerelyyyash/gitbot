# Memory Leak Fixes

This document outlines the memory leak fixes implemented in the gitbot application to ensure stable, long-running operation.

## Overview

The application was experiencing memory leaks in three main areas:
1. **Large file processing** without proper streaming
2. **Database connection pooling** without proper limits
3. **Background task cleanup** that wasn't robust enough

## Fixes Implemented

### 1. Large File Processing with Streaming

#### Problem
- Large files were loaded entirely into memory
- No size limits on file processing
- Memory not cleaned up between file operations

#### Solution
- **Streaming file processing**: Files are processed in chunks to avoid loading entire content into memory
- **Size limits**: Maximum file size of 1MB with configurable limits
- **Memory management**: Automatic garbage collection and memory monitoring
- **Batch processing**: Files are processed in batches with cleanup between batches

#### Implementation
```python
# New streaming utilities in app/core/streaming_utils.py
class MemoryMonitor:
    """Monitor memory usage and trigger cleanup when needed."""
    
class FileProcessor:
    """File processor with memory management and streaming capabilities."""

async def process_large_file_streaming(content: bytes, file_path: str, max_size: int = 1024 * 1024) -> str:
    """Process large files using streaming to avoid memory issues."""
```

#### Key Features
- **Chunked processing**: Files processed in 8KB chunks
- **Memory monitoring**: Real-time memory usage tracking
- **Automatic cleanup**: Garbage collection triggered when memory usage exceeds limits
- **Size limits**: Configurable maximum file sizes (default: 1MB)

### 2. Database Connection Pooling

#### Problem
- No connection pool limits specified
- Connections not properly recycled
- No monitoring of connection usage

#### Solution
- **Connection pool configuration**: Proper limits and timeouts
- **Connection monitoring**: Real-time monitoring of pool status
- **Automatic recycling**: Connections recycled after specified time
- **Health checks**: Pre-ping enabled for connection health

#### Implementation
```python
# Enhanced database configuration in app/config.py
database_max_connections: int = Field(default=20)
database_pool_timeout: int = Field(default=30)
database_pool_recycle: int = Field(default=3600)
database_pool_pre_ping: bool = Field(default=True)
database_max_overflow: int = Field(default=10)
```

#### Key Features
- **Pool size limits**: Maximum 20 connections with 10 overflow
- **Connection timeout**: 30-second timeout for connection acquisition
- **Connection recycling**: Connections recycled every hour
- **Health checks**: Pre-ping enabled to detect stale connections
- **Monitoring**: Real-time pool status monitoring

### 3. Background Task Management

#### Problem
- Background tasks not properly tracked
- No cleanup of old or stuck tasks
- Memory leaks from abandoned tasks

#### Solution
- **Task manager**: Centralized background task management
- **Task tracking**: All tasks tracked with metadata
- **Automatic cleanup**: Old tasks automatically cancelled
- **Graceful shutdown**: Proper shutdown with timeouts

#### Implementation
```python
# Enhanced background task management in app/main.py
class BackgroundTaskManager:
    """Manages background tasks with proper cleanup and monitoring."""
    
    async def cleanup_old_tasks(self):
        """Clean up old or stuck tasks."""
    
    async def shutdown(self, timeout: float = 30.0):
        """Shutdown all background tasks gracefully."""
```

#### Key Features
- **Task tracking**: All tasks tracked with creation time and metadata
- **Automatic cleanup**: Tasks older than 1 hour automatically cancelled
- **Periodic cleanup**: Cleanup runs every 5 minutes
- **Graceful shutdown**: 30-second timeout for task shutdown
- **Memory cleanup**: Periodic garbage collection every 10 minutes

## Configuration

### Environment Variables

Add these to your `.env` file for fine-tuning:

```bash
# Database connection pooling
DATABASE_MAX_CONNECTIONS=20
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600
DATABASE_POOL_PRE_PING=true
DATABASE_MAX_OVERFLOW=10

# Memory management
MAX_FILE_SIZE=1048576  # 1MB in bytes
MAX_CONTENT_LENGTH=50000  # 50KB for content processing
MAX_DOCUMENTS_PER_BATCH=100  # Process documents in batches
```

### Memory Limits

The application now enforces these memory limits:

- **File size limit**: 1MB maximum file size for processing
- **Content length**: 50KB maximum for content processing
- **Batch size**: 100 documents per batch
- **Memory threshold**: 100MB maximum memory usage before cleanup

## Monitoring

### Health Check Endpoint

The `/health` endpoint now includes memory and background task information:

```json
{
  "status": "healthy",
  "background_tasks": {
    "total_tasks": 5,
    "active_tasks": 2,
    "completed_tasks": 3,
    "task_types": ["rag_cleanup", "periodic_cleanup"]
  },
  "memory": {
    "rss_mb": 45.2,
    "vms_mb": 67.8,
    "percent": 2.1
  }
}
```

### Logging

Memory management events are logged:

```
[INFO] Memory cleanup performed
[WARNING] High connection usage detected: 18/20 connections checked out (90.0%)
[INFO] Cleaned up 3 old background tasks
```

## Testing

### Memory Leak Tests

Run the memory leak tests to verify fixes:

```bash
pytest tests/test_memory_leaks.py -v
```

### Test Coverage

The tests cover:
- **Memory monitoring**: Memory usage tracking and cleanup
- **File processing**: Large file handling with streaming
- **Database pooling**: Connection pool management
- **Background tasks**: Task lifecycle and cleanup
- **Integration tests**: End-to-end memory leak prevention

## Performance Impact

### Memory Usage
- **Before**: Unbounded memory growth with large files
- **After**: Controlled memory usage with automatic cleanup

### Processing Speed
- **Before**: Fast but memory-intensive
- **After**: Slightly slower but stable memory usage

### Scalability
- **Before**: Limited by memory constraints
- **After**: Can handle large repositories without memory issues

## Best Practices

### For Developers

1. **Use streaming utilities**: Always use `FileProcessor` for file operations
2. **Monitor memory usage**: Check the health endpoint regularly
3. **Set appropriate limits**: Configure memory limits based on your environment
4. **Test with large files**: Verify memory usage with large repositories

### For Operations

1. **Monitor health endpoint**: Check memory usage and background task status
2. **Set up alerts**: Alert on high memory usage (>80%)
3. **Regular restarts**: Consider periodic restarts for very long-running instances
4. **Resource limits**: Set container memory limits appropriately

## Troubleshooting

### High Memory Usage

If memory usage is high:

1. Check the health endpoint for memory information
2. Look for large files being processed
3. Check background task status
4. Consider reducing batch sizes or file size limits

### Connection Pool Exhaustion

If database connections are exhausted:

1. Check connection pool status in health endpoint
2. Look for long-running database operations
3. Consider increasing pool size or timeout
4. Check for connection leaks in code

### Background Task Issues

If background tasks are stuck:

1. Check background task status in health endpoint
2. Look for tasks older than 1 hour
3. Check task logs for errors
4. Consider restarting the application

## Future Improvements

1. **Redis caching**: Add Redis for caching frequently accessed data
2. **Streaming responses**: Implement streaming responses for large API responses
3. **Memory profiling**: Add detailed memory profiling tools
4. **Automatic scaling**: Implement automatic scaling based on memory usage
