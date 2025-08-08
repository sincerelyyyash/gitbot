"""
Streaming Utilities

Provides utilities for streaming file processing and memory management to prevent memory leaks.
"""

import asyncio
import gc
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional
from contextlib import asynccontextmanager
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Configuration for streaming operations."""
    chunk_size: int = 8192  # 8KB chunks
    max_memory_mb: int = 100  # 100MB max memory usage
    cleanup_interval: int = 50  # Cleanup every 50 operations
    timeout_seconds: float = 30.0  # Timeout for operations

class MemoryMonitor:
    """Monitor memory usage and trigger cleanup when needed."""
    
    def __init__(self, max_memory_mb: int = 100):
        self.max_memory_mb = max_memory_mb
        self.last_cleanup = time.time()
        self.operation_count = 0
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            return {
                "memory_mb": round(memory_mb, 2),
                "memory_percent": round(process.memory_percent(), 2),
                "max_memory_mb": self.max_memory_mb,
                "needs_cleanup": memory_mb > self.max_memory_mb
            }
        except ImportError:
            logger.warning("psutil not available, cannot monitor memory usage")
            return {"error": "psutil not available"}
    
    def should_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        memory_info = self.check_memory_usage()
        return memory_info.get("needs_cleanup", False) or self.operation_count >= 50
    
    def cleanup(self):
        """Perform memory cleanup."""
        gc.collect()
        self.operation_count = 0
        self.last_cleanup = time.time()
        logger.debug("Memory cleanup performed")

@asynccontextmanager
async def streaming_context(config: StreamConfig = None):
    """Context manager for streaming operations with memory management."""
    if config is None:
        config = StreamConfig()
    
    monitor = MemoryMonitor(config.max_memory_mb)
    
    try:
        yield monitor
    finally:
        monitor.cleanup()

async def stream_file_content(content: bytes, chunk_size: int = 8192) -> AsyncGenerator[bytes, None]:
    """
    Stream file content in chunks to avoid loading entire file into memory.
    
    Args:
        content: File content bytes
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of file content
    """
    for i in range(0, len(content), chunk_size):
        yield content[i:i + chunk_size]
        await asyncio.sleep(0)  # Allow other tasks to run

async def process_large_file_streaming(
    content: bytes, 
    file_path: str, 
    max_size: int = 1024 * 1024,  # 1MB
    chunk_size: int = 8192
) -> str:
    """
    Process large files using streaming to avoid memory issues.
    
    Args:
        content: File content bytes
        file_path: Path to the file
        max_size: Maximum size to process
        chunk_size: Size of chunks for processing
        
    Returns:
        Processed content string
    """
    if len(content) <= max_size:
        # Small file, process normally
        return content.decode('utf-8', errors='ignore')
    
    # Large file, process in chunks
    logger.debug(f"Processing large file {file_path} ({len(content)} bytes) using streaming")
    
    # For large files, we'll process the beginning and end
    start_size = max_size // 2
    end_size = max_size // 2
    
    start_content = content[:start_size].decode('utf-8', errors='ignore')
    end_content = content[-end_size:].decode('utf-8', errors='ignore')
    
    return f"Large file preview (first {start_size} and last {end_size} bytes):\n\nSTART:\n{start_content}\n\n... (content truncated) ...\n\nEND:\n{end_content}"

async def batch_process_items(
    items: List[Any], 
    processor_func, 
    batch_size: int = 100,
    memory_monitor: Optional[MemoryMonitor] = None
) -> List[Any]:
    """
    Process items in batches to manage memory usage.
    
    Args:
        items: List of items to process
        processor_func: Function to process each item
        batch_size: Number of items to process in each batch
        memory_monitor: Optional memory monitor
        
    Returns:
        List of processed items
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch
        batch_results = []
        for item in batch:
            try:
                result = await processor_func(item)
                if result is not None:
                    batch_results.append(result)
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
        
        results.extend(batch_results)
        
        # Check memory usage
        if memory_monitor and memory_monitor.should_cleanup():
            memory_monitor.cleanup()
            await asyncio.sleep(0.01)  # Small delay to prevent blocking
    
    return results

class FileProcessor:
    """File processor with memory management and streaming capabilities."""
    
    def __init__(self, config: StreamConfig = None):
        self.config = config or StreamConfig()
        self.monitor = MemoryMonitor(self.config.max_memory_mb)
    
    async def process_file(self, file_path: str, content: bytes) -> Optional[Dict[str, Any]]:
        """
        Process a single file with memory management.
        
        Args:
            file_path: Path to the file
            content: File content bytes
            
        Returns:
            Processed file data or None if processing failed
        """
        try:
            # Check file size
            if len(content) > self.config.max_memory_mb * 1024 * 1024:
                logger.warning(f"File {file_path} is too large ({len(content)} bytes), skipping")
                return None
            
            # Process content
            processed_content = await process_large_file_streaming(
                content, 
                file_path, 
                max_size=self.config.chunk_size * 100  # 100 chunks max
            )
            
            # Create result
            result = {
                "path": file_path,
                "content": processed_content,
                "size": len(content),
                "processed_at": time.time()
            }
            
            # Update operation count
            self.monitor.operation_count += 1
            
            # Check if cleanup is needed
            if self.monitor.should_cleanup():
                self.monitor.cleanup()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None
    
    async def process_files_batch(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of files with memory management.
        
        Args:
            files: List of file data dictionaries
            
        Returns:
            List of processed file data
        """
        return await batch_process_items(
            files,
            lambda file_data: self.process_file(file_data["path"], file_data["content"]),
            batch_size=50,  # Smaller batch size for file processing
            memory_monitor=self.monitor
        )
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        return self.monitor.check_memory_usage()

# Global file processor instance
file_processor = FileProcessor()
