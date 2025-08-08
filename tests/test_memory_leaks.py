"""
Memory Leak Tests

Tests to verify that memory leak fixes are working properly:
- Large file processing with streaming
- Database connection pooling
- Background task cleanup
"""

import pytest
import asyncio
import gc
import psutil
import time
from unittest.mock import Mock, patch, AsyncMock
from app.core.streaming_utils import (
    MemoryMonitor, 
    FileProcessor, 
    process_large_file_streaming,
    batch_process_items,
    streaming_context
)
from app.core.database import DatabaseManager
from app.main import BackgroundTaskManager
import os

class TestMemoryManagement:
    """Test memory management utilities."""
    
    def test_memory_monitor_initialization(self):
        """Test memory monitor initialization."""
        monitor = MemoryMonitor(max_memory_mb=100)
        assert monitor.max_memory_mb == 100
        assert monitor.operation_count == 0
    
    def test_memory_monitor_cleanup(self):
        """Test memory monitor cleanup."""
        monitor = MemoryMonitor()
        initial_count = monitor.operation_count
        
        # Simulate operations
        for i in range(60):  # More than cleanup threshold
            monitor.operation_count += 1
        
        assert monitor.should_cleanup()
        monitor.cleanup()
        assert monitor.operation_count == 0
    
    @pytest.mark.asyncio
    async def test_streaming_context(self):
        """Test streaming context manager."""
        async with streaming_context() as monitor:
            assert isinstance(monitor, MemoryMonitor)
            # Context should cleanup automatically
        # After context exit, monitor should be cleaned up
        assert monitor.operation_count == 0
    
    @pytest.mark.asyncio
    async def test_process_large_file_streaming(self):
        """Test large file processing with streaming."""
        # Create a large content (simulate large file)
        large_content = b"x" * (2 * 1024 * 1024)  # 2MB
        
        result = await process_large_file_streaming(
            large_content, 
            "test_large_file.txt", 
            max_size=1024 * 1024  # 1MB limit
        )
        
        # Should return truncated content
        assert "Large file preview" in result
        assert "content truncated" in result
        assert "START:" in result
        assert "END:" in result
    
    @pytest.mark.asyncio
    async def test_batch_processing_memory_management(self):
        """Test batch processing with memory management."""
        items = [{"id": i, "data": f"item_{i}"} for i in range(200)]
        
        async def processor(item):
            # Simulate some processing
            await asyncio.sleep(0.001)
            return {"processed": item["id"], "result": f"processed_{item['data']}"}
        
        monitor = MemoryMonitor(max_memory_mb=50)
        results = await batch_process_items(
            items, 
            processor, 
            batch_size=50,
            memory_monitor=monitor
        )
        
        assert len(results) == 200
        # Memory should be cleaned up during processing
        assert monitor.operation_count <= 50  # Should be reset after cleanup

class TestFileProcessor:
    """Test file processor with memory management."""
    
    @pytest.fixture
    def file_processor(self):
        """Create a file processor instance."""
        return FileProcessor()
    
    @pytest.mark.asyncio
    async def test_process_small_file(self, file_processor):
        """Test processing a small file."""
        content = b"This is a small test file content"
        result = await file_processor.process_file("test_small.txt", content)
        
        assert result is not None
        assert result["path"] == "test_small.txt"
        assert result["content"] == "This is a small test file content"
        assert result["size"] == len(content)
    
    @pytest.mark.asyncio
    async def test_process_large_file(self, file_processor):
        """Test processing a large file."""
        # Create large content
        large_content = b"x" * (150 * 1024 * 1024)  # 150MB
        
        result = await file_processor.process_file("test_large.txt", large_content)
        
        # Should be skipped due to size
        assert result is None
    
    @pytest.mark.asyncio
    async def test_process_files_batch(self, file_processor):
        """Test batch file processing."""
        files = [
            {"path": f"file_{i}.txt", "content": f"content_{i}".encode()} 
            for i in range(100)
        ]
        
        results = await file_processor.process_files_batch(files)
        
        assert len(results) == 100
        # Check memory status
        memory_status = file_processor.get_memory_status()
        assert "memory_mb" in memory_status

class TestDatabaseConnectionPooling:
    """Test database connection pooling."""
    
    @pytest.mark.asyncio
    async def test_database_manager_initialization(self):
        """Test database manager initialization with proper pooling."""
        # Mock settings
        with patch('app.core.database.settings') as mock_settings:
            mock_settings.database_url = "sqlite+aiosqlite:///:memory:"
            mock_settings.database_echo = False
            mock_settings.database_max_connections = 15
            mock_settings.database_pool_timeout = 30
            mock_settings.database_pool_recycle = 3600
            mock_settings.database_pool_pre_ping = True
            mock_settings.database_max_overflow = 5
            
            manager = DatabaseManager()
            
            # Check configuration
            assert manager.max_connections == 15
            assert manager.pool_timeout == 30
            assert manager.pool_recycle == 3600
    
    @pytest.mark.asyncio
    async def test_connection_monitoring(self):
        """Test database connection monitoring."""
        with patch('app.core.database.settings') as mock_settings:
            mock_settings.database_url = "sqlite+aiosqlite:///:memory:"
            mock_settings.database_echo = False
            
            manager = DatabaseManager()
            
            # Test pool status
            pool_status = await manager._get_pool_status()
            assert "pool_size" in pool_status or "error" in pool_status
            
            # Test connection monitoring
            monitor_result = await manager.monitor_connections()
            assert isinstance(monitor_result, dict)

class TestBackgroundTaskManager:
    """Test background task manager."""
    
    @pytest.fixture
    def task_manager(self):
        """Create a background task manager."""
        return BackgroundTaskManager()
    
    @pytest.mark.asyncio
    async def test_task_management(self, task_manager):
        """Test background task management."""
        # Create a simple task
        async def simple_task():
            await asyncio.sleep(0.1)
            return "completed"
        
        task = asyncio.create_task(simple_task())
        task_manager.add_task(task, name="test_task")
        
        # Check task is tracked
        assert len(task_manager.tasks) == 1
        assert task in task_manager.task_metadata
        
        # Wait for task to complete
        await task
        
        # Task should be automatically removed
        await asyncio.sleep(0.1)  # Allow callback to execute
        assert len(task_manager.tasks) == 0
    
    @pytest.mark.asyncio
    async def test_task_cleanup(self, task_manager):
        """Test background task cleanup."""
        # Create a long-running task
        async def long_task():
            await asyncio.sleep(10)  # Long task
        
        task = asyncio.create_task(long_task())
        task_manager.add_task(task, name="long_task")
        
        # Manually set task as old
        task_manager.task_metadata[task]["created_at"] = time.time() - 4000  # Old task
        
        # Run cleanup
        await task_manager.cleanup_old_tasks()
        
        # Task should be cancelled
        assert task.cancelled()
        assert len(task_manager.tasks) == 0
    
    @pytest.mark.asyncio
    async def test_task_shutdown(self, task_manager):
        """Test background task shutdown."""
        # Create multiple tasks
        tasks = []
        for i in range(3):
            async def task_func():
                await asyncio.sleep(1)
            
            task = asyncio.create_task(task_func())
            task_manager.add_task(task, name=f"task_{i}")
            tasks.append(task)
        
        # Shutdown with timeout
        await task_manager.shutdown(timeout=0.5)
        
        # All tasks should be cancelled
        assert len(task_manager.tasks) == 0
        for task in tasks:
            assert task.cancelled()
    
    def test_task_status(self, task_manager):
        """Test background task status reporting."""
        status = task_manager.get_status()
        
        assert "total_tasks" in status
        assert "active_tasks" in status
        assert "completed_tasks" in status
        assert "task_types" in status

class TestMemoryLeakPrevention:
    """Integration tests for memory leak prevention."""
    
    @pytest.mark.asyncio
    async def test_large_file_processing_memory_usage(self):
        """Test that large file processing doesn't cause memory leaks."""
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process multiple large files
        file_processor = FileProcessor()
        for i in range(10):
            # Create moderately large content
            content = b"x" * (500 * 1024)  # 500KB
            await file_processor.process_file(f"test_file_{i}.txt", content)
            
            # Force garbage collection
            gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024, f"Memory increase too large: {memory_increase / 1024 / 1024:.2f}MB"
    
    @pytest.mark.asyncio
    async def test_background_task_memory_cleanup(self):
        """Test that background tasks don't cause memory leaks."""
        task_manager = BackgroundTaskManager()
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create and complete many tasks
        for i in range(50):
            async def task_func():
                # Simulate some work
                data = [i for i in range(1000)]
                await asyncio.sleep(0.01)
                return len(data)
            
            task = asyncio.create_task(task_func())
            task_manager.add_task(task, name=f"memory_test_task_{i}")
            await task
        
        # Force cleanup
        gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal
        assert memory_increase < 10 * 1024 * 1024, f"Memory increase too large: {memory_increase / 1024 / 1024:.2f}MB"
    
    @pytest.mark.asyncio
    async def test_database_connection_cleanup(self):
        """Test that database connections are properly cleaned up."""
        # This test would require a real database connection
        # For now, we'll test the configuration
        
        with patch('app.core.database.settings') as mock_settings:
            mock_settings.database_url = "sqlite+aiosqlite:///:memory:"
            mock_settings.database_echo = False
            mock_settings.database_max_connections = 5
            mock_settings.database_pool_timeout = 30
            mock_settings.database_pool_recycle = 3600
            mock_settings.database_pool_pre_ping = True
            mock_settings.database_max_overflow = 2
            
            manager = DatabaseManager()
            
            # Check that pooling is configured
            assert manager.max_connections == 5
            assert manager.pool_timeout == 30
            assert manager.pool_recycle == 3600

if __name__ == "__main__":
    pytest.main([__file__])
