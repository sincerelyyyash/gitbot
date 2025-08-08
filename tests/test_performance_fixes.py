"""
Performance Fixes Tests

Tests to verify that performance improvements are working properly:
- Caching layer functionality
- Async operations in async contexts
- Payload size limits and validation
- Performance monitoring and metrics
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from app.core.cache_manager import (
    CacheManager, 
    MemoryCache, 
    RedisCache, 
    CacheBackend,
    cache_manager
)
from app.core.async_utils import (
    AsyncExecutor, 
    async_executor, 
    asyncify, 
    processify,
    retry_async,
    timeout_async
)
from app.core.payload_validator import (
    PayloadValidator,
    WebhookPayloadValidator,
    PayloadRateLimiter,
    payload_validator,
    webhook_validator,
    payload_rate_limiter
)
from app.main import app
import psutil

class TestCachingLayer:
    """Test caching layer functionality."""
    
    def test_memory_cache_basic_operations(self):
        """Test basic memory cache operations."""
        cache = MemoryCache(max_size=100)
        
        # Test set and get
        asyncio.run(cache.set("test_key", "test_value", ttl_seconds=60))
        result = asyncio.run(cache.get("test_key"))
        assert result == "test_value"
        
        # Test cache miss
        result = asyncio.run(cache.get("nonexistent_key"))
        assert result is None
        
        # Test deletion
        asyncio.run(cache.delete("test_key"))
        result = asyncio.run(cache.get("test_key"))
        assert result is None
    
    def test_memory_cache_ttl_expiration(self):
        """Test TTL expiration in memory cache."""
        cache = MemoryCache(max_size=100)
        
        # Set item with short TTL
        asyncio.run(cache.set("expire_key", "expire_value", ttl_seconds=1))
        
        # Should be available immediately
        result = asyncio.run(cache.get("expire_key"))
        assert result == "expire_value"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        result = asyncio.run(cache.get("expire_key"))
        assert result is None
    
    def test_memory_cache_lru_eviction(self):
        """Test LRU eviction in memory cache."""
        cache = MemoryCache(max_size=2)
        
        # Add items up to capacity
        asyncio.run(cache.set("key1", "value1"))
        asyncio.run(cache.set("key2", "value2"))
        
        # Both should be available
        assert asyncio.run(cache.get("key1")) == "value1"
        assert asyncio.run(cache.get("key2")) == "value2"
        
        # Add one more item - should evict the least recently used
        asyncio.run(cache.set("key3", "value3"))
        
        # key1 should be evicted (was accessed first)
        assert asyncio.run(cache.get("key1")) is None
        assert asyncio.run(cache.get("key2")) == "value2"
        assert asyncio.run(cache.get("key3")) == "value3"
    
    def test_cache_manager_fallback(self):
        """Test cache manager fallback functionality."""
        # Test with memory backend
        manager = CacheManager(backend=CacheBackend.MEMORY)
        
        # Set and get
        asyncio.run(manager.set("test_key", "test_value"))
        result = asyncio.run(manager.get("test_key"))
        assert result == "test_value"
        
        # Test key generation
        key = manager.generate_key("test_type", param1="value1", param2="value2")
        assert isinstance(key, str)
        assert len(key) > 0
    
    def test_cache_manager_context(self):
        """Test cache manager context manager."""
        manager = CacheManager(backend=CacheBackend.MEMORY)
        
        async def test_context():
            async with manager.cached_operation("test_key") as (result, is_cached):
                if not is_cached:
                    # Simulate expensive operation
                    await asyncio.sleep(0.1)
                    result = "expensive_result"
                    await manager.set("test_key", result)
                return result, is_cached
        
        # First call should not be cached
        result, is_cached = asyncio.run(test_context())
        assert result == "expensive_result"
        assert not is_cached
        
        # Second call should be cached
        result, is_cached = asyncio.run(test_context())
        assert result == "expensive_result"
        assert is_cached

class TestAsyncOperations:
    """Test async operations and utilities."""
    
    def test_async_executor_thread_pool(self):
        """Test async executor thread pool operations."""
        executor = AsyncExecutor()
        
        def sync_function(x, y):
            time.sleep(0.1)  # Simulate work
            return x + y
        
        async def test_async():
            result = await executor.run_in_thread(sync_function, 5, 3)
            return result
        
        result = asyncio.run(test_async())
        assert result == 8
    
    def test_async_executor_batch_processing(self):
        """Test batch processing with async executor."""
        executor = AsyncExecutor()
        
        def process_item(item):
            time.sleep(0.01)  # Simulate work
            return item * 2
        
        async def test_batch():
            items = list(range(10))
            results = await executor.batch_process(
                items, 
                process_item, 
                batch_size=3,
                max_concurrent=2
            )
            return results
        
        results = asyncio.run(test_batch())
        assert len(results) == 10
        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    
    def test_asyncify_decorator(self):
        """Test asyncify decorator."""
        def sync_function(x, y):
            time.sleep(0.1)
            return x * y
        
        async_func = asyncify(sync_function)
        
        async def test_asyncify():
            result = await async_func(5, 3)
            return result
        
        result = asyncio.run(test_asyncify())
        assert result == 15
    
    def test_retry_async(self):
        """Test retry_async utility."""
        call_count = 0
        
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        async def test_retry():
            result = await retry_async(
                failing_function,
                max_retries=3,
                delay=0.1
            )
            return result
        
        result = asyncio.run(test_retry())
        assert result == "success"
        assert call_count == 3
    
    def test_timeout_async(self):
        """Test timeout_async utility."""
        async def slow_function():
            await asyncio.sleep(1.0)
            return "slow_result"
        
        async def test_timeout():
            try:
                result = await timeout_async(slow_function, timeout=0.5)
                return result
            except asyncio.TimeoutError:
                return "timeout"
        
        result = asyncio.run(test_timeout())
        assert result == "timeout"

class TestPayloadValidation:
    """Test payload validation and size limits."""
    
    def test_payload_validator_size_limits(self):
        """Test payload size validation."""
        validator = PayloadValidator()
        
        # Test small payload
        small_payload = {"key": "value"}
        size = validator.estimate_payload_size(small_payload)
        assert size > 0
        assert size < validator.config.max_body_size
        
        # Test large payload
        large_payload = {"key": "x" * 1000000}  # 1MB string
        size = validator.estimate_payload_size(large_payload)
        assert size > 1000000
    
    def test_payload_validator_json_structure(self):
        """Test JSON structure validation."""
        validator = PayloadValidator()
        
        # Valid JSON
        valid_data = {"key": "value", "nested": {"inner": "data"}}
        asyncio.run(validator._validate_json_structure(valid_data))
        
        # Invalid: too deep
        deep_data = {"level1": {"level2": {"level3": {"level4": {"level5": {"level6": {"level7": {"level8": {"level9": {"level10": {"level11": "too_deep"}}}}}}}}}}}
        with pytest.raises(Exception):
            asyncio.run(validator._validate_json_structure(deep_data))
        
        # Invalid: too many keys
        many_keys = {f"key{i}": f"value{i}" for i in range(2000)}
        with pytest.raises(Exception):
            asyncio.run(validator._validate_json_structure(many_keys))
    
    def test_payload_rate_limiter(self):
        """Test payload rate limiter."""
        limiter = PayloadRateLimiter(max_large_payloads=2, time_window=1.0)
        
        # Small payloads should always be allowed
        assert asyncio.run(limiter.check_large_payload_limit(1000)) == True
        
        # Large payloads should be limited
        assert asyncio.run(limiter.check_large_payload_limit(2 * 1024 * 1024)) == True  # First
        assert asyncio.run(limiter.check_large_payload_limit(2 * 1024 * 1024)) == True  # Second
        assert asyncio.run(limiter.check_large_payload_limit(2 * 1024 * 1024)) == False  # Third (rejected)

class TestWebhookPerformance:
    """Test webhook performance improvements."""
    
    def test_webhook_with_caching(self):
        """Test webhook endpoint with caching."""
        client = TestClient(app)
        
        # Mock webhook payload
        payload = {
            "repository": {"full_name": "test/repo"},
            "issue": {"number": 1},
            "comment": {"body": "test question"},
            "installation": {"id": 123}
        }
        
        headers = {
            "X-GitHub-Event": "issue_comment",
            "X-Hub-Signature-256": "sha256=test_signature",
            "content-type": "application/json"
        }
        
        # Mock the signature verification
        with patch('app.api.auth.AuthManager.verify_webhook_signature', return_value=True):
            with patch('app.services.github_event_service.handle_issue_comment') as mock_handler:
                mock_handler.return_value = {"status": "success", "response": "test response"}
                
                # First request
                response1 = client.post("/webhook/webhook", json=payload, headers=headers)
                assert response1.status_code == 200
                
                # Second request should use cache
                response2 = client.post("/webhook/webhook", json=payload, headers=headers)
                assert response2.status_code == 200
    
    def test_webhook_payload_size_validation(self):
        """Test webhook payload size validation."""
        client = TestClient(app)
        
        # Create large payload
        large_payload = {
            "repository": {"full_name": "test/repo"},
            "data": "x" * (10 * 1024 * 1024)  # 10MB string
        }
        
        headers = {
            "X-GitHub-Event": "issue_comment",
            "X-Hub-Signature-256": "sha256=test_signature",
            "content-type": "application/json"
        }
        
        # Should be rejected due to size
        response = client.post("/webhook/webhook", json=large_payload, headers=headers)
        assert response.status_code in [413, 400]  # Payload too large or bad request

class TestPerformanceMonitoring:
    """Test performance monitoring and metrics."""
    
    def test_cache_stats(self):
        """Test cache statistics collection."""
        cache = MemoryCache(max_size=100)
        
        # Perform some operations
        asyncio.run(cache.set("key1", "value1"))
        asyncio.run(cache.get("key1"))  # Hit
        asyncio.run(cache.get("key2"))  # Miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
    
    def test_async_executor_stats(self):
        """Test async executor statistics."""
        executor = AsyncExecutor()
        
        def test_function():
            time.sleep(0.1)
            return "result"
        
        async def test_operations():
            await executor.run_in_thread(test_function)
            await executor.run_in_thread(test_function)
        
        asyncio.run(test_operations())
        
        stats = executor.get_stats()
        assert stats["thread_pool_operations"] == 2
        assert stats["total_duration"] > 0
    
    def test_payload_validator_stats(self):
        """Test payload validator statistics."""
        validator = PayloadValidator()
        
        # Simulate some validations
        validator._validation_stats["total_requests"] = 10
        validator._validation_stats["size_violations"] = 2
        validator._validation_stats["validation_errors"] = 1
        
        stats = validator.get_validation_stats()
        assert stats["total_requests"] == 10
        assert stats["size_violations"] == 2
        assert stats["validation_errors"] == 1
        assert stats["violation_rate"] == 0.3

class TestIntegrationPerformance:
    """Test integration performance improvements."""
    
    def test_health_endpoint_performance(self):
        """Test health endpoint includes performance metrics."""
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "performance" in data
        assert "cache" in data["performance"]
        assert "async_operations" in data["performance"]
        assert "payload_validation" in data["performance"]
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Should be able to get memory info
        assert memory_info.rss > 0
        assert memory_info.vms > 0
        
        # Memory usage should be reasonable (less than 1GB for tests)
        assert memory_info.rss < 1024 * 1024 * 1024

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
