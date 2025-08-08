"""
Cache Manager

Provides a comprehensive caching layer for frequently accessed data with:
- Redis support for distributed caching
- In-memory fallback for local caching
- TTL management and automatic expiration
- Cache invalidation strategies
- Performance monitoring and metrics
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Union, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class CacheBackend(Enum):
    """Supported cache backends."""
    MEMORY = "memory"
    REDIS = "redis"

@dataclass
class CacheConfig:
    """Configuration for cache operations."""
    ttl_seconds: int = 300  # 5 minutes default
    max_size: int = 1000  # Maximum number of items
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress items larger than 1KB

@dataclass
class CacheItem:
    """Cache item with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0

class MemoryCache:
    """In-memory cache with LRU eviction and TTL support."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, CacheItem] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        async with self._lock:
            await self._cleanup_expired()
            
            if key in self._cache:
                item = self._cache[key]
                if datetime.utcnow() < item.expires_at:
                    # Update access metadata
                    item.access_count += 1
                    item.last_accessed = datetime.utcnow()
                    
                    # Move to end of access order (LRU)
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._access_order.append(key)
                    
                    self._stats["hits"] += 1
                    return item.value
                else:
                    # Item expired
                    del self._cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._stats["expirations"] += 1
            
            self._stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 300) -> bool:
        """Set item in cache."""
        async with self._lock:
            await self._cleanup_expired()
            
            # Check if we need to evict items
            if len(self._cache) >= self.max_size:
                await self._evict_lru()
            
            # Create cache item
            now = datetime.utcnow()
            expires_at = now + timedelta(seconds=ttl_seconds)
            
            # Estimate size (rough calculation)
            size_bytes = len(str(value).encode('utf-8'))
            
            item = CacheItem(
                key=key,
                value=value,
                created_at=now,
                expires_at=expires_at,
                size_bytes=size_bytes
            )
            
            self._cache[key] = item
            self._access_order.append(key)
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all items from cache."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    async def _cleanup_expired(self) -> None:
        """Remove expired items from cache."""
        now = datetime.utcnow()
        expired_keys = [
            key for key, item in self._cache.items()
            if now >= item.expires_at
        ]
        
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self._stats["expirations"] += 1
    
    async def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                self._stats["evictions"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "expirations": self._stats["expirations"],
            "hit_rate": self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"])
        }

class RedisCache:
    """Redis cache implementation."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis = None
        self._connected = False
        self._stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0
        }
    
    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self.redis_url)
                await self._redis.ping()
                self._connected = True
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._connected = False
                return None
        return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from Redis cache."""
        try:
            redis_client = await self._get_redis()
            if not redis_client:
                return None
            
            data = await redis_client.get(key)
            if data:
                self._stats["hits"] += 1
                return json.loads(data)
            else:
                self._stats["misses"] += 1
                return None
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 300) -> bool:
        """Set item in Redis cache."""
        try:
            redis_client = await self._get_redis()
            if not redis_client:
                return False
            
            data = json.dumps(value)
            await redis_client.setex(key, ttl_seconds, data)
            return True
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete item from Redis cache."""
        try:
            redis_client = await self._get_redis()
            if not redis_client:
                return False
            
            result = await redis_client.delete(key)
            return result > 0
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all items from Redis cache."""
        try:
            redis_client = await self._get_redis()
            if redis_client:
                await redis_client.flushdb()
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Redis clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        return {
            "connected": self._connected,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "errors": self._stats["errors"],
            "hit_rate": self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"])
        }

class CacheManager:
    """Main cache manager with multiple backends and fallback support."""
    
    def __init__(self, backend: CacheBackend = CacheBackend.MEMORY, **kwargs):
        self.backend = backend
        self.config = CacheConfig(**kwargs)
        
        # Initialize cache backends
        self._memory_cache = MemoryCache(max_size=self.config.max_size)
        self._redis_cache = None
        
        if backend == CacheBackend.REDIS:
            redis_url = kwargs.get('redis_url', 'redis://localhost:6379')
            self._redis_cache = RedisCache(redis_url)
        
        # Cache key generators
        self._key_generators: Dict[str, Callable] = {}
        
        # Performance metrics
        self._metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "fallback_usage": 0
        }
    
    def register_key_generator(self, name: str, generator: Callable) -> None:
        """Register a key generator for specific data types."""
        self._key_generators[name] = generator
    
    def generate_key(self, name: str, *args, **kwargs) -> str:
        """Generate a cache key using registered generator."""
        if name in self._key_generators:
            return self._key_generators[name](*args, **kwargs)
        
        # Default key generation
        key_data = f"{name}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache with fallback support."""
        self._metrics["total_requests"] += 1
        
        # Try primary cache first
        if self.backend == CacheBackend.REDIS and self._redis_cache:
            value = await self._redis_cache.get(key)
            if value is not None:
                self._metrics["cache_hits"] += 1
                return value
        
        # Fallback to memory cache
        value = await self._memory_cache.get(key)
        if value is not None:
            self._metrics["cache_hits"] += 1
            return value
        
        self._metrics["cache_misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set item in cache with fallback support."""
        ttl = ttl_seconds or self.config.ttl_seconds
        
        # Set in primary cache
        success = True
        if self.backend == CacheBackend.REDIS and self._redis_cache:
            success = await self._redis_cache.set(key, value, ttl)
        
        # Always set in memory cache as fallback
        await self._memory_cache.set(key, value, ttl)
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete item from all caches."""
        success = True
        
        if self.backend == CacheBackend.REDIS and self._redis_cache:
            success = await self._redis_cache.delete(key)
        
        await self._memory_cache.delete(key)
        
        return success
    
    async def clear(self) -> None:
        """Clear all caches."""
        if self.backend == CacheBackend.REDIS and self._redis_cache:
            await self._redis_cache.clear()
        
        await self._memory_cache.clear()
    
    @asynccontextmanager
    async def cached_operation(self, key: str, ttl_seconds: Optional[int] = None):
        """Context manager for cached operations."""
        # Try to get from cache first
        cached_result = await self.get(key)
        if cached_result is not None:
            yield cached_result, True  # (result, is_cached)
            return
        
        # If not in cache, yield None and let caller set the result
        yield None, False  # (result, is_cached)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self._memory_cache.get_stats()
        redis_stats = self._redis_cache.get_stats() if self._redis_cache else {}
        
        return {
            "backend": self.backend.value,
            "total_requests": self._metrics["total_requests"],
            "cache_hits": self._metrics["cache_hits"],
            "cache_misses": self._metrics["cache_misses"],
            "hit_rate": self._metrics["cache_hits"] / max(1, self._metrics["total_requests"]),
            "memory_cache": memory_stats,
            "redis_cache": redis_stats,
            "config": {
                "ttl_seconds": self.config.ttl_seconds,
                "max_size": self.config.max_size,
                "enable_compression": self.config.enable_compression
            }
        }

# Global cache manager instance
cache_manager = CacheManager()

# Register common key generators
def repo_key_generator(repo_full_name: str, operation: str = "info") -> str:
    """Generate cache key for repository operations."""
    return f"repo:{repo_full_name}:{operation}"

def rag_key_generator(repo_full_name: str, query_hash: str) -> str:
    """Generate cache key for RAG queries."""
    return f"rag:{repo_full_name}:{query_hash}"

def github_key_generator(endpoint: str, params: Dict) -> str:
    """Generate cache key for GitHub API calls."""
    param_str = json.dumps(params, sort_keys=True)
    return f"github:{endpoint}:{hashlib.md5(param_str.encode()).hexdigest()}"

# Register key generators
cache_manager.register_key_generator("repo", repo_key_generator)
cache_manager.register_key_generator("rag", rag_key_generator)
cache_manager.register_key_generator("github", github_key_generator)
