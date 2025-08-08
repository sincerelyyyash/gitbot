"""
Async Utilities

Provides utilities for handling synchronous operations in async contexts:
- Thread pool management for CPU-bound operations
- Async wrappers for synchronous libraries
- Performance monitoring and optimization
- Batch processing with concurrency control
"""

import asyncio
import logging
import time
import functools
from typing import Any, Callable, List, Dict, Optional, TypeVar, Awaitable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
import threading
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class AsyncConfig:
    """Configuration for async operations."""
    max_workers: int = 10
    max_process_workers: int = 4
    timeout_seconds: float = 30.0
    enable_monitoring: bool = True
    batch_size: int = 100
    max_concurrent_batches: int = 5

class AsyncExecutor:
    """Manages thread and process pools for async operations."""
    
    def __init__(self, config: AsyncConfig = None):
        self.config = config or AsyncConfig()
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._lock = threading.Lock()
        self._stats = {
            "thread_pool_operations": 0,
            "process_pool_operations": 0,
            "timeouts": 0,
            "errors": 0,
            "total_duration": 0.0
        }
    
    @property
    def thread_pool(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor."""
        if self._thread_pool is None:
            with self._lock:
                if self._thread_pool is None:
                    self._thread_pool = ThreadPoolExecutor(
                        max_workers=self.config.max_workers,
                        thread_name_prefix="gitbot-async"
                    )
        return self._thread_pool
    
    @property
    def process_pool(self) -> ProcessPoolExecutor:
        """Get or create process pool executor."""
        if self._process_pool is None:
            with self._lock:
                if self._process_pool is None:
                    self._process_pool = ProcessPoolExecutor(
                        max_workers=self.config.max_process_workers
                    )
        return self._process_pool
    
    async def run_in_thread(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Run a synchronous function in a thread pool."""
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.thread_pool,
                    functools.partial(func, *args, **kwargs)
                ),
                timeout=self.config.timeout_seconds
            )
            
            duration = time.time() - start_time
            self._stats["thread_pool_operations"] += 1
            self._stats["total_duration"] += duration
            
            if self.config.enable_monitoring and duration > 1.0:
                logger.warning(f"Slow thread operation: {func.__name__} took {duration:.2f}s")
            
            return result
            
        except asyncio.TimeoutError:
            self._stats["timeouts"] += 1
            logger.error(f"Thread operation timed out: {func.__name__}")
            raise
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Thread operation failed: {func.__name__} - {e}")
            raise
    
    async def run_in_process(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Run a synchronous function in a process pool."""
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.process_pool,
                    functools.partial(func, *args, **kwargs)
                ),
                timeout=self.config.timeout_seconds
            )
            
            duration = time.time() - start_time
            self._stats["process_pool_operations"] += 1
            self._stats["total_duration"] += duration
            
            if self.config.enable_monitoring and duration > 2.0:
                logger.warning(f"Slow process operation: {func.__name__} took {duration:.2f}s")
            
            return result
            
        except asyncio.TimeoutError:
            self._stats["timeouts"] += 1
            logger.error(f"Process operation timed out: {func.__name__}")
            raise
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Process operation failed: {func.__name__} - {e}")
            raise
    
    async def batch_process(
        self,
        items: List[Any],
        processor: Callable[[Any], T],
        batch_size: Optional[int] = None,
        max_concurrent: Optional[int] = None,
        use_process_pool: bool = False
    ) -> List[T]:
        """Process items in batches with concurrency control."""
        batch_size = batch_size or self.config.batch_size
        max_concurrent = max_concurrent or self.config.max_concurrent_batches
        
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch: List[Any]) -> List[T]:
            async with semaphore:
                tasks = []
                for item in batch:
                    if use_process_pool:
                        task = self.run_in_process(processor, item)
                    else:
                        task = self.run_in_thread(processor, item)
                    tasks.append(task)
                
                return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Split items into batches
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]
        
        # Process batches concurrently
        batch_tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing failed: {batch_result}")
                continue
            
            for item_result in batch_result:
                if isinstance(item_result, Exception):
                    logger.error(f"Item processing failed: {item_result}")
                    continue
                results.append(item_result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "thread_pool_operations": self._stats["thread_pool_operations"],
            "process_pool_operations": self._stats["process_pool_operations"],
            "timeouts": self._stats["timeouts"],
            "errors": self._stats["errors"],
            "total_duration": self._stats["total_duration"],
            "avg_duration": (
                self._stats["total_duration"] / 
                max(1, self._stats["thread_pool_operations"] + self._stats["process_pool_operations"])
            ),
            "thread_pool_size": self.config.max_workers,
            "process_pool_size": self.config.max_process_workers
        }
    
    async def shutdown(self):
        """Shutdown executors gracefully."""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None

# Global async executor instance
async_executor = AsyncExecutor()

def asyncify(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """Decorator to convert synchronous functions to async."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        return await async_executor.run_in_thread(func, *args, **kwargs)
    return wrapper

def processify(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """Decorator to convert synchronous functions to async using process pool."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        return await async_executor.run_in_process(func, *args, **kwargs)
    return wrapper

@asynccontextmanager
async def async_operation_context(operation_name: str, timeout: Optional[float] = None):
    """Context manager for async operations with monitoring."""
    start_time = time.time()
    operation_id = f"{operation_name}_{int(start_time * 1000)}"
    
    logger.debug(f"Starting async operation: {operation_id}")
    
    try:
        yield operation_id
        duration = time.time() - start_time
        logger.debug(f"Completed async operation: {operation_id} in {duration:.3f}s")
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed async operation: {operation_id} after {duration:.3f}s - {e}")
        raise

class AsyncRateLimiter:
    """Rate limiter for async operations."""
    
    def __init__(self, max_operations: int = 10, time_window: float = 1.0):
        self.max_operations = max_operations
        self.time_window = time_window
        self._operations = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire permission to perform an operation."""
        async with self._lock:
            now = time.time()
            
            # Remove old operations outside the time window
            self._operations = [
                op_time for op_time in self._operations
                if now - op_time < self.time_window
            ]
            
            # Check if we can perform the operation
            if len(self._operations) < self.max_operations:
                self._operations.append(now)
                return True
            
            return False
    
    async def wait_for_slot(self) -> None:
        """Wait until a slot is available."""
        while not await self.acquire():
            await asyncio.sleep(0.1)

class AsyncBatchProcessor:
    """Process items in batches with rate limiting and error handling."""
    
    def __init__(
        self,
        batch_size: int = 100,
        max_concurrent: int = 5,
        rate_limiter: Optional[AsyncRateLimiter] = None
    ):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.rate_limiter = rate_limiter
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process(
        self,
        items: List[Any],
        processor: Callable[[Any], Awaitable[T]],
        error_handler: Optional[Callable[[Any, Exception], None]] = None
    ) -> List[T]:
        """Process items in batches."""
        results = []
        
        async def process_batch(batch: List[Any]) -> List[T]:
            async with self.semaphore:
                if self.rate_limiter:
                    await self.rate_limiter.wait_for_slot()
                
                batch_results = []
                for item in batch:
                    try:
                        result = await processor(item)
                        batch_results.append(result)
                    except Exception as e:
                        if error_handler:
                            error_handler(item, e)
                        else:
                            logger.error(f"Error processing item: {e}")
                
                return batch_results
        
        # Split items into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Process batches concurrently
        batch_tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing failed: {batch_result}")
                continue
            
            results.extend(batch_result)
        
        return results

# Utility functions for common async patterns
async def retry_async(
    func: Callable[..., Awaitable[T]],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    *args,
    **kwargs
) -> T:
    """Retry an async function with exponential backoff."""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                wait_time = delay * (backoff_factor ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time:.2f}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")
                raise last_exception
    
    raise last_exception

async def timeout_async(
    func: Callable[..., Awaitable[T]],
    timeout: float,
    *args,
    **kwargs
) -> T:
    """Execute an async function with a timeout."""
    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)

async def gather_with_concurrency_limit(
    tasks: List[Awaitable[T]],
    limit: int = 10
) -> List[T]:
    """Gather tasks with a concurrency limit."""
    semaphore = asyncio.Semaphore(limit)
    
    async def limited_task(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task
    
    limited_tasks = [limited_task(task) for task in tasks]
    return await asyncio.gather(*limited_tasks)
