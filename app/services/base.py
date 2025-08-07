"""
Base Service Class

"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Generic
from datetime import datetime
from contextlib import asynccontextmanager

from app.core.database import get_db
from app.core.quota_manager import quota_manager
from app.config import settings

T = TypeVar('T')

class BaseService(ABC, Generic[T]):
    """
    Base service class providing common functionality for all services.
    
    Features:
    - Consistent logging setup
    - Database session management
    - Error handling patterns
    - Quota management integration
    - Performance monitoring
    """
    
    def __init__(self, service_name: Optional[str] = None):
        """
        Initialize the base service.
        
        Args:
            service_name: Name of the service for logging purposes
        """
        self.service_name = service_name or self.__class__.__name__
        self.logger = logging.getLogger(f"services.{self.service_name.lower()}")
        
        # Performance tracking
        self._operation_times: Dict[str, float] = {}
        self._error_counts: Dict[str, int] = {}
    
    @asynccontextmanager
    async def get_db_session(self):
        """Get a database session with proper error handling."""
        async for db in get_db():
            try:
                yield db
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Database operation failed: {e}")
                raise
            finally:
                await db.close()
    
    async def check_quota(self, repo_full_name: str) -> bool:
        """
        Check if the repository has available quota.
        
        Args:
            repo_full_name: Repository full name
            
        Returns:
            True if quota is available, False otherwise
        """
        return await quota_manager.check_quota(repo_full_name)
    
    def log_operation_start(self, operation: str, **kwargs) -> datetime:
        """
        Log the start of an operation for performance tracking.
        
        Args:
            operation: Name of the operation
            **kwargs: Additional context data
            
        Returns:
            Start timestamp
        """
        start_time = datetime.utcnow()
        self.logger.info(f"Starting {operation}", extra=kwargs)
        return start_time
    
    def log_operation_complete(self, operation: str, start_time: datetime, 
                             success: bool = True, **kwargs) -> float:
        """
        Log the completion of an operation and calculate duration.
        
        Args:
            operation: Name of the operation
            start_time: Start timestamp from log_operation_start
            success: Whether the operation was successful
            **kwargs: Additional context data
            
        Returns:
            Duration in seconds
        """
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        log_level = self.logger.info if success else self.logger.error
        log_level(f"Completed {operation} in {duration:.2f}s", extra={
            "duration": duration,
            "success": success,
            **kwargs
        })
        
        # Track performance metrics
        if operation not in self._operation_times:
            self._operation_times[operation] = []
        self._operation_times[operation].append(duration)
        
        return duration
    
    def log_error(self, operation: str, error: Exception, **kwargs) -> None:
        """
        Log an error with consistent formatting.
        
        Args:
            operation: Name of the operation that failed
            error: The exception that occurred
            **kwargs: Additional context data
        """
        # Track error counts
        if operation not in self._error_counts:
            self._error_counts[operation] = 0
        self._error_counts[operation] += 1
        
        self.logger.error(
            f"Error in {operation}: {str(error)}",
            extra={
                "error_type": type(error).__name__,
                "error_count": self._error_counts[operation],
                **kwargs
            },
            exc_info=True
        )
    
    async def execute_with_retry(self, operation: str, func, max_retries: int = 3, 
                               *args, **kwargs) -> T:
        """
        Execute a function with retry logic and proper error handling.
        
        Args:
            operation: Name of the operation for logging
            func: Function to execute
            max_retries: Maximum number of retry attempts
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function execution
            
        Raises:
            Exception: If all retry attempts fail
        """
        start_time = self.log_operation_start(operation)
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                self.log_operation_complete(operation, start_time, success=True)
                return result
                
            except Exception as e:
                last_error = e
                self.log_error(operation, e, attempt=attempt + 1)
                
                if attempt < max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying {operation} in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
        
        # All retries failed
        self.log_operation_complete(operation, start_time, success=False)
        raise last_error
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the service.
        
        Returns:
            Dictionary containing performance metrics
        """
        stats = {
            "service_name": self.service_name,
            "operation_times": {},
            "error_counts": self._error_counts.copy(),
            "total_operations": 0,
            "total_errors": sum(self._error_counts.values())
        }
        
        for operation, times in self._operation_times.items():
            if times:
                stats["operation_times"][operation] = {
                    "count": len(times),
                    "avg_duration": sum(times) / len(times),
                    "min_duration": min(times),
                    "max_duration": max(times)
                }
                stats["total_operations"] += len(times)
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._operation_times.clear()
        self._error_counts.clear()
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check for the service.
        
        Returns:
            Dictionary containing health status information
        """
        pass
    
    async def cleanup(self) -> None:
        """
        Clean up resources used by the service.
        Override in subclasses if needed.
        """
        pass

# Import asyncio for the retry logic
import asyncio
