"""
Base Core Class

Provides common functionality for all core components including:
- Structured logging with correlation IDs
- Performance monitoring and metrics
- Error handling and retry logic
- Configuration management
- Health checks and diagnostics
"""

import logging
import time
import asyncio
import traceback
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
import json
import hashlib
from contextlib import asynccontextmanager
from app.config import settings

# Type variables for generic operations
T = TypeVar('T')
R = TypeVar('R')

@dataclass
class OperationMetrics:
    """Metrics for tracking operation performance."""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthStatus:
    """Health status for core components."""
    component_name: str
    status: str  # "healthy", "degraded", "unhealthy"
    last_check: datetime
    response_time_ms: float
    error_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

class BaseCore:
    """
    Base class for all core components providing common functionality.
    
    Features:
    - Structured logging with correlation IDs
    - Performance monitoring and metrics collection
    - Error handling with retry logic
    - Health check capabilities
    - Configuration management
    - Async operation support
    """
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = logging.getLogger(f"core.{component_name}")
        self.metrics: Dict[str, OperationMetrics] = {}
        self.health_status = HealthStatus(
            component_name=component_name,
            status="unknown",
            last_check=datetime.utcnow(),
            response_time_ms=0.0
        )
        self._operation_count = 0
        self._error_count = 0
        self._start_time = datetime.utcnow()
        
        # Performance thresholds
        self.slow_operation_threshold_ms = 1000  # 1 second
        self.error_threshold_percent = 10  # 10% error rate
        
    def _generate_correlation_id(self, operation_name: str) -> str:
        """Generate a unique correlation ID for tracking operations."""
        timestamp = int(time.time() * 1000)
        random_suffix = hashlib.md5(f"{operation_name}{timestamp}".encode()).hexdigest()[:8]
        return f"{self.component_name}_{operation_name}_{timestamp}_{random_suffix}"
    
    def _log_operation_start(self, operation_name: str, correlation_id: str, **kwargs):
        """Log the start of an operation."""
        self.logger.info(
            f"Operation started: {operation_name}",
            extra={
                "correlation_id": correlation_id,
                "operation": operation_name,
                "component": self.component_name,
                "metadata": kwargs
            }
        )
    
    def _log_operation_success(self, operation_name: str, correlation_id: str, duration_ms: float, **kwargs):
        """Log successful operation completion."""
        log_level = logging.WARNING if duration_ms > self.slow_operation_threshold_ms else logging.INFO
        self.logger.log(
            log_level,
            f"Operation completed: {operation_name} ({duration_ms:.2f}ms)",
            extra={
                "correlation_id": correlation_id,
                "operation": operation_name,
                "component": self.component_name,
                "duration_ms": duration_ms,
                "status": "success",
                "metadata": kwargs
            }
        )
    
    def _log_operation_error(self, operation_name: str, correlation_id: str, error: Exception, duration_ms: float, **kwargs):
        """Log operation errors."""
        self.logger.error(
            f"Operation failed: {operation_name} ({duration_ms:.2f}ms) - {str(error)}",
            extra={
                "correlation_id": correlation_id,
                "operation": operation_name,
                "component": self.component_name,
                "duration_ms": duration_ms,
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "metadata": kwargs
            }
        )
    
    def _record_metrics(self, operation_name: str, correlation_id: str, start_time: datetime, 
                       end_time: datetime, success: bool, error: Optional[Exception] = None, **kwargs):
        """Record operation metrics."""
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        metrics = OperationMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            success=success,
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
            metadata=kwargs
        )
        
        self.metrics[correlation_id] = metrics
        self._operation_count += 1
        
        if not success:
            self._error_count += 1
    
    def _should_retry(self, error: Exception, retry_count: int, max_retries: int) -> bool:
        """Determine if an operation should be retried."""
        if retry_count >= max_retries:
            return False
        
        # Retry on specific error types
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError,
        )
        
        return isinstance(error, retryable_errors)
    
    def _calculate_backoff_delay(self, retry_count: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Calculate exponential backoff delay."""
        delay = min(base_delay * (2 ** retry_count), max_delay)
        # Add jitter to prevent thundering herd
        jitter = delay * 0.1 * (hash(str(retry_count)) % 100) / 100
        return delay + jitter
    
    async def _execute_with_retry(
        self,
        operation: Callable[..., R],
        operation_name: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        *args,
        **kwargs
    ) -> R:
        """Execute an operation with retry logic."""
        correlation_id = self._generate_correlation_id(operation_name)
        start_time = datetime.utcnow()
        
        self._log_operation_start(operation_name, correlation_id, **kwargs)
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                end_time = datetime.utcnow()
                self._record_metrics(operation_name, correlation_id, start_time, end_time, True, **kwargs)
                self._log_operation_success(operation_name, correlation_id, 
                                          (end_time - start_time).total_seconds() * 1000, **kwargs)
                
                return result
                
            except Exception as error:
                end_time = datetime.utcnow()
                duration_ms = (end_time - start_time).total_seconds() * 1000
                
                if attempt == max_retries or not self._should_retry(error, attempt, max_retries):
                    self._record_metrics(operation_name, correlation_id, start_time, end_time, False, error, **kwargs)
                    self._log_operation_error(operation_name, correlation_id, error, duration_ms, **kwargs)
                    raise
                
                # Retry with backoff
                delay = self._calculate_backoff_delay(attempt, base_delay)
                self.logger.warning(
                    f"Operation failed, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries + 1}): {str(error)}",
                    extra={
                        "correlation_id": correlation_id,
                        "operation": operation_name,
                        "component": self.component_name,
                        "attempt": attempt + 1,
                        "max_attempts": max_retries + 1,
                        "retry_delay": delay,
                        "error": str(error)
                    }
                )
                
                await asyncio.sleep(delay)
    
    @asynccontextmanager
    async def _operation_context(self, operation_name: str, **kwargs):
        """Context manager for operation tracking."""
        correlation_id = self._generate_correlation_id(operation_name)
        start_time = datetime.utcnow()
        
        self._log_operation_start(operation_name, correlation_id, **kwargs)
        
        try:
            yield correlation_id
            end_time = datetime.utcnow()
            self._record_metrics(operation_name, correlation_id, start_time, end_time, True, **kwargs)
            self._log_operation_success(operation_name, correlation_id, 
                                      (end_time - start_time).total_seconds() * 1000, **kwargs)
        except Exception as error:
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            self._record_metrics(operation_name, correlation_id, start_time, end_time, False, error, **kwargs)
            self._log_operation_error(operation_name, correlation_id, error, duration_ms, **kwargs)
            raise
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of operation metrics."""
        if not self.metrics:
            return {
                "component": self.component_name,
                "total_operations": 0,
                "success_rate": 100.0,
                "average_duration_ms": 0.0,
                "error_count": 0
            }
        
        total_ops = len(self.metrics)
        successful_ops = sum(1 for m in self.metrics.values() if m.success)
        error_count = total_ops - successful_ops
        success_rate = (successful_ops / total_ops) * 100 if total_ops > 0 else 100.0
        
        durations = [m.duration_ms for m in self.metrics.values() if m.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        return {
            "component": self.component_name,
            "total_operations": total_ops,
            "successful_operations": successful_ops,
            "error_count": error_count,
            "success_rate": round(success_rate, 2),
            "average_duration_ms": round(avg_duration, 2),
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
            "last_operation": max(m.start_time for m in self.metrics.values()).isoformat() if self.metrics else None
        }
    
    async def health_check(self) -> HealthStatus:
        """Perform a health check for this component."""
        start_time = datetime.utcnow()
        
        try:
            # Basic health check - can be overridden by subclasses
            await self._basic_health_check()
            
            end_time = datetime.utcnow()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Determine status based on error rate
            metrics_summary = self.get_metrics_summary()
            error_rate = 100 - metrics_summary.get("success_rate", 100)
            
            if error_rate > self.error_threshold_percent:
                status = "degraded"
            elif response_time_ms > self.slow_operation_threshold_ms:
                status = "degraded"
            else:
                status = "healthy"
            
            self.health_status = HealthStatus(
                component_name=self.component_name,
                status=status,
                last_check=end_time,
                response_time_ms=response_time_ms,
                error_count=metrics_summary.get("error_count", 0),
                details=metrics_summary
            )
            
            return self.health_status
            
        except Exception as error:
            end_time = datetime.utcnow()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            
            self.health_status = HealthStatus(
                component_name=self.component_name,
                status="unhealthy",
                last_check=end_time,
                response_time_ms=response_time_ms,
                error_count=self.health_status.error_count + 1,
                details={"error": str(error)}
            )
            
            return self.health_status
    
    async def _basic_health_check(self):
        """Basic health check - can be overridden by subclasses."""
        # Default implementation - just check if component is responsive
        pass
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get component configuration."""
        return {
            "component_name": self.component_name,
            "slow_operation_threshold_ms": self.slow_operation_threshold_ms,
            "error_threshold_percent": self.error_threshold_percent,
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds()
        }
    
    def reset_metrics(self):
        """Reset operation metrics."""
        self.metrics.clear()
        self._operation_count = 0
        self._error_count = 0
        self._start_time = datetime.utcnow()
        self.logger.info(f"Metrics reset for component: {self.component_name}")

def core_operation(operation_name: str, max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for core operations with automatic retry and metrics."""
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        async def async_wrapper(self: BaseCore, *args, **kwargs) -> R:
            return await self._execute_with_retry(
                func, operation_name, max_retries, base_delay, *args, **kwargs
            )
        
        @wraps(func)
        def sync_wrapper(self: BaseCore, *args, **kwargs) -> R:
            # For sync functions, we need to handle them differently
            correlation_id = self._generate_correlation_id(operation_name)
            start_time = datetime.utcnow()
            
            self._log_operation_start(operation_name, correlation_id, **kwargs)
            
            try:
                result = func(self, *args, **kwargs)
                end_time = datetime.utcnow()
                self._record_metrics(operation_name, correlation_id, start_time, end_time, True, **kwargs)
                self._log_operation_success(operation_name, correlation_id, 
                                          (end_time - start_time).total_seconds() * 1000, **kwargs)
                return result
            except Exception as error:
                end_time = datetime.utcnow()
                duration_ms = (end_time - start_time).total_seconds() * 1000
                self._record_metrics(operation_name, correlation_id, start_time, end_time, False, error, **kwargs)
                self._log_operation_error(operation_name, correlation_id, error, duration_ms, **kwargs)
                raise
        
        # Return async wrapper if function is async, sync wrapper otherwise
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
