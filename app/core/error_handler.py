"""
Core Error Handler

Provides centralized error handling for all core components including:
- Error categorization and classification
- Recovery strategies and fallbacks
- Error reporting and monitoring
- User-friendly error messages
"""

import logging
from typing import Dict, Any, Optional, List, Type, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from .base import BaseCore

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    INTERNAL = "internal"
    UNKNOWN = "unknown"

@dataclass
class ErrorInfo:
    """Structured error information."""
    error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    component: str
    operation: str
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class RecoveryStrategy:
    """Error recovery strategy."""
    category: ErrorCategory
    severity: ErrorSeverity
    max_retries: int
    backoff_multiplier: float
    max_backoff: float
    fallback_action: Optional[Callable] = None
    user_message: str = "An error occurred. Please try again."

class CoreErrorHandler(BaseCore):
    """
    Centralized error handler for all core components.
    
    Features:
    - Error categorization and classification
    - Recovery strategies and fallbacks
    - Error reporting and monitoring
    - User-friendly error messages
    - Error pattern recognition
    """
    
    def __init__(self):
        super().__init__("error_handler")
        
        # Error categorization patterns
        self._error_patterns = {
            ErrorCategory.AUTHENTICATION: [
                "invalid token",
                "unauthorized",
                "authentication failed",
                "invalid credentials",
                "token expired",
                "api_key_ip_address_blocked"
            ],
            ErrorCategory.AUTHORIZATION: [
                "forbidden",
                "insufficient permissions",
                "access denied",
                "permission denied"
            ],
            ErrorCategory.NETWORK: [
                "connection",
                "timeout",
                "network",
                "dns",
                "socket",
                "connection refused",
                "connection reset"
            ],
            ErrorCategory.RATE_LIMIT: [
                "rate limit",
                "too many requests",
                "rate_limit_exceeded",
                "api rate limit exceeded"
            ],
            ErrorCategory.QUOTA: [
                "quota exceeded",
                "resource exhausted",
                "quota limit",
                "usage limit"
            ],
            ErrorCategory.VALIDATION: [
                "invalid",
                "validation",
                "malformed",
                "bad request",
                "400"
            ],
            ErrorCategory.CONFIGURATION: [
                "configuration",
                "config",
                "missing",
                "not found",
                "404"
            ],
            ErrorCategory.DATABASE: [
                "database",
                "sql",
                "connection",
                "transaction",
                "deadlock"
            ],
            ErrorCategory.EXTERNAL_API: [
                "api",
                "external",
                "service unavailable",
                "503",
                "502",
                "500"
            ]
        }
        
        # Recovery strategies
        self._recovery_strategies = {
            ErrorCategory.AUTHENTICATION: RecoveryStrategy(
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.HIGH,
                max_retries=1,  # Don't retry auth errors
                backoff_multiplier=1.0,
                max_backoff=0.0,
                user_message="Authentication failed. Please check your credentials."
            ),
            ErrorCategory.AUTHORIZATION: RecoveryStrategy(
                category=ErrorCategory.AUTHORIZATION,
                severity=ErrorSeverity.HIGH,
                max_retries=0,  # Don't retry auth errors
                backoff_multiplier=1.0,
                max_backoff=0.0,
                user_message="Access denied. Please check your permissions."
            ),
            ErrorCategory.NETWORK: RecoveryStrategy(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                max_retries=5,
                backoff_multiplier=2.0,
                max_backoff=60.0,
                user_message="Network connection issue. Please check your internet connection."
            ),
            ErrorCategory.RATE_LIMIT: RecoveryStrategy(
                category=ErrorCategory.RATE_LIMIT,
                severity=ErrorSeverity.MEDIUM,
                max_retries=3,
                backoff_multiplier=2.0,
                max_backoff=300.0,  # 5 minutes
                user_message="Rate limit exceeded. Please wait before trying again."
            ),
            ErrorCategory.QUOTA: RecoveryStrategy(
                category=ErrorCategory.QUOTA,
                severity=ErrorSeverity.HIGH,
                max_retries=1,
                backoff_multiplier=1.0,
                max_backoff=0.0,
                user_message="API quota exceeded. Please try again later."
            ),
            ErrorCategory.VALIDATION: RecoveryStrategy(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.LOW,
                max_retries=0,  # Don't retry validation errors
                backoff_multiplier=1.0,
                max_backoff=0.0,
                user_message="Invalid request. Please check your input."
            ),
            ErrorCategory.CONFIGURATION: RecoveryStrategy(
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.HIGH,
                max_retries=0,  # Don't retry config errors
                backoff_multiplier=1.0,
                max_backoff=0.0,
                user_message="Configuration error. Please check your settings."
            ),
            ErrorCategory.DATABASE: RecoveryStrategy(
                category=ErrorCategory.DATABASE,
                severity=ErrorSeverity.HIGH,
                max_retries=3,
                backoff_multiplier=1.5,
                max_backoff=30.0,
                user_message="Database error. Please try again."
            ),
            ErrorCategory.EXTERNAL_API: RecoveryStrategy(
                category=ErrorCategory.EXTERNAL_API,
                severity=ErrorSeverity.MEDIUM,
                max_retries=3,
                backoff_multiplier=2.0,
                max_backoff=60.0,
                user_message="External service error. Please try again."
            ),
            ErrorCategory.INTERNAL: RecoveryStrategy(
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.CRITICAL,
                max_retries=2,
                backoff_multiplier=1.5,
                max_backoff=10.0,
                user_message="Internal error. Please contact support."
            ),
            ErrorCategory.UNKNOWN: RecoveryStrategy(
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.MEDIUM,
                max_retries=2,
                backoff_multiplier=1.5,
                max_backoff=30.0,
                user_message="An unexpected error occurred. Please try again."
            )
        }
        
        # Error statistics
        self._error_stats: Dict[ErrorCategory, int] = {category: 0 for category in ErrorCategory}
        self._recent_errors: List[ErrorInfo] = []
        self._max_recent_errors = 100
    
    def categorize_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorCategory:
        """Categorize an error based on its type and message."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Check error patterns
        for category, patterns in self._error_patterns.items():
            for pattern in patterns:
                if pattern in error_str or pattern in error_type:
                    return category
        
        # Check for specific error types
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, ValueError):
            return ErrorCategory.VALIDATION
        elif isinstance(error, KeyError):
            return ErrorCategory.CONFIGURATION
        elif isinstance(error, PermissionError):
            return ErrorCategory.AUTHORIZATION
        
        return ErrorCategory.UNKNOWN
    
    def determine_severity(self, error: Exception, category: ErrorCategory, context: Dict[str, Any] = None) -> ErrorSeverity:
        """Determine error severity based on error and context."""
        # Critical errors
        if category in [ErrorCategory.AUTHENTICATION, ErrorCategory.CONFIGURATION]:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if category in [ErrorCategory.AUTHORIZATION, ErrorCategory.QUOTA, ErrorCategory.DATABASE]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if category in [ErrorCategory.NETWORK, ErrorCategory.RATE_LIMIT, ErrorCategory.EXTERNAL_API]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        if category == ErrorCategory.VALIDATION:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def create_error_info(
        self, 
        error: Exception, 
        component: str, 
        operation: str, 
        correlation_id: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> ErrorInfo:
        """Create structured error information."""
        category = self.categorize_error(error, context)
        severity = self.determine_severity(error, category, context)
        strategy = self._recovery_strategies[category]
        
        error_info = ErrorInfo(
            error=error,
            category=category,
            severity=severity,
            component=component,
            operation=operation,
            correlation_id=correlation_id,
            context=context or {},
            recoverable=strategy.max_retries > 0,
            max_retries=strategy.max_retries
        )
        
        # Update statistics
        self._error_stats[category] += 1
        self._recent_errors.append(error_info)
        
        # Keep only recent errors
        if len(self._recent_errors) > self._max_recent_errors:
            self._recent_errors.pop(0)
        
        return error_info
    
    def get_recovery_strategy(self, error_info: ErrorInfo) -> RecoveryStrategy:
        """Get recovery strategy for an error."""
        return self._recovery_strategies[error_info.category]
    
    def should_retry(self, error_info: ErrorInfo) -> bool:
        """Determine if an error should be retried."""
        strategy = self.get_recovery_strategy(error_info)
        return error_info.retry_count < strategy.max_retries
    
    def get_retry_delay(self, error_info: ErrorInfo) -> float:
        """Calculate retry delay for an error."""
        strategy = self.get_recovery_strategy(error_info)
        delay = strategy.backoff_multiplier ** error_info.retry_count
        return min(delay, strategy.max_backoff)
    
    def get_user_message(self, error_info: ErrorInfo) -> str:
        """Get user-friendly error message."""
        strategy = self.get_recovery_strategy(error_info)
        return strategy.user_message
    
    def log_error(self, error_info: ErrorInfo):
        """Log error with structured information."""
        log_level = {
            ErrorSeverity.LOW: logging.WARNING,
            ErrorSeverity.MEDIUM: logging.ERROR,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[error_info.severity]
        
        self.logger.log(
            log_level,
            f"Error in {error_info.component}.{error_info.operation}: {str(error_info.error)}",
            extra={
                "correlation_id": error_info.correlation_id,
                "component": error_info.component,
                "operation": error_info.operation,
                "category": error_info.category.value,
                "severity": error_info.severity.value,
                "recoverable": error_info.recoverable,
                "retry_count": error_info.retry_count,
                "max_retries": error_info.max_retries,
                "context": error_info.context
            }
        )
    
    def handle_error(
        self, 
        error: Exception, 
        component: str, 
        operation: str, 
        correlation_id: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> ErrorInfo:
        """Handle an error and return structured error information."""
        error_info = self.create_error_info(error, component, operation, correlation_id, context)
        self.log_error(error_info)
        
        # Log critical errors to additional channels if needed
        if error_info.severity == ErrorSeverity.CRITICAL:
            self._handle_critical_error(error_info)
        
        return error_info
    
    def _handle_critical_error(self, error_info: ErrorInfo):
        """Handle critical errors with additional logging/monitoring."""
        self.logger.critical(
            f"CRITICAL ERROR: {error_info.component}.{error_info.operation}",
            extra={
                "correlation_id": error_info.correlation_id,
                "error": str(error_info.error),
                "category": error_info.category.value,
                "context": error_info.context
            }
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        total_errors = sum(self._error_stats.values())
        
        stats = {
            "total_errors": total_errors,
            "errors_by_category": {
                category.value: count for category, count in self._error_stats.items()
            },
            "errors_by_severity": {
                severity.value: sum(
                    1 for error in self._recent_errors 
                    if error.severity == severity
                ) for severity in ErrorSeverity
            },
            "recent_errors_count": len(self._recent_errors),
            "most_common_category": max(self._error_stats.items(), key=lambda x: x[1])[0].value if total_errors > 0 else None
        }
        
        return stats
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors as dictionaries."""
        recent = self._recent_errors[-limit:] if limit > 0 else self._recent_errors
        
        return [
            {
                "timestamp": error.timestamp.isoformat(),
                "component": error.component,
                "operation": error.operation,
                "category": error.category.value,
                "severity": error.severity.value,
                "error_message": str(error.error),
                "correlation_id": error.correlation_id,
                "recoverable": error.recoverable,
                "retry_count": error.retry_count
            }
            for error in recent
        ]
    
    def reset_statistics(self):
        """Reset error statistics."""
        self._error_stats = {category: 0 for category in ErrorCategory}
        self._recent_errors.clear()
        self.logger.info("Error statistics reset")
    
    async def _basic_health_check(self):
        """Basic health check for error handler."""
        # Check if error handler is responsive
        if len(self._recent_errors) > self._max_recent_errors * 0.9:
            raise Exception("Error handler is approaching capacity")
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get error handler configuration."""
        config = super().get_configuration()
        config.update({
            "max_recent_errors": self._max_recent_errors,
            "error_categories": [category.value for category in ErrorCategory],
            "recovery_strategies": {
                category.value: {
                    "max_retries": strategy.max_retries,
                    "backoff_multiplier": strategy.backoff_multiplier,
                    "max_backoff": strategy.max_backoff
                }
                for category, strategy in self._recovery_strategies.items()
            }
        })
        return config
