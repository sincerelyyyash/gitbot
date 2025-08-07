"""
Rate Limiter Manager

Provides comprehensive rate limiting for external API calls including:
- GitHub API rate limiting with exponential backoff
- Multi-service rate limiting
- Rate limit monitoring and metrics
- Adaptive rate limiting strategies
"""

import logging
import asyncio
import time
import random
from typing import Dict, Any, Optional, Callable, TypeVar, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from enum import Enum
from github import Github, RateLimitExceededException, GithubException
from .base import BaseCore, core_operation

T = TypeVar('T')

class RateLimitCategory(Enum):
    """Rate limit categories for different API endpoints."""
    CORE = "core"
    SEARCH = "search"
    PR = "pr"
    ISSUES = "issues"
    COMMITS = "commits"
    REPOSITORIES = "repositories"
    USERS = "users"
    EXTERNAL_API = "external_api"

@dataclass
class RateLimitInfo:
    """Rate limit information for a category."""
    category: RateLimitCategory
    remaining: int
    limit: int
    reset_at: datetime
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 60.0
    jitter: float = 0.1
    burst_limit: int = 5
    window_size: int = 60  # seconds

class RateLimitManager(BaseCore):
    """
    Rate limiter manager for handling API rate limits.
    
    Features:
    - GitHub API rate limiting with exponential backoff
    - Multi-service rate limiting
    - Rate limit monitoring and metrics
    - Adaptive rate limiting strategies
    - Burst protection and window-based limiting
    """
    
    def __init__(self):
        super().__init__("rate_limiter")
        
        # Rate limit configurations
        self._rate_limit_configs = {
            RateLimitCategory.CORE: RateLimitConfig(max_retries=3, base_delay=2.0, max_delay=60.0),
            RateLimitCategory.SEARCH: RateLimitConfig(max_retries=2, base_delay=5.0, max_delay=120.0),
            RateLimitCategory.PR: RateLimitConfig(max_retries=3, base_delay=2.0, max_delay=60.0),
            RateLimitCategory.ISSUES: RateLimitConfig(max_retries=3, base_delay=2.0, max_delay=60.0),
            RateLimitCategory.COMMITS: RateLimitConfig(max_retries=2, base_delay=3.0, max_delay=90.0),
            RateLimitCategory.REPOSITORIES: RateLimitConfig(max_retries=3, base_delay=2.0, max_delay=60.0),
            RateLimitCategory.USERS: RateLimitConfig(max_retries=2, base_delay=3.0, max_delay=90.0),
            RateLimitCategory.EXTERNAL_API: RateLimitConfig(max_retries=5, base_delay=1.0, max_delay=300.0)
        }
        
        # Rate limit tracking
        self._rate_limits: Dict[RateLimitCategory, RateLimitInfo] = {}
        self._request_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000
        
        # Burst protection
        self._burst_tokens: Dict[str, int] = {}
        self._last_burst_reset: Dict[str, datetime] = {}
        
        # Rate limiting metrics
        self._rate_limit_metrics = {
            "total_requests": 0,
            "rate_limited_requests": 0,
            "retry_attempts": 0,
            "backoff_delays": 0,
            "burst_rejections": 0
        }
    
    def _calculate_delay(self, attempt: int, category: RateLimitCategory) -> float:
        """Calculate delay with exponential backoff and jitter."""
        config = self._rate_limit_configs[category]
        delay = min(
            config.base_delay * (2 ** attempt),
            config.max_delay
        )
        jitter_amount = delay * config.jitter
        return delay + random.uniform(-jitter_amount, jitter_amount)
    
    def _update_rate_limit(self, client: Github, category: RateLimitCategory):
        """Update rate limit information for a category."""
        try:
            limits = client.get_rate_limit()
            
            # Map GitHub rate limit categories to our categories
            github_category_map = {
                "core": RateLimitCategory.CORE,
                "search": RateLimitCategory.SEARCH,
                "graphql": RateLimitCategory.CORE,  # GraphQL uses core limits
            }
            
            # Get the appropriate limit info
            limit_data = None
            if category == RateLimitCategory.SEARCH:
                limit_data = getattr(limits, "search", None)
            elif category == RateLimitCategory.CORE:
                limit_data = getattr(limits, "core", None)
            else:
                # For other categories, use core limits as fallback
                limit_data = getattr(limits, "core", None)
            
            if limit_data:
                self._rate_limits[category] = RateLimitInfo(
                    category=category,
                    remaining=limit_data.remaining,
                    limit=limit_data.limit,
                    reset_at=limit_data.reset,
                    last_updated=datetime.utcnow()
                )
                
                self.logger.debug(
                    f"Rate limit for {category.value}: "
                    f"{limit_data.remaining}/{limit_data.limit} "
                    f"resets at {limit_data.reset.isoformat()}"
                )
                
        except Exception as error:
            self.logger.warning(f"Failed to fetch rate limit info for {category.value}: {error}")
    
    async def _wait_for_reset(self, category: RateLimitCategory, delay: Optional[float] = None):
        """Wait for rate limit reset."""
        if category in self._rate_limits:
            reset_at = self._rate_limits[category].reset_at
            now = datetime.utcnow()
            wait_time = (reset_at - now).total_seconds()
            
            if wait_time > 0:
                wait_time = min(wait_time, delay or wait_time)
                self.logger.info(f"Waiting {wait_time:.1f}s for rate limit reset for {category.value}")
                await asyncio.sleep(wait_time)
    
    def _check_burst_limit(self, key: str, category: RateLimitCategory) -> bool:
        """Check if request is within burst limits."""
        config = self._rate_limit_configs[category]
        now = datetime.utcnow()
        
        # Reset burst tokens if window has passed
        if key not in self._last_burst_reset or \
           (now - self._last_burst_reset[key]).total_seconds() > config.window_size:
            self._burst_tokens[key] = config.burst_limit
            self._last_burst_reset[key] = now
        
        # Check if tokens available
        if self._burst_tokens[key] > 0:
            self._burst_tokens[key] -= 1
            return True
        
        self._rate_limit_metrics["burst_rejections"] += 1
        return False
    
    def with_rate_limit(
        self,
        category: RateLimitCategory = RateLimitCategory.CORE,
        burst_key: Optional[str] = None,
        on_limit_reached: Optional[Callable] = None
    ):
        """Decorator for handling rate limits with retries."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                # Generate burst key if not provided
                if not burst_key:
                    # Use function name and first few args as key
                    key_parts = [func.__name__]
                    for arg in args[:2]:  # Use first 2 args
                        if isinstance(arg, str):
                            key_parts.append(arg[:10])  # Truncate long strings
                        else:
                            key_parts.append(str(type(arg).__name__))
                    current_burst_key = "_".join(key_parts)
                else:
                    current_burst_key = burst_key
                
                # Check burst limit
                if not self._check_burst_limit(current_burst_key, category):
                    raise RateLimitExceededException(
                        f"Burst limit exceeded for {category.value}. "
                        f"Please wait before making more requests."
                    )
                
                # Find GitHub client in arguments
                client = None
                for arg in args:
                    if isinstance(arg, Github):
                        client = arg
                        break
                
                if not client:
                    self.logger.warning(f"No Github client found in arguments for {func.__name__}")
                    return await func(*args, **kwargs)
                
                config = self._rate_limit_configs[category]
                attempt = 0
                
                while True:
                    try:
                        # Update rate limit info
                        self._update_rate_limit(client, category)
                        
                        # Check if we need to wait
                        if category in self._rate_limits:
                            rate_limit_info = self._rate_limits[category]
                            if rate_limit_info.remaining <= 1:
                                if on_limit_reached:
                                    await on_limit_reached(rate_limit_info)
                                await self._wait_for_reset(category)
                        
                        # Execute function
                        result = await func(*args, **kwargs)
                        
                        # Record successful request
                        self._record_request(category, True, attempt)
                        
                        return result
                        
                    except RateLimitExceededException as error:
                        attempt += 1
                        self._rate_limit_metrics["rate_limited_requests"] += 1
                        
                        if attempt >= config.max_retries:
                            self.logger.error(
                                f"Rate limit exceeded after {attempt} retries for {category.value}: {error}"
                            )
                            self._record_request(category, False, attempt, str(error))
                            raise
                        
                        # Calculate delay and wait
                        delay = self._calculate_delay(attempt, category)
                        self._rate_limit_metrics["retry_attempts"] += 1
                        self._rate_limit_metrics["backoff_delays"] += 1
                        
                        self.logger.warning(
                            f"Rate limit exceeded for {category.value}, attempt {attempt}/{config.max_retries}. "
                            f"Retrying in {delay:.1f}s"
                        )
                        
                        # Update rate limit info and wait
                        self._update_rate_limit(client, category)
                        await self._wait_for_reset(category, delay)
                        
                    except GithubException as error:
                        if error.status == 403 and "API rate limit exceeded" in str(error):
                            # Handle secondary rate limits
                            attempt += 1
                            self._rate_limit_metrics["rate_limited_requests"] += 1
                            
                            if attempt >= config.max_retries:
                                self.logger.error(
                                    f"Secondary rate limit exceeded after {attempt} retries for {category.value}: {error}"
                                )
                                self._record_request(category, False, attempt, str(error))
                                raise
                            
                            delay = self._calculate_delay(attempt, category)
                            self._rate_limit_metrics["retry_attempts"] += 1
                            self._rate_limit_metrics["backoff_delays"] += 1
                            
                            self.logger.warning(
                                f"Secondary rate limit exceeded for {category.value}, attempt "
                                f"{attempt}/{config.max_retries}. Retrying in {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                        else:
                            self._record_request(category, False, attempt, str(error))
                            raise
                    
                    except Exception as error:
                        self._record_request(category, False, attempt, str(error))
                        raise
            
            return wrapper
        return decorator
    
    def _record_request(self, category: RateLimitCategory, success: bool, attempt: int, error: Optional[str] = None):
        """Record request for metrics and history."""
        self._rate_limit_metrics["total_requests"] += 1
        
        request_record = {
            "timestamp": datetime.utcnow(),
            "category": category.value,
            "success": success,
            "attempt": attempt,
            "error": error
        }
        
        self._request_history.append(request_record)
        
        # Keep history size manageable
        if len(self._request_history) > self._max_history_size:
            self._request_history.pop(0)
    
    def get_rate_limit_status(self, category: RateLimitCategory) -> Optional[RateLimitInfo]:
        """Get current rate limit status for a category."""
        return self._rate_limits.get(category)
    
    def get_all_rate_limits(self) -> Dict[str, Dict[str, Any]]:
        """Get all rate limit statuses."""
        result = {}
        for category, info in self._rate_limits.items():
            result[category.value] = {
                "remaining": info.remaining,
                "limit": info.limit,
                "reset_at": info.reset_at.isoformat(),
                "last_updated": info.last_updated.isoformat(),
                "utilization_percent": round((info.limit - info.remaining) / info.limit * 100, 2)
            }
        return result
    
    def get_burst_status(self) -> Dict[str, Dict[str, Any]]:
        """Get burst limit status for all keys."""
        result = {}
        now = datetime.utcnow()
        
        for key, tokens in self._burst_tokens.items():
            last_reset = self._last_burst_reset.get(key, now)
            window_remaining = max(0, self._rate_limit_configs[RateLimitCategory.CORE].window_size - 
                                 (now - last_reset).total_seconds())
            
            result[key] = {
                "tokens_remaining": tokens,
                "window_remaining_seconds": round(window_remaining, 1),
                "last_reset": last_reset.isoformat()
            }
        
        return result
    
    def get_rate_limit_metrics(self) -> Dict[str, Any]:
        """Get rate limiting metrics."""
        return {
            "component": self.component_name,
            "rate_limit_metrics": self._rate_limit_metrics.copy(),
            "recent_requests": len(self._request_history),
            "rate_limits": self.get_all_rate_limits(),
            "burst_status": self.get_burst_status()
        }
    
    async def _basic_health_check(self):
        """Basic health check for rate limiter."""
        # Check if rate limiter is responsive
        if len(self._request_history) > self._max_history_size * 0.9:
            raise Exception("Rate limiter history is approaching capacity")
    
    async def reset_metrics(self):
        """Reset rate limiting metrics."""
        self._rate_limit_metrics = {
            "total_requests": 0,
            "rate_limited_requests": 0,
            "retry_attempts": 0,
            "backoff_delays": 0,
            "burst_rejections": 0
        }
        self._request_history.clear()
        self.logger.info("Rate limiting metrics reset")
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get rate limiter configuration."""
        config = super().get_configuration()
        config.update({
            "rate_limit_configs": {
                category.value: {
                    "max_retries": config.max_retries,
                    "base_delay": config.base_delay,
                    "max_delay": config.max_delay,
                    "jitter": config.jitter,
                    "burst_limit": config.burst_limit,
                    "window_size": config.window_size
                }
                for category, config in self._rate_limit_configs.items()
            },
            "max_history_size": self._max_history_size
        })
        return config

# Global rate limiter instance
rate_limit_manager = RateLimitManager()
