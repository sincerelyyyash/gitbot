from github import Github, RateLimitExceededException, GithubException
from typing import TypeVar, Callable, Any, Optional, Dict
import asyncio
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
import random

logger = logging.getLogger(__name__)

T = TypeVar('T')

class GitHubRateLimitHandler:
    """Handles GitHub API rate limiting with exponential backoff retry."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        jitter: float = 0.1
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self._rate_limits: Dict[str, Dict] = {}
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = min(
            self.base_delay * (2 ** attempt),
            self.max_delay
        )
        jitter_amount = delay * self.jitter
        return delay + random.uniform(-jitter_amount, jitter_amount)
    
    def _update_rate_limit(self, client: Github, category: str = "core"):
        """Update rate limit information for a category."""
        try:
            limits = client.get_rate_limit()
            limit_data = getattr(limits, category, None)
            if limit_data:
                self._rate_limits[category] = {
                    "remaining": limit_data.remaining,
                    "limit": limit_data.limit,
                    "reset_at": limit_data.reset.timestamp(),
                }
                logger.debug(
                    f"Rate limit for {category}: "
                    f"{limit_data.remaining}/{limit_data.limit} "
                    f"resets at {limit_data.reset.isoformat()}"
                )
        except Exception as e:
            logger.warning(f"Failed to fetch rate limit info: {e}")
    
    async def _wait_for_reset(self, category: str, delay: Optional[float] = None):
        """Wait for rate limit reset."""
        if category in self._rate_limits:
            reset_at = self._rate_limits[category]["reset_at"]
            now = time.time()
            wait_time = reset_at - now
            if wait_time > 0:
                wait_time = min(wait_time, delay or wait_time)
                logger.info(f"Waiting {wait_time:.1f}s for rate limit reset")
                await asyncio.sleep(wait_time)
    
    def with_rate_limit(
        self,
        category: str = "core",
        on_limit_reached: Optional[Callable] = None
    ):
        """Decorator for handling rate limits with retries."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                client = None
                for arg in args:
                    if isinstance(arg, Github):
                        client = arg
                        break
                
                if not client:
                    logger.warning("No Github client found in arguments")
                    return await func(*args, **kwargs)
                
                attempt = 0
                while True:
                    try:
                        # Update rate limit info
                        self._update_rate_limit(client, category)
                        
                        # Check if we need to wait
                        if category in self._rate_limits:
                            if self._rate_limits[category]["remaining"] <= 1:
                                if on_limit_reached:
                                    await on_limit_reached(self._rate_limits[category])
                                await self._wait_for_reset(category)
                        
                        # Execute function
                        return await func(*args, **kwargs)
                    
                    except RateLimitExceededException as e:
                        attempt += 1
                        if attempt >= self.max_retries:
                            logger.error(
                                f"Rate limit exceeded after {attempt} retries: {e}"
                            )
                            raise
                        
                        # Calculate delay and wait
                        delay = self._calculate_delay(attempt)
                        logger.warning(
                            f"Rate limit exceeded, attempt {attempt}/{self.max_retries}. "
                            f"Retrying in {delay:.1f}s"
                        )
                        
                        # Update rate limit info and wait
                        self._update_rate_limit(client, category)
                        await self._wait_for_reset(category, delay)
                    
                    except GithubException as e:
                        if e.status == 403 and "API rate limit exceeded" in str(e):
                            # Handle secondary rate limits
                            attempt += 1
                            if attempt >= self.max_retries:
                                logger.error(
                                    f"Secondary rate limit exceeded after {attempt} "
                                    f"retries: {e}"
                                )
                                raise
                            
                            delay = self._calculate_delay(attempt)
                            logger.warning(
                                f"Secondary rate limit exceeded, attempt "
                                f"{attempt}/{self.max_retries}. Retrying in {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                        else:
                            raise
            
            return wrapper
        return decorator

# Global instance
github_rate_limiter = GitHubRateLimitHandler() 