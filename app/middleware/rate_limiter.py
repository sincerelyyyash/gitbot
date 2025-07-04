from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional, Callable, Dict
import time
import asyncio
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class RateLimitState:
    requests: int = 0
    reset_at: float = 0.0
    last_request: float = 0.0

class RateLimiter:
    """In-memory rate limiter with repository-specific limits."""
    
    def __init__(
        self,
        rate_limit: int = 30,  # requests per window
        window: int = 60,  # window in seconds
        burst_limit: int = 5,  # max burst requests
    ):
        self.rate_limit = rate_limit
        self.window = window
        self.burst_limit = burst_limit
        self._states: Dict[str, RateLimitState] = {}
        self._lock = asyncio.Lock()
    
    async def is_rate_limited(self, key: str) -> tuple[bool, Dict]:
        """Check if the request should be rate limited."""
        async with self._lock:
            now = time.time()
            
            # Initialize or get state
            if key not in self._states:
                self._states[key] = RateLimitState(
                    reset_at=now + self.window
                )
            state = self._states[key]
            
            # Reset if window expired
            if now > state.reset_at:
                state.requests = 0
                state.reset_at = now + self.window
            
            # Calculate rate and burst
            time_since_last = now - state.last_request if state.last_request else self.window
            effective_rate = state.requests / self.window
            is_burst = time_since_last < (self.window / self.burst_limit)
            
            # Update state
            state.last_request = now
            state.requests += 1
            
            # Check limits
            is_limited = (
                state.requests > self.rate_limit or
                (is_burst and state.requests > self.burst_limit)
            )
            
            remaining = max(0, self.rate_limit - state.requests)
            reset_in = max(0, state.reset_at - now)
            
            return is_limited, {
                "remaining": remaining,
                "reset": int(state.reset_at),
                "reset_in_seconds": int(reset_in),
            }

class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    def __init__(
        self,
        app,
        rate_limit: int = 30,
        window: int = 60,
        burst_limit: int = 5,
        exclude_paths: Optional[set[str]] = None
    ):
        super().__init__(app)
        self.rate_limiter = RateLimiter(rate_limit, window, burst_limit)
        self.exclude_paths = exclude_paths or set()
    
    async def get_key(self, request: Request) -> str:
        """Get rate limit key from request."""
        # Try to get repository from webhook payload
        try:
            body = await request.json()
            if "repository" in body and "full_name" in body["repository"]:
                return f"repo:{body['repository']['full_name']}"
        except:
            pass
        
        # Fallback to IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        return f"ip:{request.client.host}"
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Apply rate limiting to requests."""
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Get rate limit key
        key = await self.get_key(request)
        
        # Check rate limit
        is_limited, limit_info = await self.rate_limiter.is_rate_limited(key)
        
        # Set rate limit headers
        headers = {
            "X-RateLimit-Remaining": str(limit_info["remaining"]),
            "X-RateLimit-Reset": str(limit_info["reset"]),
        }
        
        if is_limited:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "reset_in_seconds": limit_info["reset_in_seconds"]
                },
                headers=headers
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value
        
        return response 