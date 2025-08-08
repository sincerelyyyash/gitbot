"""
Payload Validator

Provides payload validation and size limiting for incoming requests:
- Request body size validation
- Payload structure validation
- Rate limiting for large payloads
- Memory-efficient payload processing
"""

import json
import logging
import hashlib
from typing import Any, Dict, Optional, List, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from fastapi import HTTPException, status, Request
from pydantic import BaseModel, ValidationError, Field
import re

logger = logging.getLogger(__name__)

@dataclass
class PayloadConfig:
    """Configuration for payload validation."""
    max_body_size: int = 10 * 1024 * 1024  # 10MB default
    max_json_depth: int = 10
    max_array_length: int = 10000
    max_string_length: int = 1000000  # 1MB
    max_object_keys: int = 1000
    enable_compression: bool = True
    compression_threshold: int = 1024  # 1KB
    validation_timeout: float = 5.0  # 5 seconds

class PayloadSizeError(Exception):
    """Raised when payload exceeds size limits."""
    pass

class PayloadValidationError(Exception):
    """Raised when payload validation fails."""
    pass

class PayloadValidator:
    """Validates and processes incoming payloads with size limits."""
    
    def __init__(self, config: PayloadConfig = None):
        self.config = config or PayloadConfig()
        self._size_cache: Dict[str, int] = {}
        self._validation_stats = {
            "total_requests": 0,
            "size_violations": 0,
            "validation_errors": 0,
            "processing_timeouts": 0
        }
    
    async def validate_request_body(
        self,
        request: Request,
        max_size: Optional[int] = None
    ) -> bytes:
        """
        Validate and read request body with size limits.
        
        Args:
            request: FastAPI request object
            max_size: Maximum allowed body size in bytes
            
        Returns:
            Request body as bytes
            
        Raises:
            HTTPException: If validation fails
        """
        max_size = max_size or self.config.max_body_size
        
        try:
            # Check content length header first
            content_length = request.headers.get("content-length")
            if content_length:
                content_length = int(content_length)
                if content_length > max_size:
                    self._validation_stats["size_violations"] += 1
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Request body too large. Maximum size: {max_size} bytes"
                    )
            
            # Read body with timeout
            try:
                body = await asyncio.wait_for(
                    request.body(),
                    timeout=self.config.validation_timeout
                )
            except asyncio.TimeoutError:
                self._validation_stats["processing_timeouts"] += 1
                raise HTTPException(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    detail="Request body reading timed out"
                )
            
            # Validate actual body size
            if len(body) > max_size:
                self._validation_stats["size_violations"] += 1
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request body too large. Size: {len(body)} bytes, Maximum: {max_size} bytes"
                )
            
            self._validation_stats["total_requests"] += 1
            return body
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Request body validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request body"
            )
    
    async def validate_json_payload(
        self,
        body: bytes,
        schema: Optional[BaseModel] = None,
        max_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate JSON payload with size and structure limits.
        
        Args:
            body: Request body as bytes
            schema: Optional Pydantic schema for validation
            max_size: Maximum allowed JSON size in bytes
            
        Returns:
            Parsed JSON data
            
        Raises:
            HTTPException: If validation fails
        """
        max_size = max_size or self.config.max_body_size
        
        try:
            # Parse JSON with timeout
            try:
                data = await asyncio.wait_for(
                    asyncio.to_thread(json.loads, body.decode('utf-8')),
                    timeout=self.config.validation_timeout
                )
            except asyncio.TimeoutError:
                self._validation_stats["processing_timeouts"] += 1
                raise HTTPException(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    detail="JSON parsing timed out"
                )
            
            # Validate JSON structure
            await self._validate_json_structure(data)
            
            # Validate against schema if provided
            if schema:
                try:
                    validated_data = schema(**data)
                    return validated_data.dict()
                except ValidationError as e:
                    self._validation_stats["validation_errors"] += 1
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Schema validation failed: {e.errors()}"
                    )
            
            return data
            
        except json.JSONDecodeError as e:
            self._validation_stats["validation_errors"] += 1
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON: {str(e)}"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"JSON payload validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON validation failed"
            )
    
    async def _validate_json_structure(self, data: Any, depth: int = 0) -> None:
        """
        Validate JSON structure for depth, size, and complexity.
        
        Args:
            data: JSON data to validate
            depth: Current nesting depth
            
        Raises:
            PayloadValidationError: If validation fails
        """
        if depth > self.config.max_json_depth:
            raise PayloadValidationError(f"JSON depth exceeds maximum: {self.config.max_json_depth}")
        
        if isinstance(data, dict):
            if len(data) > self.config.max_object_keys:
                raise PayloadValidationError(f"Object has too many keys: {len(data)}")
            
            for key, value in data.items():
                if not isinstance(key, str):
                    raise PayloadValidationError("Object keys must be strings")
                
                if len(key) > self.config.max_string_length:
                    raise PayloadValidationError(f"Key too long: {len(key)} characters")
                
                await self._validate_json_structure(value, depth + 1)
        
        elif isinstance(data, list):
            if len(data) > self.config.max_array_length:
                raise PayloadValidationError(f"Array too long: {len(data)} items")
            
            for item in data:
                await self._validate_json_structure(item, depth + 1)
        
        elif isinstance(data, str):
            if len(data) > self.config.max_string_length:
                raise PayloadValidationError(f"String too long: {len(data)} characters")
        
        elif isinstance(data, (int, float, bool, type(None))):
            # These types are always valid
            pass
        
        else:
            raise PayloadValidationError(f"Unsupported data type: {type(data)}")
    
    def estimate_payload_size(self, data: Any) -> int:
        """
        Estimate the size of a payload in bytes.
        
        Args:
            data: Data to estimate size for
            
        Returns:
            Estimated size in bytes
        """
        try:
            json_str = json.dumps(data, separators=(',', ':'))
            return len(json_str.encode('utf-8'))
        except Exception:
            # Fallback estimation
            return len(str(data).encode('utf-8'))
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "total_requests": self._validation_stats["total_requests"],
            "size_violations": self._validation_stats["size_violations"],
            "validation_errors": self._validation_stats["validation_errors"],
            "processing_timeouts": self._validation_stats["processing_timeouts"],
            "violation_rate": (
                self._validation_stats["size_violations"] + self._validation_stats["validation_errors"]
            ) / max(1, self._validation_stats["total_requests"]),
            "config": {
                "max_body_size": self.config.max_body_size,
                "max_json_depth": self.config.max_json_depth,
                "max_array_length": self.config.max_array_length,
                "max_string_length": self.config.max_string_length,
                "max_object_keys": self.config.max_object_keys
            }
        }

class WebhookPayloadValidator(PayloadValidator):
    """Specialized validator for GitHub webhook payloads."""
    
    def __init__(self, config: PayloadConfig = None):
        super().__init__(config)
        self._webhook_schemas = self._initialize_webhook_schemas()
    
    def _initialize_webhook_schemas(self) -> Dict[str, BaseModel]:
        """Initialize webhook payload schemas."""
        from app.models.github import (
            IssueCommentPayload, IssuesPayload, PushPayload,
            InstallationPayload, InstallationRepositoriesPayload,
            PullRequestPayload, PullRequestReviewPayload,
            PullRequestReviewCommentPayload
        )
        
        return {
            "issue_comment": IssueCommentPayload,
            "issues": IssuesPayload,
            "push": PushPayload,
            "installation": InstallationPayload,
            "installation_repositories": InstallationRepositoriesPayload,
            "pull_request": PullRequestPayload,
            "pull_request_review": PullRequestReviewPayload,
            "pull_request_review_comment": PullRequestReviewCommentPayload
        }
    
    async def validate_webhook_payload(
        self,
        request: Request,
        event_type: str
    ) -> Dict[str, Any]:
        """
        Validate GitHub webhook payload.
        
        Args:
            request: FastAPI request object
            event_type: GitHub event type
            
        Returns:
            Validated webhook payload
            
        Raises:
            HTTPException: If validation fails
        """
        # Get schema for event type
        schema = self._webhook_schemas.get(event_type)
        
        # Validate request body
        body = await self.validate_request_body(request)
        
        # Validate JSON payload
        payload = await self.validate_json_payload(body, schema)
        
        # Additional webhook-specific validation
        await self._validate_webhook_specific(payload, event_type)
        
        return payload
    
    async def _validate_webhook_specific(
        self,
        payload: Dict[str, Any],
        event_type: str
    ) -> None:
        """
        Perform webhook-specific validation.
        
        Args:
            payload: Webhook payload
            event_type: GitHub event type
            
        Raises:
            PayloadValidationError: If validation fails
        """
        # Validate required fields based on event type
        required_fields = self._get_required_fields(event_type)
        
        for field in required_fields:
            if field not in payload:
                raise PayloadValidationError(f"Missing required field: {field}")
        
        # Validate repository information
        if "repository" in payload:
            repo = payload["repository"]
            if not isinstance(repo, dict):
                raise PayloadValidationError("Repository must be an object")
            
            if "full_name" not in repo:
                raise PayloadValidationError("Repository missing full_name")
            
            # Validate repository name format
            full_name = repo["full_name"]
            if not re.match(r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$', full_name):
                raise PayloadValidationError(f"Invalid repository name format: {full_name}")
    
    def _get_required_fields(self, event_type: str) -> List[str]:
        """Get required fields for a webhook event type."""
        base_fields = ["repository"]
        
        event_specific_fields = {
            "issue_comment": ["issue", "comment"],
            "issues": ["issue"],
            "push": ["ref", "commits"],
            "installation": ["installation"],
            "installation_repositories": ["installation", "repositories_added"],
            "pull_request": ["pull_request"],
            "pull_request_review": ["pull_request", "review"],
            "pull_request_review_comment": ["pull_request", "comment"]
        }
        
        return base_fields + event_specific_fields.get(event_type, [])

class PayloadRateLimiter:
    """Rate limiter for large payloads."""
    
    def __init__(self, max_large_payloads: int = 10, time_window: float = 60.0):
        self.max_large_payloads = max_large_payloads
        self.time_window = time_window
        self._large_payloads = []
        self._lock = asyncio.Lock()
    
    async def check_large_payload_limit(self, payload_size: int, threshold: int = 1024 * 1024) -> bool:
        """
        Check if large payload is within rate limits.
        
        Args:
            payload_size: Size of the payload in bytes
            threshold: Size threshold for considering payload "large"
            
        Returns:
            True if within limits, False otherwise
        """
        if payload_size < threshold:
            return True
        
        async with self._lock:
            now = datetime.utcnow()
            
            # Remove old entries
            self._large_payloads = [
                timestamp for timestamp in self._large_payloads
                if now - timestamp < timedelta(seconds=self.time_window)
            ]
            
            # Check if we can accept another large payload
            if len(self._large_payloads) >= self.max_large_payloads:
                return False
            
            # Add current payload
            self._large_payloads.append(now)
            return True

# Global validator instances
payload_validator = PayloadValidator()
webhook_validator = WebhookPayloadValidator()
payload_rate_limiter = PayloadRateLimiter()

# Decorator for payload validation
def validate_payload(max_size: Optional[int] = None, schema: Optional[BaseModel] = None):
    """
    Decorator for payload validation.
    
    Args:
        max_size: Maximum allowed payload size
        schema: Optional Pydantic schema for validation
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(request: Request, *args, **kwargs):
            # Validate request body
            body = await payload_validator.validate_request_body(request, max_size)
            
            # Validate JSON payload if schema provided
            if schema:
                payload = await payload_validator.validate_json_payload(body, schema, max_size)
                return await func(request, payload, *args, **kwargs)
            
            return await func(request, body, *args, **kwargs)
        
        return wrapper
    return decorator
