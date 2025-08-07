"""
Base API Module

Provides the foundational BaseAPI class that all API components inherit from.
Centralizes common API functionality like logging, error handling, response formatting,
and performance monitoring.
"""

import logging
import time
import uuid
from typing import Any, Dict, Optional, Union
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core import error_handler


class BaseAPI:
    """
    Base class for all API components.
    
    Provides common functionality for:
    - Structured logging with correlation IDs
    - Error handling and response formatting
    - Performance monitoring and metrics
    - Input validation and sanitization
    - Response standardization
    """
    
    def __init__(self, name: str):
        """
        Initialize the base API component.
        
        Args:
            name: Name of the API component for logging and metrics
        """
        self.name = name
        self.logger = logging.getLogger(f"api.{name}")
        
    def _generate_correlation_id(self) -> str:
        """Generate a unique correlation ID for request tracking."""
        return str(uuid.uuid4())
    
    def _log_request(self, correlation_id: str, method: str, path: str, **kwargs):
        """Log incoming request details."""
        self.logger.info(
            f"Request started",
            extra={
                "correlation_id": correlation_id,
                "method": method,
                "path": path,
                "component": self.name,
                **kwargs
            }
        )
    
    def _log_response(self, correlation_id: str, status_code: int, duration_ms: float, **kwargs):
        """Log response details."""
        self.logger.info(
            f"Request completed",
            extra={
                "correlation_id": correlation_id,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "component": self.name,
                **kwargs
            }
        )
    
    def _log_error(self, correlation_id: str, error: Exception, **kwargs):
        """Log error details."""
        self.logger.error(
            f"Request failed: {str(error)}",
            extra={
                "correlation_id": correlation_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "component": self.name,
                **kwargs
            },
            exc_info=True
        )
    
    def create_success_response(
        self, 
        data: Any, 
        message: str = "Success",
        status_code: int = status.HTTP_200_OK,
        **kwargs
    ) -> JSONResponse:
        """
        Create a standardized success response.
        
        Args:
            data: Response data
            message: Success message
            status_code: HTTP status code
            **kwargs: Additional response metadata
            
        Returns:
            Standardized JSONResponse
        """
        response_data = {
            "status": "success",
            "message": message,
            "data": data,
            **kwargs
        }
        return JSONResponse(
            status_code=status_code,
            content=response_data
        )
    
    def create_error_response(
        self,
        error: Union[str, Exception],
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: Optional[str] = None,
        **kwargs
    ) -> JSONResponse:
        """
        Create a standardized error response.
        
        Args:
            error: Error message or exception
            status_code: HTTP status code
            error_code: Optional error code for client handling
            **kwargs: Additional error metadata
            
        Returns:
            Standardized error JSONResponse
        """
        error_message = str(error) if isinstance(error, Exception) else error
        
        # Use error handler to categorize and enhance error information
        error_info = error_handler.categorize_error(error)
        
        response_data = {
            "status": "error",
            "message": error_message,
            "error_type": error_info.get("type", "unknown"),
            "error_code": error_code or error_info.get("code"),
            "severity": error_info.get("severity", "medium"),
            **kwargs
        }
        
        return JSONResponse(
            status_code=status_code,
            content=response_data
        )
    
    def handle_request(
        self,
        method: str,
        path: str,
        handler_func,
        *args,
        **kwargs
    ) -> JSONResponse:
        """
        Generic request handler with logging, timing, and error handling.
        
        Args:
            method: HTTP method
            path: Request path
            handler_func: Function to handle the request
            *args: Arguments to pass to handler function
            **kwargs: Keyword arguments to pass to handler function
            
        Returns:
            JSONResponse from handler or error response
        """
        correlation_id = self._generate_correlation_id()
        start_time = time.time()
        
        try:
            # Log request start
            self._log_request(correlation_id, method, path)
            
            # Execute handler
            result = handler_func(*args, **kwargs)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log successful response
            self._log_response(correlation_id, 200, duration_ms)
            
            return result
            
        except HTTPException as e:
            # FastAPI HTTP exceptions are already formatted
            duration_ms = (time.time() - start_time) * 1000
            self._log_response(correlation_id, e.status_code, duration_ms)
            self._log_error(correlation_id, e)
            raise
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._log_error(correlation_id, e)
            
            # Create standardized error response
            return self.create_error_response(
                error=e,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def validate_pagination_params(
        self,
        limit: int,
        offset: int = 0,
        max_limit: int = 100
    ) -> tuple[int, int]:
        """
        Validate and sanitize pagination parameters.
        
        Args:
            limit: Requested limit
            offset: Requested offset
            max_limit: Maximum allowed limit
            
        Returns:
            Tuple of (sanitized_limit, sanitized_offset)
            
        Raises:
            HTTPException: If parameters are invalid
        """
        if limit < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be at least 1"
            )
        
        if limit > max_limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Limit cannot exceed {max_limit}"
            )
        
        if offset < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Offset must be non-negative"
            )
        
        return limit, offset
    
    def validate_date_range(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        max_days: int = 365
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Validate date range parameters.
        
        Args:
            start_date: Start date string (ISO format)
            end_date: End date string (ISO format)
            max_days: Maximum allowed date range in days
            
        Returns:
            Tuple of (validated_start_date, validated_end_date)
            
        Raises:
            HTTPException: If date range is invalid
        """
        if start_date and end_date:
            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                
                if start_dt >= end_dt:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Start date must be before end date"
                    )
                
                days_diff = (end_dt - start_dt).days
                if days_diff > max_days:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Date range cannot exceed {max_days} days"
                    )
                    
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid date format: {str(e)}"
                )
        
        return start_date, end_date
    
    def sanitize_string_input(self, value: str, max_length: int = 1000) -> str:
        """
        Sanitize string input for security.
        
        Args:
            value: Input string
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            HTTPException: If input is invalid
        """
        if not value:
            return ""
        
        # Trim whitespace
        sanitized = value.strip()
        
        # Check length
        if len(sanitized) > max_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input too long. Maximum {max_length} characters allowed."
            )
        
        # Basic XSS protection - remove script tags
        sanitized = sanitized.replace("<script>", "").replace("</script>", "")
        
        return sanitized
    
    def format_paginated_response(
        self,
        items: list,
        total_count: int,
        limit: int,
        offset: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Format a paginated response with metadata.
        
        Args:
            items: List of items
            total_count: Total number of items
            limit: Items per page
            offset: Offset from start
            **kwargs: Additional response metadata
            
        Returns:
            Formatted pagination response
        """
        total_pages = (total_count + limit - 1) // limit if limit > 0 else 0
        current_page = (offset // limit) + 1 if limit > 0 else 1
        
        return {
            "items": items,
            "pagination": {
                "total_count": total_count,
                "total_pages": total_pages,
                "current_page": current_page,
                "limit": limit,
                "offset": offset,
                "has_next": offset + limit < total_count,
                "has_previous": offset > 0
            },
            **kwargs
        }
