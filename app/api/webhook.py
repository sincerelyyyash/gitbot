"""
Webhook API Module

Provides webhook endpoints for handling GitHub events.

"""

import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import ValidationError

from app.config import settings
from app.models.github import (
    IssueCommentPayload, 
    IssuesPayload, 
    PushPayload, 
    InstallationPayload, 
    InstallationRepositoriesPayload,
    PullRequestPayload,
    PullRequestReviewPayload,
    PullRequestReviewCommentPayload
)
from app.services import (
    github_event_service,
    rag_service,
    repository_service
)
from app.services.indexing_service import indexing_service
from .base import BaseAPI
from .auth import AuthManager
from app.core.payload_validator import webhook_validator, payload_rate_limiter
from app.core.cache_manager import cache_manager
from app.core.async_utils import async_operation_context, retry_async


class WebhookAPI(BaseAPI):
    """
    Webhook API component for handling GitHub events.
    
    Provides endpoints for:
    - GitHub webhook event processing
    - Event validation and routing
    - Installation management
    - Repository indexing coordination
    """
    
    def __init__(self):
        """Initialize the webhook API."""
        super().__init__("webhook")
        self.auth_manager = AuthManager()
        self.router = APIRouter(prefix="/webhook", tags=["webhook"])
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup all webhook routes."""
        # Main webhook endpoint
        self.router.post("/webhook")(self.github_webhook)
        
        # Health check
        self.router.get("/health")(self.health_check)
    
    async def github_webhook(self, request: Request) -> JSONResponse:
        """
        Handle GitHub webhook events with enhanced performance and validation.
        
        Features:
        - Payload size validation and rate limiting
        - Caching for frequently accessed data
        - Async processing with proper error handling
        - Performance monitoring and metrics
        """
        async with async_operation_context("github_webhook"):
            try:
                # Extract event type
                event_type = request.headers.get("X-GitHub-Event")
                if not event_type:
                    self.logger.warning("Missing X-GitHub-Event header")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Missing X-GitHub-Event header"
                    )
                
                # Check payload size limits
                content_length = request.headers.get("content-length")
                if content_length:
                    payload_size = int(content_length)
                    if not await payload_rate_limiter.check_large_payload_limit(
                        payload_size, 
                        settings.large_payload_threshold
                    ):
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Too many large payloads. Please try again later."
                        )
                
                # Validate webhook signature
                signature = self.auth_manager.extract_webhook_signature(request)
                if not signature:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid webhook signature"
                    )
                
                # Validate and parse payload
                payload = await webhook_validator.validate_webhook_payload(request, event_type)
                
                # Verify webhook signature with validated payload
                if not self.auth_manager.verify_webhook_signature(
                    request.body(), signature
                ):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid webhook signature"
                    )
                
                # Process webhook event with caching and async processing
                result = await self._process_webhook_event(event_type, payload)
                
                return JSONResponse(
                    content={"status": "success", "message": "Webhook processed successfully"},
                    status_code=status.HTTP_200_OK
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Webhook processing failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
    
    async def _process_webhook_event(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process webhook event with caching and performance optimizations.
        
        Args:
            event_type: GitHub event type
            payload: Validated webhook payload
            
        Returns:
            Processing result
        """
        repo_full_name = payload.get("repository", {}).get("full_name")
        
        # Generate cache key for this event
        cache_key = cache_manager.generate_key(
            "webhook_event",
            event_type=event_type,
            repo=repo_full_name,
            event_id=payload.get("action", "unknown")
        )
        
        # Check cache for duplicate events
        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            self.logger.info(f"Duplicate webhook event detected, using cached result: {event_type}")
            return cached_result
        
        # Process event based on type
        try:
            if event_type == "issue_comment":
                result = await self._handle_issue_comment(payload)
            elif event_type == "issues":
                result = await self._handle_issue_event(payload)
            elif event_type == "push":
                result = await self._handle_push_event(payload)
            elif event_type == "installation":
                result = await self._handle_installation_event(payload)
            elif event_type == "installation_repositories":
                result = await self._handle_installation_repositories_event(payload)
            elif event_type == "pull_request":
                result = await self._handle_pull_request_event(payload)
            elif event_type == "pull_request_review":
                result = await self._handle_pull_request_review_event(payload)
            elif event_type == "pull_request_review_comment":
                result = await self._handle_pull_request_review_comment_event(payload)
            else:
                self.logger.info(f"Unhandled webhook event type: {event_type}")
                result = {"status": "ignored", "reason": "Unhandled event type"}
            
            # Cache the result for duplicate detection
            await cache_manager.set(cache_key, result, ttl_seconds=300)  # 5 minutes
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {event_type} event: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _handle_issue_comment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle issue comment events with caching and async processing."""
        repo_full_name = payload.get("repository", {}).get("full_name")
        issue_number = payload.get("issue", {}).get("number")
        comment_body = payload.get("comment", {}).get("body", "")
        
        # Check cache for similar queries
        query_hash = cache_manager.generate_key("rag_query", repo=repo_full_name, query=comment_body)
        cached_response = await cache_manager.get(query_hash)
        
        if cached_response:
            self.logger.info(f"Using cached response for issue comment query")
            return await self._post_cached_response(payload, cached_response)
        
        # Process with RAG service
        result = await retry_async(
            github_event_service.handle_issue_comment,
            max_retries=3,
            delay=1.0,
            payload=payload
        )
        
        # Cache the response
        if result and result.get("status") == "success":
            await cache_manager.set(query_hash, result, ttl_seconds=1800)  # 30 minutes
        
        return result
    
    async def _handle_issue_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle issue events with async processing."""
        return await retry_async(
            github_event_service.handle_issue_event,
            max_retries=3,
            delay=1.0,
            payload=payload
        )
    
    async def _handle_push_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle push events with indexing optimization."""
        repo_full_name = payload.get("repository", {}).get("full_name")
        installation_id = payload.get("installation", {}).get("id")
        
        # Queue repository for re-indexing
        if repo_full_name and installation_id:
            await indexing_service.queue_repository(
                repo_full_name=repo_full_name,
                installation_id=installation_id,
                force_refresh=True
            )
        
        return {"status": "success", "action": "queued_for_indexing"}
    
    async def _handle_installation_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle installation events."""
        return await retry_async(
            github_event_service.handle_installation_event,
            max_retries=3,
            delay=1.0,
            payload=payload
        )
    
    async def _handle_installation_repositories_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle installation repositories events."""
        return await retry_async(
            github_event_service.handle_installation_repositories_event,
            max_retries=3,
            delay=1.0,
            payload=payload
        )
    
    async def _handle_pull_request_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pull request events with async processing."""
        return await retry_async(
            github_event_service.handle_pull_request_event,
            max_retries=3,
            delay=1.0,
            payload=payload
        )
    
    async def _handle_pull_request_review_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pull request review events."""
        return await retry_async(
            github_event_service.handle_pull_request_review_event,
            max_retries=3,
            delay=1.0,
            payload=payload
        )
    
    async def _handle_pull_request_review_comment_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pull request review comment events."""
        return await retry_async(
            github_event_service.handle_pull_request_review_comment_event,
            max_retries=3,
            delay=1.0,
            payload=payload
        )
    
    async def _post_cached_response(self, payload: Dict[str, Any], cached_response: Dict[str, Any]) -> Dict[str, Any]:
        """Post cached response to GitHub."""
        try:
            # Extract necessary information from payload
            repo_full_name = payload.get("repository", {}).get("full_name")
            installation_id = payload.get("installation", {}).get("id")
            issue_number = payload.get("issue", {}).get("number")
            
            if not all([repo_full_name, installation_id, issue_number]):
                return {"status": "error", "error": "Missing required payload fields"}
            
            # Post the cached response
            from app.core.github_utils import post_issue_comment
            await post_issue_comment(
                repo_full_name=repo_full_name,
                issue_number=issue_number,
                comment=cached_response.get("response", "Cached response"),
                installation_id=installation_id
            )
            
            return {"status": "success", "source": "cache"}
            
        except Exception as e:
            self.logger.error(f"Error posting cached response: {e}")
            return {"status": "error", "error": str(e)}
    
    async def health_check(self) -> JSONResponse:
        """Enhanced health check with performance metrics."""
        try:
            # Basic health check
            health_status = {
                "status": "healthy",
                "service": "webhook",
                "timestamp": self.get_current_timestamp()
            }
            
            # Add performance metrics
            cache_stats = cache_manager.get_stats()
            payload_stats = webhook_validator.get_validation_stats()
            
            health_status.update({
                "cache": cache_stats,
                "payload_validation": payload_stats,
                "rate_limiting": {
                    "large_payloads": len(payload_rate_limiter._large_payloads)
                }
            })
            
            return JSONResponse(content=health_status)
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return JSONResponse(
                content={"status": "unhealthy", "error": str(e)},
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            ) 