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
        """Handle GitHub webhook events."""
        try:
            # Log request info
            headers = dict(request.headers)
            self.logger.info(f"Received webhook request from {request.client.host}")
            self.logger.debug(f"Headers: {headers}")
            
            # Validate webhook request
            request_body, event_type = await self.auth_manager.validate_webhook_request(request)
            
            # Parse JSON payload
            try:
                payload = await request.json()
            except Exception as e:
                self.logger.error(f"Failed to parse JSON payload: {str(e)}")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid JSON payload"}
                )
            
            # Handle different event types
            try:
                if event_type == "ping":
                    self.logger.info("Received ping event")
                    return JSONResponse(
                        status_code=status.HTTP_200_OK,
                        content={"detail": "Pong! Webhook received successfully"}
                    )
                elif event_type == "issue_comment":
                    return await self._handle_issue_comment(payload)
                elif event_type == "issues":
                    return await self._handle_issues(payload)
                elif event_type == "push":
                    return await self._handle_push(payload)
                elif event_type == "installation":
                    return await self._handle_installation(payload)
                elif event_type == "installation_repositories":
                    return await self._handle_installation_repositories(payload)
                elif event_type == "pull_request":
                    return await self._handle_pull_request(payload)
                elif event_type == "pull_request_review":
                    return await self._handle_pull_request_review(payload)
                elif event_type == "pull_request_review_comment":
                    return await self._handle_pull_request_review_comment(payload)
                else:
                    self.logger.info(f"Unhandled event type: {event_type}")
                    return JSONResponse(
                        status_code=status.HTTP_202_ACCEPTED,
                        content={"detail": f"Event type {event_type} is not handled"}
                    )
            except ValidationError as e:
                self.logger.error(f"Payload validation error for {event_type}: {str(e)}")
                self.logger.debug(f"Raw payload that failed validation: {payload}")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": f"Invalid payload format for {event_type}: {str(e)}"}
                )
            except Exception as e:
                self.logger.exception(f"Error processing {event_type} event")
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"detail": f"Error processing webhook: {str(e)}"}
                )
            
        except Exception as e:
            self.logger.exception("Unexpected error in webhook handler")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Internal server error: {str(e)}"}
            )
    
    async def _handle_issue_comment(self, payload: Dict[str, Any]) -> JSONResponse:
        """Handle issue comment events."""
        try:
            data = IssueCommentPayload(**payload)
            
            # Check if comment is from a bot
            if self.auth_manager.should_ignore_bot_action(data.sender.dict() if data.sender else None):
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={"detail": "Comment from bot ignored"}
                )
            
            await github_event_service.process_event(
                event_type="issue_comment",
                event_action=data.action if hasattr(data, 'action') else None,
                payload=data.dict(),
                repo_full_name=data.repository.full_name,
                installation_id=data.installation.id
            )
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"detail": "Successfully processed issue_comment event"}
            )
            
        except ValidationError as e:
            self.logger.error(f"Validation error for issue_comment event: {str(e)}")
            self.logger.debug(f"Raw payload that failed validation: {payload}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": f"Invalid issue_comment payload: {str(e)}"}
            )
    
    async def _handle_issues(self, payload: Dict[str, Any]) -> JSONResponse:
        """Handle issues events."""
        try:
            data = IssuesPayload(**payload)
            await github_event_service.process_event(
                event_type="issues",
                event_action=data.action,
                payload=data.dict(),
                repo_full_name=data.repository.full_name,
                installation_id=data.installation.id
            )
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"detail": "Successfully processed issues event"}
            )
            
        except ValidationError as e:
            self.logger.error(f"Validation error for issues event: {str(e)}")
            self.logger.debug(f"Raw payload that failed validation: {payload}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": f"Invalid issues payload: {str(e)}"}
            )
    
    async def _handle_push(self, payload: Dict[str, Any]) -> JSONResponse:
        """Handle push events."""
        try:
            data = PushPayload(**payload)
            await github_event_service.process_event(
                event_type="push",
                event_action=None,
                payload=data.dict(),
                repo_full_name=data.repository.full_name,
                installation_id=data.installation.id
            )
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"detail": "Successfully processed push event"}
            )
            
        except ValidationError as e:
            self.logger.error(f"Validation error for push event: {str(e)}")
            self.logger.debug(f"Raw payload that failed validation: {payload}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": f"Invalid push payload: {str(e)}"}
            )
    
    async def _handle_installation(self, payload: Dict[str, Any]) -> JSONResponse:
        """Handle installation events."""
        try:
            data = InstallationPayload(**payload)
            await self._process_installation_event(data)
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"detail": "Successfully processed installation event"}
            )
            
        except ValidationError as e:
            self.logger.error(f"Validation error for installation event: {str(e)}")
            self.logger.debug(f"Raw payload that failed validation: {payload}")
            # For installation events, we want to be more graceful
            self.logger.warning(f"Installation event validation failed, but continuing: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"detail": f"Installation event processed with warnings: {str(e)}"}
            )
    
    async def _handle_installation_repositories(self, payload: Dict[str, Any]) -> JSONResponse:
        """Handle installation repositories events."""
        try:
            data = InstallationRepositoriesPayload(**payload)
            await self._process_installation_repositories_event(data)
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"detail": "Successfully processed installation_repositories event"}
            )
            
        except ValidationError as e:
            self.logger.error(f"Validation error for installation_repositories event: {str(e)}")
            self.logger.debug(f"Raw payload that failed validation: {payload}")
            # For installation events, we want to be more graceful
            self.logger.warning(f"Installation_repositories event validation failed, but continuing: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"detail": f"Installation_repositories event processed with warnings: {str(e)}"}
            )
    
    async def _handle_pull_request(self, payload: Dict[str, Any]) -> JSONResponse:
        """Handle pull request events."""
        try:
            data = PullRequestPayload(**payload)
            
            # Skip PRs created by bots
            if self.auth_manager.should_ignore_bot_action(data.sender.dict() if data.sender else None):
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={"detail": "Pull request from bot ignored"}
                )
            
            action = data.action
            await github_event_service.process_event(
                event_type="pull_request",
                event_action=action,
                payload=data.dict(),
                repo_full_name=data.repository.full_name,
                installation_id=data.installation.id
            )
            
            self.logger.info(f"Successfully processed PR {action} event for #{data.pull_request.number}")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"detail": "Successfully processed pull_request event"}
            )
            
        except ValidationError as e:
            self.logger.error(f"Validation error for pull_request event: {str(e)}")
            self.logger.debug(f"Raw payload that failed validation: {payload}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": f"Invalid pull_request payload: {str(e)}"}
            )
        except Exception as e:
            self.logger.exception(f"Error processing pull_request event: {str(e)}")
            # Don't fail the webhook for processing errors
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"detail": f"Pull request event received but processing failed: {str(e)}"}
            )
    
    async def _handle_pull_request_review(self, payload: Dict[str, Any]) -> JSONResponse:
        """Handle pull request review events."""
        try:
            data = PullRequestReviewPayload(**payload)
            await github_event_service.process_event(
                event_type="pull_request_review",
                event_action=data.action,
                payload=data.dict(),
                repo_full_name=data.repository.full_name,
                installation_id=data.installation.id
            )
            
            self.logger.info(f"Successfully processed PR review {data.action} event")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"detail": "Successfully processed pull_request_review event"}
            )
            
        except ValidationError as e:
            self.logger.error(f"Validation error for pull_request_review event: {str(e)}")
            self.logger.debug(f"Raw payload that failed validation: {payload}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": f"Invalid pull_request_review payload: {str(e)}"}
            )
    
    async def _handle_pull_request_review_comment(self, payload: Dict[str, Any]) -> JSONResponse:
        """Handle pull request review comment events."""
        try:
            data = PullRequestReviewCommentPayload(**payload)
            # For now, we'll just log the comment - detailed review analysis happens elsewhere
            self.logger.info(f"Received pull request review comment from {data.sender.login if data.sender else 'unknown'} on PR #{data.pull_request.number}")
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"detail": "Successfully processed pull_request_review_comment event"}
            )
            
        except ValidationError as e:
            self.logger.error(f"Validation error for pull_request_review_comment event: {str(e)}")
            self.logger.debug(f"Raw payload that failed validation: {payload}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": f"Invalid pull_request_review_comment payload: {str(e)}"}
            )
    
    async def _process_installation_event(self, payload: InstallationPayload):
        """Process installation event."""
        action = payload.action
        installation_id = payload.installation.id
        
        self.logger.info(f"Handling installation event: {action} for installation {installation_id}")
        
        try:
            if action == "created":
                # App was installed
                if payload.repositories:
                    self.logger.info(f"App installed with access to {len(payload.repositories)} repositories")
                    
                    # Queue all repositories for indexing with high priority
                    for repo in payload.repositories:
                        try:
                            await indexing_service.add_repository(
                                repo_full_name=repo.full_name,
                                installation_id=installation_id,
                                priority=0,  # Highest priority for new installations
                                force_refresh=False
                            )
                            self.logger.info(f"Queued {repo.full_name} for initial indexing")
                        except Exception as e:
                            self.logger.error(f"Failed to queue {repo.full_name} for indexing: {str(e)}")
                else:
                    self.logger.info("App installed with repository selection - will wait for repository access events")
            
            elif action == "deleted":
                # App was uninstalled - clean up data if repositories are provided
                self.logger.info(f"App uninstalled from installation {installation_id}")
                
                if payload.repositories:
                    self.logger.info(f"Cleaning up data for {len(payload.repositories)} repositories")
                    for repo in payload.repositories:
                        try:
                            # Cancel any pending indexing
                            cancelled = await indexing_service.cancel_indexing(repo.full_name)
                            if cancelled:
                                self.logger.info(f"Cancelled pending indexing for {repo.full_name}")
                            
                            # Optionally delete the knowledge base
                            # Note: Commented out to preserve data in case of accidental uninstalls
                            # success = await delete_repository_collection(repo.full_name)
                            # if success:
                            #     self.logger.info(f"Deleted collection for {repo.full_name}")
                            
                        except Exception as e:
                            self.logger.error(f"Error cleaning up {repo.full_name}: {str(e)}")
                else:
                    self.logger.info("App deletion event received without repository list - skipping cleanup")
            
            elif action == "suspend":
                self.logger.info(f"App suspended for installation {installation_id}")
            
            elif action == "unsuspend": 
                self.logger.info(f"App unsuspended for installation {installation_id}")
                
                # Re-queue repositories for indexing if available
                if payload.repositories:
                    self.logger.info(f"Re-queuing {len(payload.repositories)} repositories after unsuspension")
                    for repo in payload.repositories:
                        try:
                            await indexing_service.add_repository(
                                repo_full_name=repo.full_name,
                                installation_id=installation_id,
                                priority=1,  # High priority for unsuspended apps
                                force_refresh=False
                            )
                            self.logger.info(f"Re-queued {repo.full_name} for indexing after unsuspension")
                        except Exception as e:
                            self.logger.error(f"Failed to re-queue {repo.full_name} for indexing: {str(e)}")
            
            else:
                self.logger.info(f"Unhandled installation action: {action}")
                
        except Exception as e:
            self.logger.exception(f"Error handling installation event {action} for installation {installation_id}: {str(e)}")
            # Don't re-raise the exception to avoid webhook failure
    
    async def _process_installation_repositories_event(self, payload: InstallationRepositoriesPayload):
        """Process installation repositories event."""
        action = payload.action
        installation_id = payload.installation.id
        
        self.logger.info(f"Handling installation_repositories event: {action} for installation {installation_id}")
        
        try:
            if action == "added" and payload.repositories_added:
                self.logger.info(f"Repositories added: {len(payload.repositories_added)}")
                
                # Queue new repositories for indexing
                for repo in payload.repositories_added:
                    try:
                        await indexing_service.add_repository(
                            repo_full_name=repo.full_name,
                            installation_id=installation_id,
                            priority=1,  # High priority for newly added repos
                            force_refresh=False
                        )
                        self.logger.info(f"Queued {repo.full_name} for indexing after being added to installation")
                    except Exception as e:
                        self.logger.error(f"Failed to queue {repo.full_name} for indexing: {str(e)}")
            
            elif action == "removed" and payload.repositories_removed:
                self.logger.info(f"Repositories removed: {len(payload.repositories_removed)}")
                
                # Cancel indexing and optionally clean up data
                for repo in payload.repositories_removed:
                    try:
                        # Cancel any pending indexing
                        cancelled = await indexing_service.cancel_indexing(repo.full_name)
                        if cancelled:
                            self.logger.info(f"Cancelled pending indexing for {repo.full_name}")
                        
                        # Optionally delete the knowledge base
                        # Note: You might want to keep it for some time in case repo is re-added
                        # await delete_repository_collection(repo.full_name)
                        self.logger.info(f"Repository {repo.full_name} removed from installation")
                    except Exception as e:
                        self.logger.error(f"Error handling removal of {repo.full_name}: {str(e)}")
            
            else:
                self.logger.info(f"Unhandled installation_repositories action: {action}")
                
        except Exception as e:
            self.logger.exception(f"Error handling installation_repositories event {action} for installation {installation_id}: {str(e)}")
            # Don't re-raise the exception to avoid webhook failure
    
    async def health_check(self) -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse({
            "status": "healthy",
            "service": "gitbot",
            "version": "1.0.0"
        }) 