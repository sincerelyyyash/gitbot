from fastapi import APIRouter, Request, HTTPException, status, Depends
from fastapi.responses import PlainTextResponse, JSONResponse
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
from app.services.rag_service import (
    handle_issue_comment_event, 
    handle_issue_event,
    handle_issue_event_enhanced,
    handle_push_event,
    handle_pr_opened_event,
    handle_pr_updated_event,
    handle_pr_closed_event,
    handle_pr_review_submitted_event,
    get_repository_collection_info,
    delete_repository_collection,
    list_all_repository_collections,
    refresh_repository_knowledge_base
)
from app.services.indexing_service import indexing_service
import logging
import hmac
import hashlib
from typing import Optional
from pydantic import ValidationError

router = APIRouter()
logger = logging.getLogger("webhook")

def verify_webhook_signature(request_body: bytes, signature: str):
    """Verify webhook signature using SHA256 or SHA1."""
    if not settings.github_webhook_secret:
        raise ValueError("GITHUB_WEBHOOK_SECRET is not set.")
        
    # Get the signature algorithm and hash
    if signature.startswith("sha256="):
        algorithm = hashlib.sha256
        sig = signature[7:]  # Remove 'sha256=' prefix
    elif signature.startswith("sha1="):
        algorithm = hashlib.sha1
        sig = signature[5:]  # Remove 'sha1=' prefix
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unsupported signature format"
        )
    
    # Calculate expected signature
    mac = hmac.new(
        settings.github_webhook_secret.encode("utf-8"),
        msg=request_body,
        digestmod=algorithm,
    )
    
    # Compare signatures using constant-time comparison
    if not hmac.compare_digest(mac.hexdigest(), sig):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature"
        )

async def handle_installation_event(payload: InstallationPayload):
    """Handle GitHub App installation events."""
    action = payload.action
    installation_id = payload.installation.id
    
    logger.info(f"Handling installation event: {action} for installation {installation_id}")
    
    try:
        if action == "created":
            # App was installed
            if payload.repositories:
                logger.info(f"App installed with access to {len(payload.repositories)} repositories")
                
                # Queue all repositories for indexing with high priority
                for repo in payload.repositories:
                    try:
                        await indexing_service.add_repository(
                            repo_full_name=repo.full_name,
                            installation_id=installation_id,
                            priority=0,  # Highest priority for new installations
                            force_refresh=False
                        )
                        logger.info(f"Queued {repo.full_name} for initial indexing")
                    except Exception as e:
                        logger.error(f"Failed to queue {repo.full_name} for indexing: {str(e)}")
            else:
                logger.info("App installed with repository selection - will wait for repository access events")
        
        elif action == "deleted":
            # App was uninstalled - clean up data if repositories are provided
            logger.info(f"App uninstalled from installation {installation_id}")
            
            if payload.repositories:
                logger.info(f"Cleaning up data for {len(payload.repositories)} repositories")
                for repo in payload.repositories:
                    try:
                        # Cancel any pending indexing
                        cancelled = await indexing_service.cancel_indexing(repo.full_name)
                        if cancelled:
                            logger.info(f"Cancelled pending indexing for {repo.full_name}")
                        
                        # Optionally delete the knowledge base
                        # Note: Commented out to preserve data in case of accidental uninstalls
                        # success = await delete_repository_collection(repo.full_name)
                        # if success:
                        #     logger.info(f"Deleted collection for {repo.full_name}")
                        
                    except Exception as e:
                        logger.error(f"Error cleaning up {repo.full_name}: {str(e)}")
            else:
                logger.info("App deletion event received without repository list - skipping cleanup")
        
        elif action == "suspend":
            logger.info(f"App suspended for installation {installation_id}")
        
        elif action == "unsuspend": 
            logger.info(f"App unsuspended for installation {installation_id}")
            
            # Re-queue repositories for indexing if available
            if payload.repositories:
                logger.info(f"Re-queuing {len(payload.repositories)} repositories after unsuspension")
                for repo in payload.repositories:
                    try:
                        await indexing_service.add_repository(
                            repo_full_name=repo.full_name,
                            installation_id=installation_id,
                            priority=1,  # High priority for unsuspended apps
                            force_refresh=False
                        )
                        logger.info(f"Re-queued {repo.full_name} for indexing after unsuspension")
                    except Exception as e:
                        logger.error(f"Failed to re-queue {repo.full_name} for indexing: {str(e)}")
        
        else:
            logger.info(f"Unhandled installation action: {action}")
            
    except Exception as e:
        logger.exception(f"Error handling installation event {action} for installation {installation_id}: {str(e)}")
        # Don't re-raise the exception to avoid webhook failure

async def handle_installation_repositories_event(payload: InstallationRepositoriesPayload):
    """Handle when repositories are added/removed from installation."""
    action = payload.action
    installation_id = payload.installation.id
    
    logger.info(f"Handling installation_repositories event: {action} for installation {installation_id}")
    
    try:
        if action == "added" and payload.repositories_added:
            logger.info(f"Repositories added: {len(payload.repositories_added)}")
            
            # Queue new repositories for indexing
            for repo in payload.repositories_added:
                try:
                    await indexing_service.add_repository(
                        repo_full_name=repo.full_name,
                        installation_id=installation_id,
                        priority=1,  # High priority for newly added repos
                        force_refresh=False
                    )
                    logger.info(f"Queued {repo.full_name} for indexing after being added to installation")
                except Exception as e:
                    logger.error(f"Failed to queue {repo.full_name} for indexing: {str(e)}")
        
        elif action == "removed" and payload.repositories_removed:
            logger.info(f"Repositories removed: {len(payload.repositories_removed)}")
            
            # Cancel indexing and optionally clean up data
            for repo in payload.repositories_removed:
                try:
                    # Cancel any pending indexing
                    cancelled = await indexing_service.cancel_indexing(repo.full_name)
                    if cancelled:
                        logger.info(f"Cancelled pending indexing for {repo.full_name}")
                    
                    # Optionally delete the knowledge base
                    # Note: You might want to keep it for some time in case repo is re-added
                    # await delete_repository_collection(repo.full_name)
                    logger.info(f"Repository {repo.full_name} removed from installation")
                except Exception as e:
                    logger.error(f"Error handling removal of {repo.full_name}: {str(e)}")
        
        else:
            logger.info(f"Unhandled installation_repositories action: {action}")
            
    except Exception as e:
        logger.exception(f"Error handling installation_repositories event {action} for installation {installation_id}: {str(e)}")
        # Don't re-raise the exception to avoid webhook failure

@router.post("/webhook")
async def github_webhook(request: Request):
    """Handle GitHub webhook events."""
    try:
        # Log request info
        headers = dict(request.headers)
        logger.info(f"Received webhook request from {request.client.host}")
        logger.debug(f"Headers: {headers}")
        
        # Get event type first - we want to log this even if other validations fail
        event_type = request.headers.get("X-GitHub-Event")
        if event_type:
            logger.info(f"GitHub event type: {event_type}")
        else:
            logger.warning("Missing X-GitHub-Event header")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Missing X-GitHub-Event header"}
            )

        # Read request body early so we can log issues
        try:
            request_body = await request.body()
            logger.debug(f"Received payload size: {len(request_body)} bytes")
        except Exception as e:
            logger.error(f"Failed to read request body: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Failed to read request body"}
            )

        # Verify signature if secret is configured
        if settings.github_webhook_secret:
            # Try SHA256 first, then fall back to SHA1
            signature_256 = request.headers.get("X-Hub-Signature-256")
            signature_1 = request.headers.get("X-Hub-Signature")
            signature = signature_256 or signature_1
            
            if not signature:
                logger.warning("Missing webhook signature")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Missing webhook signature"}
                )
            
            try:
                verify_webhook_signature(request_body, signature)
                logger.debug(f"Webhook signature verified successfully")
            except ValueError as e:
                logger.error(f"Signature verification failed: {str(e)}")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": str(e)}
                )
            except HTTPException as e:
                logger.error(f"Signature verification failed: {e.detail}")
                return JSONResponse(
                    status_code=e.status_code,
                    content={"detail": e.detail}
                )
        else:
            logger.warning("Webhook secret not configured - skipping signature verification")

        # Parse JSON payload
        try:
            payload = await request.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON payload: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid JSON payload"}
            )

        # Handle different event types
        try:
            if event_type == "ping":
                logger.info("Received ping event")
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={"detail": "Pong! Webhook received successfully"}
                )
            elif event_type == "issue_comment":
                try:
                    data = IssueCommentPayload(**payload)
                    if data.sender and data.sender.type == "Bot":
                        logger.info(f"Ignoring comment from bot: {data.sender.login}")
                        return JSONResponse(
                            status_code=status.HTTP_200_OK,
                            content={"detail": "Comment from bot ignored"}
                        )
                    await handle_issue_comment_event(data)
                except ValidationError as e:
                    logger.error(f"Validation error for issue_comment event: {str(e)}")
                    # Log the raw payload for debugging
                    logger.debug(f"Raw payload that failed validation: {payload}")
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": f"Invalid issue_comment payload: {str(e)}"}
                    )
            elif event_type == "issues":
                try:
                    data = IssuesPayload(**payload)
                    # Use enhanced handler for better issue similarity detection
                    await handle_issue_event_enhanced(data)
                except ValidationError as e:
                    logger.error(f"Validation error for issues event: {str(e)}")
                    logger.debug(f"Raw payload that failed validation: {payload}")
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": f"Invalid issues payload: {str(e)}"}
                    )
            elif event_type == "push":
                try:
                    data = PushPayload(**payload)
                    await handle_push_event(data)
                except ValidationError as e:
                    logger.error(f"Validation error for push event: {str(e)}")
                    logger.debug(f"Raw payload that failed validation: {payload}")
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": f"Invalid push payload: {str(e)}"}
                    )
            elif event_type == "installation":
                try:
                    data = InstallationPayload(**payload)
                    await handle_installation_event(data)
                except ValidationError as e:
                    logger.error(f"Validation error for installation event: {str(e)}")
                    logger.debug(f"Raw payload that failed validation: {payload}")
                    # For installation events, we want to be more graceful since GitHub might send incomplete data
                    logger.warning(f"Installation event validation failed, but continuing: {str(e)}")
                    return JSONResponse(
                        status_code=status.HTTP_200_OK,
                        content={"detail": f"Installation event processed with warnings: {str(e)}"}
                    )
            elif event_type == "installation_repositories":
                try:
                    data = InstallationRepositoriesPayload(**payload)
                    await handle_installation_repositories_event(data)
                except ValidationError as e:
                    logger.error(f"Validation error for installation_repositories event: {str(e)}")
                    logger.debug(f"Raw payload that failed validation: {payload}")
                    # For installation events, we want to be more graceful
                    logger.warning(f"Installation_repositories event validation failed, but continuing: {str(e)}")
                    return JSONResponse(
                        status_code=status.HTTP_200_OK,
                        content={"detail": f"Installation_repositories event processed with warnings: {str(e)}"}
                    )
            elif event_type == "pull_request":
                try:
                    data = PullRequestPayload(**payload)
                    
                    # Skip PRs created by bots (including this bot itself)
                    if data.sender and data.sender.type == "Bot":
                        logger.info(f"Ignoring pull request from bot: {data.sender.login}")
                        return JSONResponse(
                            status_code=status.HTTP_200_OK,
                            content={"detail": "Pull request from bot ignored"}
                        )
                    
                    action = data.action
                    if action == "opened":
                        await handle_pr_opened_event(data)
                        logger.info(f"Successfully processed PR opened event for #{data.pull_request.number}")
                    elif action in ["synchronize", "edited"]:
                        await handle_pr_updated_event(data)
                        logger.info(f"Successfully processed PR updated event for #{data.pull_request.number}")
                    elif action == "closed":
                        await handle_pr_closed_event(data)
                        logger.info(f"Successfully processed PR closed event for #{data.pull_request.number}")
                    else:
                        logger.info(f"Unhandled pull_request action: {action}")
                except ValidationError as e:
                    logger.error(f"Validation error for pull_request event: {str(e)}")
                    logger.debug(f"Raw payload that failed validation: {payload}")
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": f"Invalid pull_request payload: {str(e)}"}
                    )
                except Exception as e:
                    logger.exception(f"Error processing pull_request event: {str(e)}")
                    # Don't fail the webhook for processing errors
                    return JSONResponse(
                        status_code=status.HTTP_200_OK,
                        content={"detail": f"Pull request event received but processing failed: {str(e)}"}
                    )
            elif event_type == "pull_request_review":
                try:
                    data = PullRequestReviewPayload(**payload)
                    if data.action == "submitted":
                        await handle_pr_review_submitted_event(data)
                    else:
                        logger.info(f"Unhandled pull_request_review action: {data.action}")
                except ValidationError as e:
                    logger.error(f"Validation error for pull_request_review event: {str(e)}")
                    logger.debug(f"Raw payload that failed validation: {payload}")
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": f"Invalid pull_request_review payload: {str(e)}"}
                    )
            elif event_type == "pull_request_review_comment":
                try:
                    data = PullRequestReviewCommentPayload(**payload)
                    # For now, we'll just log the comment - detailed review analysis happens elsewhere
                    logger.info(f"Received pull request review comment from {data.sender.login if data.sender else 'unknown'} on PR #{data.pull_request.number}")
                except ValidationError as e:
                    logger.error(f"Validation error for pull_request_review_comment event: {str(e)}")
                    logger.debug(f"Raw payload that failed validation: {payload}")
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": f"Invalid pull_request_review_comment payload: {str(e)}"}
                    )
            else:
                logger.info(f"Unhandled event type: {event_type}")
                return JSONResponse(
                    status_code=status.HTTP_202_ACCEPTED,
                    content={"detail": f"Event type {event_type} is not handled"}
                )
        except ValidationError as e:
            logger.error(f"Payload validation error for {event_type}: {str(e)}")
            logger.debug(f"Raw payload that failed validation: {payload}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": f"Invalid payload format for {event_type}: {str(e)}"}
            )
        except Exception as e:
            logger.exception(f"Error processing {event_type} event")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Error processing webhook: {str(e)}"}
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"detail": f"Successfully processed {event_type} event"}
        )
        
    except Exception as e:
        logger.exception("Unexpected error in webhook handler")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Internal server error: {str(e)}"}
        )

# Admin endpoints for collection management
def verify_admin_access(admin_token: Optional[str] = None):
    """Simple admin token verification. In production, use proper authentication."""
    if not admin_token or admin_token != settings.admin_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Admin access required"
        )
    return True

@router.get("/admin/collections")
async def list_collections(admin_token: Optional[str] = None):
    """List all repository collections."""
    verify_admin_access(admin_token)
    
    try:
        collections = await list_all_repository_collections()
        return JSONResponse({
            "status": "success",
            "collections": collections,
            "total_count": len(collections)
        })
    except Exception as e:
        logger.exception("Error listing collections")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list collections: {str(e)}"
        )

@router.get("/admin/collections/{repo_owner}/{repo_name}")
async def get_collection_info(repo_owner: str, repo_name: str, admin_token: Optional[str] = None):
    """Get information about a specific repository's collection."""
    verify_admin_access(admin_token)
    
    repo_full_name = f"{repo_owner}/{repo_name}"
    
    try:
        info = await get_repository_collection_info(repo_full_name)
        return JSONResponse({
            "status": "success",
            "repository": repo_full_name,
            "collection_info": info
        })
    except Exception as e:
        logger.exception(f"Error getting collection info for {repo_full_name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collection info: {str(e)}"
        )

@router.post("/admin/collections/{repo_owner}/{repo_name}/refresh")
async def refresh_collection(repo_owner: str, repo_name: str, installation_id: int, admin_token: Optional[str] = None):
    """Refresh a repository's collection by reindexing all content."""
    verify_admin_access(admin_token)
    
    repo_full_name = f"{repo_owner}/{repo_name}"
    
    try:
        success = await refresh_repository_knowledge_base(repo_full_name, installation_id)
        if success:
            return JSONResponse({
                "status": "success",
                "message": f"Successfully refreshed collection for {repo_full_name}",
                "repository": repo_full_name
            })
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to refresh collection for {repo_full_name}"
            )
    except Exception as e:
        logger.exception(f"Error refreshing collection for {repo_full_name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh collection: {str(e)}"
        )

@router.delete("/admin/collections/{repo_owner}/{repo_name}")
async def delete_collection(repo_owner: str, repo_name: str, admin_token: Optional[str] = None):
    """Delete a repository's collection."""
    verify_admin_access(admin_token)
    
    repo_full_name = f"{repo_owner}/{repo_name}"
    
    try:
        success = await delete_repository_collection(repo_full_name)
        if success:
            return JSONResponse({
                "status": "success",
                "message": f"Successfully deleted collection for {repo_full_name}",
                "repository": repo_full_name
            })
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete collection for {repo_full_name}"
            )
    except Exception as e:
        logger.exception(f"Error deleting collection for {repo_full_name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete collection: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "gitbot",
        "version": "1.0.0"
    })

# Indexing management endpoints

@router.get("/admin/indexing/status")
async def get_indexing_status(admin_token: Optional[str] = None):
    """Get overall indexing status."""
    verify_admin_access(admin_token)
    
    try:
        status = await indexing_service.get_status()
        return JSONResponse({
            "status": "success",
            "indexing": status
        })
    except Exception as e:
        logger.exception("Error getting indexing status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting indexing status: {str(e)}"
        )

@router.get("/admin/indexing/status/{repo_owner}/{repo_name}")
async def get_repo_indexing_status(repo_owner: str, repo_name: str, admin_token: Optional[str] = None):
    """Get indexing status for a specific repository."""
    verify_admin_access(admin_token)
    
    try:
        repo_full_name = f"{repo_owner}/{repo_name}"
        status = await indexing_service.get_status(repo_full_name)
        
        if status is None:
            return JSONResponse({
                "status": "not_found",
                "message": f"No indexing job found for {repo_full_name}"
            })
        
        return JSONResponse({
            "status": "success",
            "repository": repo_full_name,
            "indexing": status
        })
    except Exception as e:
        logger.exception("Error getting repository indexing status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting repository indexing status: {str(e)}"
        )

@router.post("/admin/indexing/queue/{repo_owner}/{repo_name}")
async def queue_repository_indexing(
    repo_owner: str, 
    repo_name: str, 
    installation_id: int,
    priority: int = 1,
    force_refresh: bool = False,
    admin_token: Optional[str] = None
):
    """Manually queue a repository for indexing."""
    verify_admin_access(admin_token)
    
    try:
        repo_full_name = f"{repo_owner}/{repo_name}"
        added = await indexing_service.add_repository(
            repo_full_name=repo_full_name,
            installation_id=installation_id,
            priority=priority,
            force_refresh=force_refresh
        )
        
        if added:
            return JSONResponse({
                "status": "success",
                "message": f"Queued {repo_full_name} for indexing",
                "repository": repo_full_name,
                "priority": priority,
                "force_refresh": force_refresh
            })
        else:
            return JSONResponse({
                "status": "already_queued",
                "message": f"{repo_full_name} is already queued or being processed",
                "repository": repo_full_name
            })
    except Exception as e:
        logger.exception("Error queuing repository for indexing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error queuing repository for indexing: {str(e)}"
        )

@router.delete("/admin/indexing/cancel/{repo_owner}/{repo_name}")
async def cancel_repository_indexing(repo_owner: str, repo_name: str, admin_token: Optional[str] = None):
    """Cancel indexing for a repository."""
    verify_admin_access(admin_token)
    
    try:
        repo_full_name = f"{repo_owner}/{repo_name}"
        cancelled = await indexing_service.cancel_indexing(repo_full_name)
        
        if cancelled:
            return JSONResponse({
                "status": "success",
                "message": f"Cancelled indexing for {repo_full_name}",
                "repository": repo_full_name
            })
        else:
            return JSONResponse({
                "status": "not_found",
                "message": f"No pending indexing job found for {repo_full_name}",
                "repository": repo_full_name
            })
    except Exception as e:
        logger.exception("Error cancelling repository indexing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cancelling repository indexing: {str(e)}"
        ) 