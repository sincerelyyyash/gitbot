from fastapi import APIRouter, Request, HTTPException, status, Depends
from fastapi.responses import PlainTextResponse, JSONResponse
from app.config import settings
from app.models.github import IssueCommentPayload, IssuesPayload, PushPayload
from app.services.rag_service import (
    handle_issue_comment_event, 
    handle_issue_event,
    handle_push_event,
    get_repository_collection_info,
    delete_repository_collection,
    list_all_repository_collections,
    refresh_repository_knowledge_base
)
import logging
import hmac
import hashlib
from typing import Optional

router = APIRouter()
logger = logging.getLogger("webhook")

def verify_webhook_signature(request_body: bytes, signature: str):
    if not settings.github_webhook_secret:
        raise ValueError("GITHUB_WEBHOOK_SECRET is not set.")
    if signature.startswith("sha256="):
        signature = signature[7:]
    mac = hmac.new(
        settings.github_webhook_secret.encode("utf-8"),
        msg=request_body,
        digestmod=hashlib.sha256,
    )
    if not hmac.compare_digest(mac.hexdigest(), signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook signature"
        )

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
            signature = request.headers.get("X-Hub-Signature-256")
            if not signature:
                logger.warning("Missing webhook signature")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Missing X-Hub-Signature-256 header"}
                )
            
            try:
                verify_webhook_signature(request_body, signature)
            except ValueError as e:
                logger.error(f"Signature verification failed: {str(e)}")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid webhook signature"}
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

        # Log successful payload parsing
        logger.info(f"Successfully parsed {event_type} event payload")
        logger.debug(f"Event action: {payload.get('action')}")
        
        # Handle different event types
        try:
            if event_type == "issue_comment":
                data = IssueCommentPayload(**payload)
                await handle_issue_comment_event(data)
            elif event_type == "issues":
                data = IssuesPayload(**payload)
                await handle_issue_event(data)
            elif event_type == "push":
                data = PushPayload(**payload)
                await handle_push_event(data)
            else:
                logger.info(f"Unhandled event type: {event_type}")
                return JSONResponse(
                    status_code=status.HTTP_202_ACCEPTED,
                    content={"detail": f"Event type {event_type} is not handled"}
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