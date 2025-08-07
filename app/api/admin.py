"""
Admin API Module

Provides admin endpoints for collection management and indexing.

"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse

from app.services import rag_service
from app.services.indexing_service import indexing_service
from .base import BaseAPI
from .auth import AuthManager


class AdminAPI(BaseAPI):
    """
    Admin API component for collection management and indexing.
    
    Provides endpoints for:
    - Collection management (list, info, refresh, delete)
    - Indexing management (status, queue, cancel)
    - Administrative operations
    """
    
    def __init__(self):
        """Initialize the admin API."""
        super().__init__("admin")
        self.auth_manager = AuthManager()
        self.router = APIRouter(prefix="/admin", tags=["admin"])
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup all admin routes."""
        # Collection management
        self.router.get("/collections")(self.list_collections)
        self.router.get("/collections/{repo_owner}/{repo_name}")(self.get_collection_info)
        self.router.post("/collections/{repo_owner}/{repo_name}/refresh")(self.refresh_collection)
        self.router.delete("/collections/{repo_owner}/{repo_name}")(self.delete_collection)
        
        # Indexing management
        self.router.get("/indexing/status")(self.get_indexing_status)
        self.router.get("/indexing/status/{repo_owner}/{repo_name}")(self.get_repo_indexing_status)
        self.router.post("/indexing/queue/{repo_owner}/{repo_name}")(self.queue_repository_indexing)
        self.router.delete("/indexing/cancel/{repo_owner}/{repo_name}")(self.cancel_repository_indexing)
    
    async def list_collections(self, admin_token: Optional[str] = None) -> JSONResponse:
        """List all repository collections."""
        self.auth_manager.verify_admin_access(admin_token)
        
        try:
            collections = await rag_service.list_all_repository_collections()
            return self.create_success_response({
                "collections": collections,
                "total_count": len(collections)
            })
        except Exception as e:
            self.logger.exception("Error listing collections")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list collections: {str(e)}"
            )
    
    async def get_collection_info(
        self, 
        repo_owner: str, 
        repo_name: str, 
        admin_token: Optional[str] = None
    ) -> JSONResponse:
        """Get information about a specific repository's collection."""
        self.auth_manager.verify_admin_access(admin_token)
        
        try:
            # Sanitize repository name
            repo_owner, repo_name = self.auth_manager.sanitize_repository_name(repo_owner, repo_name)
            repo_full_name = f"{repo_owner}/{repo_name}"
            
            info = await rag_service.get_repository_collection_info(repo_full_name)
            return self.create_success_response({
                "repository": repo_full_name,
                "collection_info": info
            })
        except Exception as e:
            self.logger.exception(f"Error getting collection info for {repo_owner}/{repo_name}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get collection info: {str(e)}"
            )
    
    async def refresh_collection(
        self, 
        repo_owner: str, 
        repo_name: str, 
        installation_id: int, 
        admin_token: Optional[str] = None
    ) -> JSONResponse:
        """Refresh a repository's collection by reindexing all content."""
        self.auth_manager.verify_admin_access(admin_token)
        
        try:
            # Sanitize repository name
            repo_owner, repo_name = self.auth_manager.sanitize_repository_name(repo_owner, repo_name)
            repo_full_name = f"{repo_owner}/{repo_name}"
            
            success = await rag_service.refresh_repository_knowledge_base(repo_full_name, installation_id)
            if success:
                return self.create_success_response({
                    "message": f"Successfully refreshed collection for {repo_full_name}",
                    "repository": repo_full_name
                })
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to refresh collection for {repo_full_name}"
                )
        except Exception as e:
            self.logger.exception(f"Error refreshing collection for {repo_owner}/{repo_name}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to refresh collection: {str(e)}"
            )
    
    async def delete_collection(
        self, 
        repo_owner: str, 
        repo_name: str, 
        admin_token: Optional[str] = None
    ) -> JSONResponse:
        """Delete a repository's collection."""
        self.auth_manager.verify_admin_access(admin_token)
        
        try:
            # Sanitize repository name
            repo_owner, repo_name = self.auth_manager.sanitize_repository_name(repo_owner, repo_name)
            repo_full_name = f"{repo_owner}/{repo_name}"
            
            success = await rag_service.delete_repository_collection(repo_full_name)
            if success:
                return self.create_success_response({
                    "message": f"Successfully deleted collection for {repo_full_name}",
                    "repository": repo_full_name
                })
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to delete collection for {repo_full_name}"
                )
        except Exception as e:
            self.logger.exception(f"Error deleting collection for {repo_owner}/{repo_name}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete collection: {str(e)}"
            )
    
    async def get_indexing_status(self, admin_token: Optional[str] = None) -> JSONResponse:
        """Get overall indexing status."""
        self.auth_manager.verify_admin_access(admin_token)
        
        try:
            status_data = await indexing_service.get_status()
            return self.create_success_response({
                "indexing": status_data
            })
        except Exception as e:
            self.logger.exception("Error getting indexing status")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting indexing status: {str(e)}"
            )
    
    async def get_repo_indexing_status(
        self, 
        repo_owner: str, 
        repo_name: str, 
        admin_token: Optional[str] = None
    ) -> JSONResponse:
        """Get indexing status for a specific repository."""
        self.auth_manager.verify_admin_access(admin_token)
        
        try:
            # Sanitize repository name
            repo_owner, repo_name = self.auth_manager.sanitize_repository_name(repo_owner, repo_name)
            repo_full_name = f"{repo_owner}/{repo_name}"
            
            status_data = await indexing_service.get_status(repo_full_name)
            
            if status_data is None:
                return self.create_success_response({
                    "status": "not_found",
                    "message": f"No indexing job found for {repo_full_name}"
                })
            
            return self.create_success_response({
                "repository": repo_full_name,
                "indexing": status_data
            })
        except Exception as e:
            self.logger.exception("Error getting repository indexing status")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting repository indexing status: {str(e)}"
            )
    
    async def queue_repository_indexing(
        self,
        repo_owner: str, 
        repo_name: str, 
        installation_id: int,
        priority: int = Query(1, ge=0, le=10),
        force_refresh: bool = Query(False),
        admin_token: Optional[str] = None
    ) -> JSONResponse:
        """Manually queue a repository for indexing."""
        self.auth_manager.verify_admin_access(admin_token)
        
        try:
            # Sanitize repository name
            repo_owner, repo_name = self.auth_manager.sanitize_repository_name(repo_owner, repo_name)
            repo_full_name = f"{repo_owner}/{repo_name}"
            
            added = await indexing_service.add_repository(
                repo_full_name=repo_full_name,
                installation_id=installation_id,
                priority=priority,
                force_refresh=force_refresh
            )
            
            if added:
                return self.create_success_response({
                    "message": f"Queued {repo_full_name} for indexing",
                    "repository": repo_full_name,
                    "priority": priority,
                    "force_refresh": force_refresh
                })
            else:
                return self.create_success_response({
                    "status": "already_queued",
                    "message": f"{repo_full_name} is already queued or being processed",
                    "repository": repo_full_name
                })
        except Exception as e:
            self.logger.exception("Error queuing repository for indexing")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error queuing repository for indexing: {str(e)}"
            )
    
    async def cancel_repository_indexing(
        self, 
        repo_owner: str, 
        repo_name: str, 
        admin_token: Optional[str] = None
    ) -> JSONResponse:
        """Cancel indexing for a repository."""
        self.auth_manager.verify_admin_access(admin_token)
        
        try:
            # Sanitize repository name
            repo_owner, repo_name = self.auth_manager.sanitize_repository_name(repo_owner, repo_name)
            repo_full_name = f"{repo_owner}/{repo_name}"
            
            cancelled = await indexing_service.cancel_indexing(repo_full_name)
            
            if cancelled:
                return self.create_success_response({
                    "message": f"Cancelled indexing for {repo_full_name}",
                    "repository": repo_full_name
                })
            else:
                return self.create_success_response({
                    "status": "not_found",
                    "message": f"No pending indexing job found for {repo_full_name}",
                    "repository": repo_full_name
                })
        except Exception as e:
            self.logger.exception("Error cancelling repository indexing")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error cancelling repository indexing: {str(e)}"
            )
