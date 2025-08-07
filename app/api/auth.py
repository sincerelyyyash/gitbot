"""
Authentication and Authorization Module

Provides centralized authentication and authorization functionality for the API layer.
Handles admin token verification, webhook signature validation, and security utilities.
"""

import hmac
import hashlib
from typing import Optional
from fastapi import HTTPException, status, Request
from fastapi.responses import JSONResponse

from app.config import settings
from .base import BaseAPI


class AuthManager(BaseAPI):
    """
    Centralized authentication and authorization manager.
    
    Handles:
    - Admin token verification
    - Webhook signature validation
    - Security utilities and validation
    - Rate limiting integration
    """
    
    def __init__(self):
        """Initialize the authentication manager."""
        super().__init__("auth")
    
    def verify_admin_access(self, admin_token: Optional[str] = None) -> bool:
        """
        Verify admin access using admin token.
        
        Args:
            admin_token: Admin token from request
            
        Returns:
            True if access is granted
            
        Raises:
            HTTPException: If access is denied
        """
        if not settings.admin_token:
            self.logger.warning("Admin token not configured - admin access disabled")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access not configured"
            )
        
        if not admin_token or admin_token != settings.admin_token:
            self.logger.warning("Invalid admin token provided")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid admin token"
            )
        
        self.logger.debug("Admin access verified successfully")
        return True
    
    def verify_webhook_signature(
        self, 
        request_body: bytes, 
        signature: str
    ) -> bool:
        """
        Verify GitHub webhook signature using SHA256 or SHA1.
        
        Args:
            request_body: Raw request body bytes
            signature: Signature header from GitHub
            
        Returns:
            True if signature is valid
            
        Raises:
            HTTPException: If signature verification fails
        """
        if not settings.github_webhook_secret:
            self.logger.error("GITHUB_WEBHOOK_SECRET is not configured")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Webhook secret not configured"
            )
        
        # Determine signature algorithm
        if signature.startswith("sha256="):
            algorithm = hashlib.sha256
            sig = signature[7:]  # Remove 'sha256=' prefix
        elif signature.startswith("sha1="):
            algorithm = hashlib.sha1
            sig = signature[5:]  # Remove 'sha1=' prefix
        else:
            self.logger.warning(f"Unsupported signature format: {signature[:20]}...")
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
            self.logger.warning("Webhook signature verification failed")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature"
            )
        
        self.logger.debug("Webhook signature verified successfully")
        return True
    
    def extract_webhook_signature(self, request: Request) -> Optional[str]:
        """
        Extract webhook signature from request headers.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Signature string or None if not found
        """
        # Try SHA256 first, then fall back to SHA1
        signature_256 = request.headers.get("X-Hub-Signature-256")
        signature_1 = request.headers.get("X-Hub-Signature")
        
        signature = signature_256 or signature_1
        
        if not signature:
            self.logger.warning("No webhook signature found in headers")
            return None
        
        return signature
    
    async def validate_webhook_request(self, request: Request) -> tuple[bytes, str]:
        """
        Validate webhook request and extract necessary data.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Tuple of (request_body, event_type)
            
        Raises:
            HTTPException: If validation fails
        """
        # Extract event type
        event_type = request.headers.get("X-GitHub-Event")
        if not event_type:
            self.logger.warning("Missing X-GitHub-Event header")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing X-GitHub-Event header"
            )
        
        # Read request body
        try:
            request_body = await request.body()
            self.logger.debug(f"Received webhook payload: {len(request_body)} bytes")
        except Exception as e:
            self.logger.error(f"Failed to read request body: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to read request body"
            )
        
        # Verify signature if secret is configured
        if settings.github_webhook_secret:
            signature = self.extract_webhook_signature(request)
            if not signature:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing webhook signature"
                )
            
            try:
                self.verify_webhook_signature(request_body, signature)
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error during signature verification: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Signature verification failed"
                )
        else:
            self.logger.warning("Webhook secret not configured - skipping signature verification")
        
        return request_body, event_type
    
    def is_bot_user(self, sender_data: Optional[dict]) -> bool:
        """
        Check if the sender is a bot user.
        
        Args:
            sender_data: Sender data from GitHub payload
            
        Returns:
            True if sender is a bot
        """
        if not sender_data:
            return False
        
        return sender_data.get("type") == "Bot"
    
    def should_ignore_bot_action(self, sender_data: Optional[dict]) -> bool:
        """
        Determine if a bot action should be ignored.
        
        Args:
            sender_data: Sender data from GitHub payload
            
        Returns:
            True if the action should be ignored
        """
        if not self.is_bot_user(sender_data):
            return False
        
        # Ignore actions from our own bot to prevent loops
        bot_login = sender_data.get("login", "").lower()
        if "gitbot" in bot_login or "github-app" in bot_login:
            self.logger.info(f"Ignoring action from bot: {bot_login}")
            return True
        
        return False
    
    def validate_repository_access(
        self, 
        repo_full_name: str, 
        installation_id: int
    ) -> bool:
        """
        Validate that the installation has access to the repository.
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            installation_id: GitHub App installation ID
            
        Returns:
            True if access is valid
            
        Raises:
            HTTPException: If access is invalid
        """
        if not repo_full_name or "/" not in repo_full_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid repository format"
            )
        
        if not installation_id or installation_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid installation ID"
            )
        
        # TODO: Add actual repository access validation
        # This would check if the installation has access to the specific repository
        
        self.logger.debug(f"Repository access validated: {repo_full_name} (installation {installation_id})")
        return True
    
    def sanitize_repository_name(self, owner: str, repo: str) -> tuple[str, str]:
        """
        Sanitize repository owner and name.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Tuple of (sanitized_owner, sanitized_repo)
            
        Raises:
            HTTPException: If repository name is invalid
        """
        owner = self.sanitize_string_input(owner, max_length=100)
        repo = self.sanitize_string_input(repo, max_length=100)
        
        if not owner or not repo:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Repository owner and name are required"
            )
        
        # Basic validation for GitHub repository naming conventions
        if not owner.replace("-", "").replace("_", "").isalnum():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid repository owner format"
            )
        
        if not repo.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid repository name format"
            )
        
        return owner, repo
    
    def create_unauthorized_response(self, message: str = "Unauthorized") -> JSONResponse:
        """
        Create a standardized unauthorized response.
        
        Args:
            message: Unauthorized message
            
        Returns:
            JSONResponse with 401 status
        """
        return self.create_error_response(
            error=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="UNAUTHORIZED"
        )
    
    def create_forbidden_response(self, message: str = "Forbidden") -> JSONResponse:
        """
        Create a standardized forbidden response.
        
        Args:
            message: Forbidden message
            
        Returns:
            JSONResponse with 403 status
        """
        return self.create_error_response(
            error=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="FORBIDDEN"
        )
