"""
GitHub Authentication Manager

Provides GitHub App authentication and token management including:
- JWT token generation for GitHub App authentication
- Installation token management
- Token caching and refresh logic
- Authentication error handling
"""

import logging
import time
import jwt
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from github import Auth, GithubIntegration
from .base import BaseCore, core_operation
from app.config import settings

@dataclass
class TokenInfo:
    """Token information for caching and management."""
    token: str
    expires_at: datetime
    installation_id: int
    permissions: Dict[str, str]
    created_at: datetime

class GitHubAuthManager(BaseCore):
    """
    GitHub authentication manager for handling GitHub App authentication.
    
    Features:
    - JWT token generation for GitHub App authentication
    - Installation token management with caching
    - Token refresh logic
    - Authentication error handling
    - Token validation and health checks
    """
    
    def __init__(self):
        super().__init__("github_auth")
        
        # GitHub App configuration
        self.app_id = settings.github_app_id
        self.private_key = settings.github_private_key
        
        # Token cache
        self._jwt_cache: Optional[TokenInfo] = None
        self._installation_tokens: Dict[int, TokenInfo] = {}
        
        # Token configuration
        self.jwt_expiry_buffer = 60  # 1 minute buffer before expiry
        self.installation_token_expiry_buffer = 300  # 5 minutes buffer before expiry
        self.max_cache_size = 100
        
        # Authentication metrics
        self._auth_metrics = {
            "jwt_generations": 0,
            "installation_token_requests": 0,
            "token_refreshes": 0,
            "auth_failures": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Validate configuration
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate GitHub App configuration."""
        if not self.app_id:
            raise ValueError("GitHub App ID not configured")
        
        if not self.private_key:
            raise ValueError("GitHub App private key not configured")
        
        try:
            # Test JWT generation
            self._generate_jwt_token()
            self.logger.info("GitHub App configuration validated successfully")
        except Exception as error:
            self.logger.error(f"GitHub App configuration validation failed: {error}")
            raise
    
    @core_operation("generate_jwt_token")
    def _generate_jwt_token(self) -> str:
        """Generate a JWT token for GitHub App authentication."""
        try:
            now = int(time.time())
            payload = {
                "iat": now - 60,  # Issued 1 minute ago
                "exp": now + (10 * 60),  # Expires in 10 minutes
                "iss": self.app_id
            }
            
            encoded_jwt = jwt.encode(payload, self.private_key, algorithm="RS256")
            jwt_token = encoded_jwt if isinstance(encoded_jwt, str) else encoded_jwt.decode("utf-8")
            
            # Cache JWT token
            self._jwt_cache = TokenInfo(
                token=jwt_token,
                expires_at=datetime.fromtimestamp(now + (10 * 60)),
                installation_id=0,  # JWT doesn't have installation ID
                permissions={},
                created_at=datetime.utcnow()
            )
            
            self._auth_metrics["jwt_generations"] += 1
            self.logger.debug("JWT token generated successfully")
            
            return jwt_token
            
        except Exception as error:
            self._auth_metrics["auth_failures"] += 1
            self.logger.error(f"Failed to generate JWT token: {error}")
            raise
    
    def get_jwt_token(self) -> str:
        """Get a valid JWT token, generating a new one if needed."""
        # Check if cached JWT is still valid
        if (self._jwt_cache and 
            self._jwt_cache.expires_at > datetime.utcnow() + timedelta(seconds=self.jwt_expiry_buffer)):
            self._auth_metrics["cache_hits"] += 1
            return self._jwt_cache.token
        
        # Generate new JWT token
        self._auth_metrics["cache_misses"] += 1
        return self._generate_jwt_token()
    
    @core_operation("get_installation_token")
    async def get_installation_token(self, installation_id: int) -> str:
        """Get an installation access token for a specific installation."""
        # Check if cached token is still valid
        if installation_id in self._installation_tokens:
            cached_token = self._installation_tokens[installation_id]
            if cached_token.expires_at > datetime.utcnow() + timedelta(seconds=self.installation_token_expiry_buffer):
                self._auth_metrics["cache_hits"] += 1
                return cached_token.token
        
        # Get new installation token
        self._auth_metrics["cache_misses"] += 1
        return await self._request_installation_token(installation_id)
    
    @core_operation("request_installation_token")
    async def _request_installation_token(self, installation_id: int) -> str:
        """Request a new installation access token from GitHub."""
        try:
            # Create GitHub integration with JWT
            jwt_token = self.get_jwt_token()
            git_integration = GithubIntegration(
                auth=Auth.AppAuth(
                    app_id=self.app_id,
                    private_key=self.private_key
                )
            )
            
            # Get installation access token
            access_token_response = git_integration.get_access_token(installation_id)
            
            # Cache the token
            token_info = TokenInfo(
                token=access_token_response.token,
                expires_at=access_token_response.expires_at,
                installation_id=installation_id,
                permissions=access_token_response.permissions,
                created_at=datetime.utcnow()
            )
            
            self._installation_tokens[installation_id] = token_info
            self._auth_metrics["installation_token_requests"] += 1
            
            # Clean up cache if it's too large
            if len(self._installation_tokens) > self.max_cache_size:
                self._cleanup_cache()
            
            self.logger.debug(f"Installation token obtained for installation {installation_id}")
            return access_token_response.token
            
        except Exception as error:
            self._auth_metrics["auth_failures"] += 1
            self.logger.error(f"Failed to get installation token for {installation_id}: {error}")
            raise
    
    def _cleanup_cache(self):
        """Clean up expired tokens from cache."""
        now = datetime.utcnow()
        expired_installations = [
            installation_id for installation_id, token_info in self._installation_tokens.items()
            if token_info.expires_at <= now
        ]
        
        for installation_id in expired_installations:
            del self._installation_tokens[installation_id]
        
        if expired_installations:
            self.logger.debug(f"Cleaned up {len(expired_installations)} expired installation tokens")
    
    @core_operation("validate_token")
    async def validate_token(self, token: str, installation_id: Optional[int] = None) -> bool:
        """Validate if a token is still valid."""
        try:
            # Check if token is in cache
            if installation_id and installation_id in self._installation_tokens:
                cached_token = self._installation_tokens[installation_id]
                if cached_token.token == token:
                    return cached_token.expires_at > datetime.utcnow()
            
            # For JWT tokens, check if it's our cached JWT
            if self._jwt_cache and self._jwt_cache.token == token:
                return self._jwt_cache.expires_at > datetime.utcnow()
            
            # If not in cache, we can't validate it (would require API call)
            return False
            
        except Exception as error:
            self.logger.error(f"Token validation failed: {error}")
            return False
    
    @core_operation("refresh_installation_token")
    async def refresh_installation_token(self, installation_id: int) -> str:
        """Force refresh of an installation token."""
        # Remove from cache to force refresh
        if installation_id in self._installation_tokens:
            del self._installation_tokens[installation_id]
        
        self._auth_metrics["token_refreshes"] += 1
        return await self.get_installation_token(installation_id)
    
    def get_token_info(self, installation_id: int) -> Optional[TokenInfo]:
        """Get information about a cached installation token."""
        return self._installation_tokens.get(installation_id)
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status and statistics."""
        now = datetime.utcnow()
        
        # Count valid and expired tokens
        valid_tokens = 0
        expired_tokens = 0
        
        for token_info in self._installation_tokens.values():
            if token_info.expires_at > now:
                valid_tokens += 1
            else:
                expired_tokens += 1
        
        return {
            "jwt_token_cached": self._jwt_cache is not None,
            "jwt_token_expires_at": self._jwt_cache.expires_at.isoformat() if self._jwt_cache else None,
            "installation_tokens_cached": len(self._installation_tokens),
            "valid_installation_tokens": valid_tokens,
            "expired_installation_tokens": expired_tokens,
            "cache_size_limit": self.max_cache_size
        }
    
    async def _basic_health_check(self):
        """Basic health check for authentication manager."""
        try:
            # Test JWT generation
            jwt_token = self.get_jwt_token()
            if not jwt_token:
                raise Exception("JWT token generation failed")
            
            # Test GitHub integration creation
            git_integration = GithubIntegration(
                auth=Auth.AppAuth(
                    app_id=self.app_id,
                    private_key=self.private_key
                )
            )
            
            # Test basic API call
            app_info = git_integration.get_app()
            if not app_info:
                raise Exception("Failed to get app information")
            
        except Exception as error:
            raise Exception(f"Authentication health check failed: {error}")
    
    def get_auth_metrics(self) -> Dict[str, Any]:
        """Get authentication metrics."""
        return {
            "component": self.component_name,
            "app_id": self.app_id,
            "auth_metrics": self._auth_metrics.copy(),
            "cache_status": self.get_cache_status()
        }
    
    async def reset_metrics(self):
        """Reset authentication metrics."""
        self._auth_metrics = {
            "jwt_generations": 0,
            "installation_token_requests": 0,
            "token_refreshes": 0,
            "auth_failures": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self.logger.info("Authentication metrics reset")
    
    def clear_cache(self):
        """Clear all cached tokens."""
        self._jwt_cache = None
        self._installation_tokens.clear()
        self.logger.info("Authentication cache cleared")
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get authentication configuration."""
        config = super().get_configuration()
        config.update({
            "app_id": self.app_id,
            "private_key_configured": bool(self.private_key),
            "jwt_expiry_buffer": self.jwt_expiry_buffer,
            "installation_token_expiry_buffer": self.installation_token_expiry_buffer,
            "max_cache_size": self.max_cache_size
        })
        return config

# Legacy compatibility functions removed - use GitHubAuthManager instead
