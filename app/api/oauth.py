"""
OAuth API Module

Provides GitHub OAuth authentication endpoints for user login and authorization.
Handles OAuth flow, token management, and user session creation.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.auth_service import AuthService, get_current_user, get_current_active_user
from app.models.user import (
    GitHubOAuthRequest, Token, LoginResponse, UserResponse, UserUpdate, User, GitHubOAuthResponse
)

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Initialize services
auth_service = AuthService()


@router.get("/login")
async def login(state: Optional[str] = None):
    """
    Initiate GitHub OAuth login flow.
    
    Args:
        state: Optional state parameter for CSRF protection
        
    Returns:
        Redirect to GitHub OAuth authorization URL
    """
    try:
        oauth_url = auth_service.generate_github_oauth_url(state)
        return RedirectResponse(url=oauth_url)
    except Exception as e:
        auth_service.logger.error(f"Failed to generate OAuth URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate OAuth login"
        )


@router.get("/callback")
async def oauth_callback(
    code: str,
    state: Optional[str] = None,
    session: AsyncSession = Depends(get_db)
):
    """
    Handle GitHub OAuth callback and complete authentication.
    
    Args:
        code: OAuth authorization code from GitHub
        state: State parameter for CSRF protection
        session: Database session
        
    Returns:
        Login response with user data and tokens
    """
    try:
        # Exchange code for token
        oauth_response = await auth_service.exchange_oauth_code_for_token(code, state)
        
        # Fetch GitHub user data
        github_user_data = await auth_service.fetch_github_user_data(oauth_response.access_token)
        
        # Create or update user in database
        user = await auth_service.create_or_update_user(session, github_user_data)
        
        # Create user session with JWT tokens
        tokens = await auth_service.create_user_session(session, user)
        
        # Store GitHub token
        await auth_service.store_github_token(session, user.id, oauth_response)
        
        # Check if this is a new user
        is_new_user = user.created_at == user.updated_at
        
        auth_service.logger.info(f"Successful OAuth login for user: {user.github_login}")
        
        return LoginResponse(
            user=UserResponse.from_orm(user),
            tokens=tokens,
            is_new_user=is_new_user
        )
        
    except HTTPException:
        raise
    except Exception as e:
        auth_service.logger.error(f"OAuth callback error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@router.post("/refresh")
async def refresh_token(
    refresh_token: str,
    session: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    
    Args:
        refresh_token: Refresh token string
        session: Database session
        
    Returns:
        New token pair
    """
    try:
        tokens = await auth_service.refresh_access_token(session, refresh_token)
        return tokens
    except HTTPException:
        raise
    except Exception as e:
        auth_service.logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
    request: Request = None
):
    """
    Logout user and invalidate session.
    
    Args:
        current_user: Current authenticated user
        session: Database session
        request: HTTP request object
        
    Returns:
        Success message
    """
    try:
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            await auth_service.logout_user(session, current_user.id, token)
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        auth_service.logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me")
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current user information.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current user data
    """
    return UserResponse.from_orm(current_user)


@router.put("/me")
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Update current user information.
    
    Args:
        user_update: User update data
        current_user: Current authenticated user
        session: Database session
        
    Returns:
        Updated user data
    """
    try:
        updated_user = await auth_service.update_user(session, current_user.id, user_update)
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse.from_orm(updated_user)
        
    except HTTPException:
        raise
    except Exception as e:
        auth_service.logger.error(f"User update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.get("/validate")
async def validate_token(
    current_user: User = Depends(get_current_active_user)
):
    """
    Validate current authentication token.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Token validation status
    """
    return {
        "valid": True,
        "user": UserResponse.from_orm(current_user)
    }


# Add missing method to AuthService
async def store_github_token(
    self,
    session: AsyncSession,
    user_id: int,
    oauth_response: GitHubOAuthResponse
):
    """
    Store GitHub OAuth token in database.
    
    Args:
        session: Database session
        user_id: User ID
        oauth_response: GitHub OAuth response
    """
    from app.models.user import GitHubToken
    from datetime import datetime, timedelta
    
    # Calculate expiration time if provided
    expires_at = None
    if oauth_response.expires_in:
        expires_at = datetime.utcnow() + timedelta(seconds=oauth_response.expires_in)
    
    # Create or update GitHub token
    github_token = GitHubToken(
        user_id=user_id,
        access_token=oauth_response.access_token,
        refresh_token=oauth_response.refresh_token,
        token_type=oauth_response.token_type,
        scope=oauth_response.scope,
        expires_at=expires_at
    )
    
    session.add(github_token)
    await session.commit()
    
    self.logger.info(f"Stored GitHub token for user ID: {user_id}")


# Add the method to AuthService class
AuthService.store_github_token = store_github_token
