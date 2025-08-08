"""
Authentication Service

Provides comprehensive authentication and authorization services including:
- GitHub OAuth integration
- JWT token management
- User session management
- Repository access validation
"""

import jwt
import httpx
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import selectinload
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings
from app.core.base import BaseCore, core_operation
from app.models.user import (
    User, UserSession, GitHubToken, UserRepositoryAccess, UserPreference,
    UserCreate, UserUpdate, TokenData, GitHubOAuthRequest, GitHubOAuthResponse,
    Token, LoginResponse, UserResponse
)


class AuthService(BaseCore):
    """
    Authentication service for handling user authentication and authorization.
    
    Features:
    - GitHub OAuth integration
    - JWT token generation and validation
    - User session management
    - Repository access validation
    - User profile management
    """
    
    def __init__(self):
        """Initialize the authentication service."""
        super().__init__("auth_service")
        
        # GitHub OAuth endpoints
        self.github_oauth_url = "https://github.com/login/oauth/authorize"
        self.github_token_url = "https://github.com/login/oauth/access_token"
        self.github_user_url = "https://api.github.com/user"
        self.github_user_emails_url = "https://api.github.com/user/emails"
        
        # JWT configuration
        self.jwt_secret = settings.jwt_secret_key
        self.jwt_algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.jwt_access_token_expire_minutes
        self.refresh_token_expire_days = settings.jwt_refresh_token_expire_days
    
    @core_operation("generate_oauth_url")
    def generate_github_oauth_url(self, state: Optional[str] = None) -> str:
        """
        Generate GitHub OAuth authorization URL.
        
        Args:
            state: Optional state parameter for CSRF protection
            
        Returns:
            GitHub OAuth authorization URL
        """
        if not state:
            state = secrets.token_urlsafe(32)
        
        params = {
            "client_id": settings.github_oauth_client_id,
            "redirect_uri": settings.github_oauth_redirect_uri,
            "scope": "read:user user:email repo",
            "state": state
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        oauth_url = f"{self.github_oauth_url}?{query_string}"
        
        self.logger.debug(f"Generated OAuth URL with state: {state}")
        return oauth_url
    
    @core_operation("exchange_oauth_code")
    async def exchange_oauth_code_for_token(
        self, 
        code: str, 
        state: Optional[str] = None
    ) -> GitHubOAuthResponse:
        """
        Exchange OAuth authorization code for access token.
        
        Args:
            code: OAuth authorization code from GitHub
            state: State parameter for CSRF protection
            
        Returns:
            GitHub OAuth response with access token
            
        Raises:
            HTTPException: If token exchange fails
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.github_token_url,
                    data={
                        "client_id": settings.github_oauth_client_id,
                        "client_secret": settings.github_oauth_client_secret,
                        "code": code,
                        "redirect_uri": settings.github_oauth_redirect_uri
                    },
                    headers={"Accept": "application/json"}
                )
                
                if response.status_code != 200:
                    self.logger.error(f"GitHub OAuth token exchange failed: {response.status_code}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Failed to exchange OAuth code for token"
                    )
                
                token_data = response.json()
                
                if "error" in token_data:
                    self.logger.error(f"GitHub OAuth error: {token_data['error']}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"OAuth error: {token_data['error']}"
                    )
                
                return GitHubOAuthResponse(**token_data)
                
        except httpx.RequestError as e:
            self.logger.error(f"HTTP request error during OAuth token exchange: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to communicate with GitHub OAuth service"
            )
    
    @core_operation("fetch_github_user")
    async def fetch_github_user_data(self, access_token: str) -> Dict[str, Any]:
        """
        Fetch GitHub user data using access token.
        
        Args:
            access_token: GitHub OAuth access token
            
        Returns:
            GitHub user data
            
        Raises:
            HTTPException: If user data fetch fails
        """
        try:
            async with httpx.AsyncClient() as client:
                # Fetch user profile
                user_response = await client.get(
                    self.github_user_url,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/vnd.github.v3+json"
                    }
                )
                
                if user_response.status_code != 200:
                    self.logger.error(f"Failed to fetch GitHub user: {user_response.status_code}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Failed to fetch GitHub user data"
                    )
                
                user_data = user_response.json()
                
                # Fetch user emails
                emails_response = await client.get(
                    self.github_user_emails_url,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/vnd.github.v3+json"
                    }
                )
                
                if emails_response.status_code == 200:
                    emails_data = emails_response.json()
                    # Find primary email
                    primary_email = next(
                        (email["email"] for email in emails_data if email.get("primary")),
                        None
                    )
                    user_data["email"] = primary_email
                
                return user_data
                
        except httpx.RequestError as e:
            self.logger.error(f"HTTP request error fetching GitHub user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to communicate with GitHub API"
            )
    
    @core_operation("create_or_update_user")
    async def create_or_update_user(
        self, 
        session: AsyncSession, 
        github_user_data: Dict[str, Any]
    ) -> User:
        """
        Create or update user in database.
        
        Args:
            session: Database session
            github_user_data: GitHub user data
            
        Returns:
            User object
        """
        # Check if user exists
        stmt = select(User).where(User.github_id == github_user_data["id"])
        result = await session.execute(stmt)
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            # Update existing user
            existing_user.github_name = github_user_data.get("name")
            existing_user.github_email = github_user_data.get("email")
            existing_user.github_avatar_url = github_user_data.get("avatar_url")
            existing_user.updated_at = datetime.utcnow()
            
            await session.commit()
            await session.refresh(existing_user)
            
            self.logger.info(f"Updated existing user: {existing_user.github_login}")
            return existing_user
        else:
            # Create new user
            new_user = User(
                github_id=github_user_data["id"],
                github_login=github_user_data["login"],
                github_name=github_user_data.get("name"),
                github_email=github_user_data.get("email"),
                github_avatar_url=github_user_data.get("avatar_url"),
                role="user",
                is_active=True
            )
            
            session.add(new_user)
            await session.commit()
            await session.refresh(new_user)
            
            self.logger.info(f"Created new user: {new_user.github_login}")
            return new_user
    
    @core_operation("create_user_session")
    async def create_user_session(
        self, 
        session: AsyncSession, 
        user: User
    ) -> Token:
        """
        Create user session with JWT tokens.
        
        Args:
            session: Database session
            user: User object
            
        Returns:
            Token object with access and refresh tokens
        """
        # Generate JWT tokens
        access_token_expires = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        refresh_token_expires = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        # Create token payload
        access_token_payload = {
            "user_id": user.id,
            "github_id": user.github_id,
            "github_login": user.github_login,
            "role": user.role,
            "exp": access_token_expires,
            "type": "access"
        }
        
        refresh_token_payload = {
            "user_id": user.id,
            "exp": refresh_token_expires,
            "type": "refresh"
        }
        
        # Generate tokens
        access_token = jwt.encode(access_token_payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        refresh_token = jwt.encode(refresh_token_payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        # Store session in database
        user_session = UserSession(
            user_id=user.id,
            jwt_token=access_token,
            refresh_token=refresh_token,
            expires_at=access_token_expires
        )
        
        session.add(user_session)
        await session.commit()
        
        self.logger.info(f"Created session for user: {user.github_login}")
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=self.access_token_expire_minutes * 60
        )
    
    @core_operation("validate_jwt_token")
    def validate_jwt_token(self, token: str) -> TokenData:
        """
        Validate JWT token and extract user data.
        
        Args:
            token: JWT token string
            
        Returns:
            TokenData object with user information
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check token type
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            return TokenData(
                user_id=payload.get("user_id"),
                github_id=payload.get("github_id"),
                github_login=payload.get("github_login"),
                role=payload.get("role")
            )
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError as e:
            self.logger.warning(f"JWT validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    @core_operation("refresh_access_token")
    async def refresh_access_token(
        self, 
        session: AsyncSession, 
        refresh_token: str
    ) -> Token:
        """
        Refresh access token using refresh token.
        
        Args:
            session: Database session
            refresh_token: Refresh token string
            
        Returns:
            New Token object
            
        Raises:
            HTTPException: If refresh token is invalid
        """
        try:
            # Validate refresh token
            payload = jwt.decode(refresh_token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )
            
            user_id = payload.get("user_id")
            
            # Get user
            stmt = select(User).where(User.id == user_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            # Generate new tokens
            return await self.create_user_session(session, user)
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
    
    @core_operation("get_user_by_id")
    async def get_user_by_id(
        self, 
        session: AsyncSession, 
        user_id: int
    ) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            session: Database session
            user_id: User ID
            
        Returns:
            User object or None
        """
        stmt = select(User).where(User.id == user_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    @core_operation("get_user_by_github_id")
    async def get_user_by_github_id(
        self, 
        session: AsyncSession, 
        github_id: int
    ) -> Optional[User]:
        """
        Get user by GitHub ID.
        
        Args:
            session: Database session
            github_id: GitHub user ID
            
        Returns:
            User object or None
        """
        stmt = select(User).where(User.github_id == github_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    @core_operation("update_user")
    async def update_user(
        self, 
        session: AsyncSession, 
        user_id: int, 
        user_update: UserUpdate
    ) -> Optional[User]:
        """
        Update user information.
        
        Args:
            session: Database session
            user_id: User ID
            user_update: User update data
            
        Returns:
            Updated User object or None
        """
        stmt = select(User).where(User.id == user_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        # Update fields
        update_data = user_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        
        await session.commit()
        await session.refresh(user)
        
        self.logger.info(f"Updated user: {user.github_login}")
        return user
    
    @core_operation("validate_repository_access")
    async def validate_repository_access(
        self, 
        session: AsyncSession, 
        user_id: int, 
        repository_id: int, 
        required_access: str = "read"
    ) -> bool:
        """
        Validate user access to repository.
        
        Args:
            session: Database session
            user_id: User ID
            repository_id: Repository ID
            required_access: Required access level (read, write, admin)
            
        Returns:
            True if user has required access
        """
        # Access level hierarchy
        access_levels = {"read": 1, "write": 2, "admin": 3}
        required_level = access_levels.get(required_access, 1)
        
        stmt = select(UserRepositoryAccess).where(
            and_(
                UserRepositoryAccess.user_id == user_id,
                UserRepositoryAccess.repository_id == repository_id
            )
        )
        result = await session.execute(stmt)
        access = result.scalar_one_or_none()
        
        if not access:
            return False
        
        user_level = access_levels.get(access.access_level, 0)
        return user_level >= required_level
    
    @core_operation("logout_user")
    async def logout_user(
        self, 
        session: AsyncSession, 
        user_id: int, 
        token: str
    ) -> bool:
        """
        Logout user by invalidating session.
        
        Args:
            session: Database session
            user_id: User ID
            token: JWT token to invalidate
            
        Returns:
            True if logout successful
        """
        # Delete session
        stmt = select(UserSession).where(
            and_(
                UserSession.user_id == user_id,
                UserSession.jwt_token == token
            )
        )
        result = await session.execute(stmt)
        user_session = result.scalar_one_or_none()
        
        if user_session:
            await session.delete(user_session)
            await session.commit()
            
            self.logger.info(f"Logged out user ID: {user_id}")
            return True
        
        return False
    
    @core_operation("cleanup_expired_sessions")
    async def cleanup_expired_sessions(self, session: AsyncSession) -> int:
        """
        Clean up expired user sessions.
        
        Args:
            session: Database session
            
        Returns:
            Number of sessions cleaned up
        """
        stmt = select(UserSession).where(UserSession.expires_at < datetime.utcnow())
        result = await session.execute(stmt)
        expired_sessions = result.scalars().all()
        
        for expired_session in expired_sessions:
            await session.delete(expired_session)
        
        await session.commit()
        
        self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)


# Security scheme for FastAPI
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = security,
    session: AsyncSession = None
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP authorization credentials
        session: Database session
        
    Returns:
        Current authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    if not session:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database session not available"
        )
    
    auth_service = AuthService()
    token_data = auth_service.validate_jwt_token(credentials.credentials)
    
    if not token_data.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user = await auth_service.get_user_by_id(session, token_data.user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    return user


async def get_current_active_user(current_user: User = None) -> User:
    """
    Get current active user.
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        Current active user
        
    Raises:
        HTTPException: If user is not active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def require_role(required_role: str):
    """
    Dependency to require specific user role.
    
    Args:
        required_role: Required role (user, admin, owner)
        
    Returns:
        Dependency function
    """
    async def role_checker(current_user: User = None):
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Role hierarchy
        role_hierarchy = {"user": 1, "admin": 2, "owner": 3}
        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        
        return current_user
    
    return role_checker
