"""
User Models

Defines database models for user authentication, authorization, and management.
Includes models for users, sessions, tokens, and preferences.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from app.core.database import Base


class User(Base):
    """User model for storing GitHub user information and authentication data."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    github_id = Column(Integer, unique=True, nullable=False, index=True)
    github_login = Column(String(100), unique=True, nullable=False, index=True)
    github_name = Column(String(200))
    github_email = Column(String(200))
    github_avatar_url = Column(String(500))
    role = Column(String(20), default="user", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    tokens = relationship("GitHubToken", back_populates="user", cascade="all, delete-orphan")
    repository_access = relationship("UserRepositoryAccess", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", uselist=False, cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")


class UserSession(Base):
    """User session model for JWT token management."""
    
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    jwt_token = Column(String(500), nullable=False, index=True)
    refresh_token = Column(String(500), index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")


class GitHubToken(Base):
    """GitHub OAuth token model for storing user GitHub access tokens."""
    
    __tablename__ = "github_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    access_token = Column(String(500), nullable=False)
    refresh_token = Column(String(500))
    token_type = Column(String(50), default="bearer")
    scope = Column(Text)
    expires_at = Column(DateTime(timezone=True), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="tokens")


class UserRepositoryAccess(Base):
    """User repository access model for managing repository permissions."""
    
    __tablename__ = "user_repository_access"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False, index=True)
    access_level = Column(String(20), default="read", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="repository_access")
    repository = relationship("Repository", back_populates="user_access")
    
    # Constraints
    __table_args__ = (UniqueConstraint("user_id", "repository_id", name="uq_user_repository"),)


class UserPreference(Base):
    """User preferences model for storing user-specific settings."""
    
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    dashboard_settings = Column(JSON)
    notification_preferences = Column(JSON)
    theme = Column(String(50), default="light")
    language = Column(String(10), default="en")
    timezone = Column(String(50), default="UTC")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="preferences")


class Notification(Base):
    """Notification model for storing user notifications."""
    
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    type = Column(String(50), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(String(20), default="info")
    repository_id = Column(Integer, ForeignKey("repositories.id"), index=True)
    action_log_id = Column(Integer, ForeignKey("action_logs.id"), index=True)
    metadata_json = Column(JSON)
    is_read = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="notifications")
    repository = relationship("Repository", back_populates="notifications")
    action_log = relationship("ActionLog", back_populates="notifications")


# Pydantic models for API requests/responses
class UserBase(BaseModel):
    """Base user model for API requests/responses."""
    github_id: int
    github_login: str
    github_name: Optional[str] = None
    github_email: Optional[str] = None
    github_avatar_url: Optional[str] = None
    role: str = "user"
    is_active: bool = True


class UserCreate(UserBase):
    """Model for creating a new user."""
    pass


class UserUpdate(BaseModel):
    """Model for updating user information."""
    github_name: Optional[str] = None
    github_email: Optional[str] = None
    github_avatar_url: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """Model for user API responses."""
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UserSessionCreate(BaseModel):
    """Model for creating a new user session."""
    user_id: int
    jwt_token: str
    refresh_token: Optional[str] = None
    expires_at: datetime


class UserSessionResponse(BaseModel):
    """Model for user session API responses."""
    id: int
    user_id: int
    expires_at: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True


class GitHubTokenCreate(BaseModel):
    """Model for creating a new GitHub token."""
    user_id: int
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    scope: Optional[str] = None
    expires_at: Optional[datetime] = None


class GitHubTokenResponse(BaseModel):
    """Model for GitHub token API responses."""
    id: int
    user_id: int
    token_type: str
    scope: Optional[str] = None
    expires_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserRepositoryAccessCreate(BaseModel):
    """Model for creating user repository access."""
    user_id: int
    repository_id: int
    access_level: str = "read"


class UserRepositoryAccessResponse(BaseModel):
    """Model for user repository access API responses."""
    id: int
    user_id: int
    repository_id: int
    access_level: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserPreferenceCreate(BaseModel):
    """Model for creating user preferences."""
    user_id: int
    dashboard_settings: Optional[Dict[str, Any]] = None
    notification_preferences: Optional[Dict[str, Any]] = None
    theme: str = "light"
    language: str = "en"
    timezone: str = "UTC"


class UserPreferenceUpdate(BaseModel):
    """Model for updating user preferences."""
    dashboard_settings: Optional[Dict[str, Any]] = None
    notification_preferences: Optional[Dict[str, Any]] = None
    theme: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None


class UserPreferenceResponse(BaseModel):
    """Model for user preferences API responses."""
    id: int
    user_id: int
    dashboard_settings: Optional[Dict[str, Any]] = None
    notification_preferences: Optional[Dict[str, Any]] = None
    theme: str
    language: str
    timezone: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class NotificationCreate(BaseModel):
    """Model for creating a new notification."""
    user_id: int
    type: str
    title: str
    message: str
    severity: str = "info"
    repository_id: Optional[int] = None
    action_log_id: Optional[int] = None
    metadata_json: Optional[Dict[str, Any]] = None


class NotificationUpdate(BaseModel):
    """Model for updating a notification."""
    is_read: Optional[bool] = None


class NotificationResponse(BaseModel):
    """Model for notification API responses."""
    id: int
    user_id: int
    type: str
    title: str
    message: str
    severity: str
    repository_id: Optional[int] = None
    action_log_id: Optional[int] = None
    metadata_json: Optional[Dict[str, Any]] = None
    is_read: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# Authentication models
class Token(BaseModel):
    """JWT token model for authentication responses."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data model for JWT payload."""
    user_id: Optional[int] = None
    github_id: Optional[int] = None
    github_login: Optional[str] = None
    role: Optional[str] = None


class GitHubOAuthRequest(BaseModel):
    """GitHub OAuth authorization request model."""
    code: str
    state: Optional[str] = None


class GitHubOAuthResponse(BaseModel):
    """GitHub OAuth response model."""
    access_token: str
    token_type: str
    scope: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None


class LoginResponse(BaseModel):
    """Login response model."""
    user: UserResponse
    tokens: Token
    is_new_user: bool = False
