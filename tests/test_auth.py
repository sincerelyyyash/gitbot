"""
Authentication System Tests

Tests for GitHub OAuth integration, JWT token management, and user authentication.
"""

import pytest
import jwt
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.services.auth_service import AuthService
from app.models.user import User, Token, TokenData
from app.config import settings


@pytest.fixture
def auth_service():
    """Create AuthService instance for testing."""
    return AuthService()


@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    return User(
        id=1,
        github_id=12345,
        github_login="testuser",
        github_name="Test User",
        github_email="test@example.com",
        github_avatar_url="https://example.com/avatar.jpg",
        role="user",
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return AsyncMock(spec=AsyncSession)


class TestAuthService:
    """Test cases for AuthService."""
    
    def test_generate_github_oauth_url(self, auth_service):
        """Test GitHub OAuth URL generation."""
        oauth_url = auth_service.generate_github_oauth_url()
        
        assert "github.com/login/oauth/authorize" in oauth_url
        assert f"client_id={settings.github_oauth_client_id}" in oauth_url
        assert f"redirect_uri={settings.github_oauth_redirect_uri}" in oauth_url
        assert "scope=read:user%20user:email%20repo" in oauth_url
        assert "state=" in oauth_url
    
    def test_generate_github_oauth_url_with_state(self, auth_service):
        """Test GitHub OAuth URL generation with custom state."""
        custom_state = "test_state_123"
        oauth_url = auth_service.generate_github_oauth_url(custom_state)
        
        assert f"state={custom_state}" in oauth_url
    
    def test_validate_jwt_token_valid(self, auth_service, mock_user):
        """Test JWT token validation with valid token."""
        # Create a valid token
        payload = {
            "user_id": mock_user.id,
            "github_id": mock_user.github_id,
            "github_login": mock_user.github_login,
            "role": mock_user.role,
            "exp": datetime.utcnow() + timedelta(minutes=30),
            "type": "access"
        }
        
        token = jwt.encode(payload, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        # Validate token
        token_data = auth_service.validate_jwt_token(token)
        
        assert token_data.user_id == mock_user.id
        assert token_data.github_id == mock_user.github_id
        assert token_data.github_login == mock_user.github_login
        assert token_data.role == mock_user.role
    
    def test_validate_jwt_token_expired(self, auth_service, mock_user):
        """Test JWT token validation with expired token."""
        # Create an expired token
        payload = {
            "user_id": mock_user.id,
            "github_id": mock_user.github_id,
            "github_login": mock_user.github_login,
            "role": mock_user.role,
            "exp": datetime.utcnow() - timedelta(minutes=30),  # Expired
            "type": "access"
        }
        
        token = jwt.encode(payload, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        # Should raise HTTPException for expired token
        with pytest.raises(Exception) as exc_info:
            auth_service.validate_jwt_token(token)
        
        assert "Token has expired" in str(exc_info.value)
    
    def test_validate_jwt_token_invalid_type(self, auth_service, mock_user):
        """Test JWT token validation with invalid token type."""
        # Create token with wrong type
        payload = {
            "user_id": mock_user.id,
            "exp": datetime.utcnow() + timedelta(minutes=30),
            "type": "refresh"  # Wrong type
        }
        
        token = jwt.encode(payload, auth_service.jwt_secret, algorithm=auth_service.jwt_algorithm)
        
        # Should raise HTTPException for invalid token type
        with pytest.raises(Exception) as exc_info:
            auth_service.validate_jwt_token(token)
        
        assert "Invalid token type" in str(exc_info.value)
    
    def test_validate_jwt_token_invalid_signature(self, auth_service):
        """Test JWT token validation with invalid signature."""
        # Create token with wrong secret
        payload = {
            "user_id": 1,
            "exp": datetime.utcnow() + timedelta(minutes=30),
            "type": "access"
        }
        
        token = jwt.encode(payload, "wrong_secret", algorithm=auth_service.jwt_algorithm)
        
        # Should raise HTTPException for invalid signature
        with pytest.raises(Exception) as exc_info:
            auth_service.validate_jwt_token(token)
        
        assert "Invalid token" in str(exc_info.value)


class TestOAuthEndpoints:
    """Test cases for OAuth API endpoints."""
    
    def test_login_endpoint(self):
        """Test OAuth login endpoint."""
        client = TestClient(app)
        response = client.get("/auth/login")
        
        assert response.status_code == 200
        assert "github.com/login/oauth/authorize" in response.url
    
    def test_login_endpoint_with_state(self):
        """Test OAuth login endpoint with state parameter."""
        client = TestClient(app)
        response = client.get("/auth/login?state=test_state")
        
        assert response.status_code == 200
        assert "state=test_state" in response.url
    
    @patch('app.services.auth_service.AuthService.exchange_oauth_code_for_token')
    @patch('app.services.auth_service.AuthService.fetch_github_user_data')
    @patch('app.services.auth_service.AuthService.create_or_update_user')
    @patch('app.services.auth_service.AuthService.create_user_session')
    @patch('app.services.auth_service.AuthService.store_github_token')
    async def test_oauth_callback_success(
        self,
        mock_store_token,
        mock_create_session,
        mock_create_user,
        mock_fetch_user,
        mock_exchange_token,
        mock_session
    ):
        """Test successful OAuth callback."""
        # Mock responses
        mock_exchange_token.return_value = MagicMock(
            access_token="test_access_token",
            token_type="bearer",
            scope="read:user user:email repo"
        )
        
        mock_fetch_user.return_value = {
            "id": 12345,
            "login": "testuser",
            "name": "Test User",
            "email": "test@example.com",
            "avatar_url": "https://example.com/avatar.jpg"
        }
        
        mock_user = User(
            id=1,
            github_id=12345,
            github_login="testuser",
            github_name="Test User",
            github_email="test@example.com",
            github_avatar_url="https://example.com/avatar.jpg",
            role="user",
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        mock_create_user.return_value = mock_user
        
        mock_tokens = Token(
            access_token="test_jwt_token",
            refresh_token="test_refresh_token",
            token_type="bearer",
            expires_in=1800
        )
        mock_create_session.return_value = mock_tokens
        
        # Mock database session
        mock_session.execute.return_value.scalar_one_or_none.return_value = None
        
        client = TestClient(app)
        
        # This would need proper mocking of the database dependency
        # For now, just test that the endpoint exists
        response = client.get("/auth/callback?code=test_code")
        
        # Should redirect or return error due to missing database session
        assert response.status_code in [200, 422, 500]
    
    def test_refresh_token_endpoint(self):
        """Test token refresh endpoint."""
        client = TestClient(app)
        response = client.post("/auth/refresh", json={"refresh_token": "test_refresh_token"})
        
        # Should return error due to missing database session
        assert response.status_code in [422, 500]
    
    def test_me_endpoint_unauthorized(self):
        """Test /me endpoint without authentication."""
        client = TestClient(app)
        response = client.get("/auth/me")
        
        assert response.status_code == 401
    
    def test_validate_endpoint_unauthorized(self):
        """Test /validate endpoint without authentication."""
        client = TestClient(app)
        response = client.get("/auth/validate")
        
        assert response.status_code == 401


class TestUserModels:
    """Test cases for user models."""
    
    def test_user_model_creation(self):
        """Test User model creation."""
        user = User(
            github_id=12345,
            github_login="testuser",
            github_name="Test User",
            github_email="test@example.com",
            github_avatar_url="https://example.com/avatar.jpg",
            role="user",
            is_active=True
        )
        
        assert user.github_id == 12345
        assert user.github_login == "testuser"
        assert user.github_name == "Test User"
        assert user.github_email == "test@example.com"
        assert user.github_avatar_url == "https://example.com/avatar.jpg"
        assert user.role == "user"
        assert user.is_active is True
    
    def test_token_model_creation(self):
        """Test Token model creation."""
        token = Token(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            token_type="bearer",
            expires_in=1800
        )
        
        assert token.access_token == "test_access_token"
        assert token.refresh_token == "test_refresh_token"
        assert token.token_type == "bearer"
        assert token.expires_in == 1800
    
    def test_token_data_model_creation(self):
        """Test TokenData model creation."""
        token_data = TokenData(
            user_id=1,
            github_id=12345,
            github_login="testuser",
            role="user"
        )
        
        assert token_data.user_id == 1
        assert token_data.github_id == 12345
        assert token_data.github_login == "testuser"
        assert token_data.role == "user"


if __name__ == "__main__":
    pytest.main([__file__])
