"""
Core Module

This module provides the foundational components for GitBot, including:
- Database management and session handling
- GitHub API integration and authentication
- RAG system initialization and management
- Rate limiting and quota management
- Error handling and logging utilities

All core components follow SOLID principles and DRY patterns for maintainability.
"""

from .base import BaseCore
from .database import DatabaseManager
from .github_auth import GitHubAuthManager
from .rate_limiter import RateLimitManager
from .quota_manager import QuotaManager
from .error_handler import CoreErrorHandler

# Core manager instances
database_manager = DatabaseManager()
github_auth_manager = GitHubAuthManager()
rate_limit_manager = RateLimitManager()
quota_manager = QuotaManager()
error_handler = CoreErrorHandler()

__all__ = [
    # Base classes
    "BaseCore",
    
    # Core manager classes
    "DatabaseManager",
    "GitHubAuthManager", 
    "RateLimitManager",
    "QuotaManager",
    "CoreErrorHandler",
    
    # Core manager instances
    "database_manager",
    "github_auth_manager",
    "rate_limit_manager",
    "quota_manager",
    "error_handler",
] 