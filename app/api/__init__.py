"""
API Module

This module provides a clean, modular API layer for the GitBot application.
APIs are organized by domain and follow SOLID principles and DRY patterns.

The API layer is responsible for:
- Request/response handling and validation
- Authentication and authorization
- Input sanitization and output formatting
- Error handling and status codes
- Rate limiting and security
"""

from .base import BaseAPI
from .auth import AuthManager
from .dashboard import DashboardAPI
from .webhook import WebhookAPI
from .admin import AdminAPI

# API instances
auth_manager = AuthManager()
dashboard_api = DashboardAPI()
webhook_api = WebhookAPI()
admin_api = AdminAPI()

__all__ = [
    # Base classes
    "BaseAPI",
    
    # API classes
    "AuthManager",
    "DashboardAPI", 
    "WebhookAPI",
    "AdminAPI",
    
    # API instances
    "auth_manager",
    "dashboard_api",
    "webhook_api", 
    "admin_api",
] 