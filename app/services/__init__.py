"""
Services Module

"""

from .base import BaseService
from .analytics import AnalyticsService
from .indexing import IndexingService
from .issue_similarity import IssueSimilarityService
from .pr_analysis import PRAnalysisService
from .rag import RAGService
from .github_events import GitHubEventService
from .repository import RepositoryService
from .comment import CommentService
from .health import HealthService
from .auth_service import AuthService

# Service instances
analytics_service = AnalyticsService()
indexing_service = IndexingService()
issue_similarity_service = IssueSimilarityService()
pr_analysis_service = PRAnalysisService()
rag_service = RAGService()
github_event_service = GitHubEventService()
repository_service = RepositoryService()
comment_service = CommentService()
health_service = HealthService()
auth_service = AuthService()

__all__ = [
    # Base classes
    "BaseService",
    
    # Service classes
    "AnalyticsService",
    "IndexingService", 
    "IssueSimilarityService",
    "PRAnalysisService",
    "RAGService",
    "GitHubEventService",
    "RepositoryService",
    "CommentService",
    "HealthService",
    "AuthService",
    
    # Service instances
    "analytics_service",
    "indexing_service",
    "issue_similarity_service", 
    "pr_analysis_service",
    "rag_service",
    "github_event_service",
    "repository_service",
    "comment_service",
    "health_service",
    "auth_service",
] 