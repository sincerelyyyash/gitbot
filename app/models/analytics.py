"""
SQLAlchemy models for GitBot analytics and tracking.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean, Float, Index
from sqlalchemy.sql import func
from app.core.database import Base

class Repository(Base):
    """Repository tracking model."""
    __tablename__ = "repositories"
    
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), unique=True, index=True, nullable=False)
    installation_id = Column(Integer, nullable=False)
    owner = Column(String(100), nullable=False)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    indexed_at = Column(DateTime, nullable=True)
    last_activity = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Metadata
    metadata_json = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index('idx_repo_owner_name', 'owner', 'name'),
        Index('idx_repo_installation', 'installation_id'),
        Index('idx_repo_activity', 'last_activity'),
    )

class ActionLog(Base):
    """Log of all gitbot actions."""
    __tablename__ = "action_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    repository_id = Column(Integer, nullable=True)  # Can be null for system actions
    repo_full_name = Column(String(255), index=True, nullable=True)
    
    # Action details
    action_type = Column(String(50), nullable=False, index=True)  # 'issue_comment', 'pr_analysis', 'indexing', etc.
    action_subtype = Column(String(50), nullable=True)  # 'similarity_check', 'security_scan', etc.
    
    # Event context
    github_event_type = Column(String(50), nullable=True)  # 'issues', 'pull_request', etc.
    github_event_action = Column(String(50), nullable=True)  # 'opened', 'closed', etc.
    
    # Target details
    target_type = Column(String(20), nullable=True)  # 'issue', 'pr', 'comment', 'repository'
    target_number = Column(Integer, nullable=True)  # Issue/PR number
    target_id = Column(String(100), nullable=True)  # GitHub ID
    
    # Status and timing
    status = Column(String(20), nullable=False, default='started')  # 'started', 'completed', 'failed', 'skipped'
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    
    # Result details
    success = Column(Boolean, nullable=True)
    error_message = Column(Text, nullable=True)
    response_posted = Column(Boolean, default=False)
    
    # Metrics
    tokens_used = Column(Integer, nullable=True)
    api_calls_made = Column(Integer, nullable=True)
    
    # Additional data
    metadata_json = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index('idx_action_type_status', 'action_type', 'status'),
        Index('idx_action_repo_time', 'repo_full_name', 'started_at'),
        Index('idx_action_target', 'target_type', 'target_number'),
        Index('idx_action_time', 'started_at'),
    )

class IssueAnalysis(Base):
    """Detailed tracking of issue analysis."""
    __tablename__ = "issue_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    action_log_id = Column(Integer, nullable=True)  # Link to action log
    
    # Issue details
    repo_full_name = Column(String(255), nullable=False, index=True)
    issue_number = Column(Integer, nullable=False)
    issue_title = Column(Text, nullable=True)
    issue_body_length = Column(Integer, nullable=True)
    
    # Analysis results
    is_question = Column(Boolean, nullable=True)
    is_bug_report = Column(Boolean, nullable=True)
    is_feature_request = Column(Boolean, nullable=True)
    is_duplicate = Column(Boolean, nullable=True)
    is_invalid = Column(Boolean, nullable=True)
    
    # Similarity analysis
    similar_issues_found = Column(Integer, default=0)
    highest_similarity_score = Column(Float, nullable=True)
    similar_issue_numbers = Column(JSON, nullable=True)  # Array of issue numbers
    
    # Response details
    response_generated = Column(Boolean, default=False)
    response_length = Column(Integer, nullable=True)
    response_type = Column(String(50), nullable=True)  # 'answer', 'analysis', 'similarity_warning'
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_issue_repo_number', 'repo_full_name', 'issue_number'),
        Index('idx_issue_analysis_type', 'is_question', 'is_bug_report', 'is_feature_request'),
        Index('idx_issue_duplicate', 'is_duplicate'),
    )

class PRAnalysis(Base):
    """Detailed tracking of pull request analysis."""
    __tablename__ = "pr_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    action_log_id = Column(Integer, nullable=True)  # Link to action log
    
    # PR details
    repo_full_name = Column(String(255), nullable=False, index=True)
    pr_number = Column(Integer, nullable=False)
    pr_title = Column(Text, nullable=True)
    
    # File changes
    files_changed = Column(Integer, default=0)
    lines_added = Column(Integer, default=0)
    lines_deleted = Column(Integer, default=0)
    
    # Analysis results
    security_issues_found = Column(Integer, default=0)
    quality_issues_found = Column(Integer, default=0)
    complexity_issues_found = Column(Integer, default=0)
    potential_bugs_found = Column(Integer, default=0)
    duplicate_functionality_found = Column(Integer, default=0)
    
    # Scores
    overall_score = Column(Integer, nullable=True)  # 0-100
    review_priority = Column(String(20), nullable=True)  # 'low', 'medium', 'high', 'critical'
    
    # Languages detected
    languages_detected = Column(JSON, nullable=True)  # Array of languages
    
    # Response details
    review_posted = Column(Boolean, default=False)
    suggestions_count = Column(Integer, default=0)
    labels_added = Column(JSON, nullable=True)  # Array of labels added
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_pr_repo_number', 'repo_full_name', 'pr_number'),
        Index('idx_pr_priority', 'review_priority'),
        Index('idx_pr_issues', 'security_issues_found', 'quality_issues_found'),
    )

class PRSummary(Base):
    """Detailed tracking of pull request summary generation."""
    __tablename__ = "pr_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    action_log_id = Column(Integer, nullable=True)  # Link to action log
    
    # PR details
    repo_full_name = Column(String(255), nullable=False, index=True)
    pr_number = Column(Integer, nullable=False)
    pr_title = Column(Text, nullable=True)
    pr_body_length = Column(Integer, nullable=True)
    
    # File changes
    files_changed = Column(Integer, default=0)
    files_list = Column(JSON, nullable=True)  # Array of changed file names
    lines_added = Column(Integer, default=0)
    lines_deleted = Column(Integer, default=0)
    
    # Summary details
    summary_generated = Column(Boolean, default=False)
    summary_length = Column(Integer, nullable=True)
    summary_type = Column(String(50), nullable=True)  # 'rag_generated', 'fallback'
    rag_system_available = Column(Boolean, nullable=True)
    
    # Response details
    summary_posted = Column(Boolean, default=False)
    response_time_ms = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_pr_summary_repo_number', 'repo_full_name', 'pr_number'),
        Index('idx_pr_summary_type', 'summary_type'),
        Index('idx_pr_summary_rag', 'rag_system_available'),
    )

class IndexingJob(Base):
    """Tracking of repository indexing jobs."""
    __tablename__ = "indexing_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    action_log_id = Column(Integer, nullable=True)  # Link to action log
    
    # Repository details
    repo_full_name = Column(String(255), nullable=False, index=True)
    installation_id = Column(Integer, nullable=False)
    
    # Job details
    job_type = Column(String(50), nullable=False)  # 'initial', 'refresh', 'incremental'
    trigger = Column(String(50), nullable=True)  # 'installation', 'push', 'manual', 'scheduled'
    priority = Column(Integer, default=1)
    force_refresh = Column(Boolean, default=False)
    
    # Status tracking
    status = Column(String(20), nullable=False, default='queued')  # 'queued', 'processing', 'completed', 'failed'
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Content metrics
    files_processed = Column(Integer, nullable=True)
    documents_created = Column(Integer, nullable=True)
    embeddings_generated = Column(Integer, nullable=True)
    
    # Timing
    queued_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    last_error_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        Index('idx_indexing_repo_status', 'repo_full_name', 'status'),
        Index('idx_indexing_status_priority', 'status', 'priority'),
        Index('idx_indexing_queued', 'queued_at'),
    )

class UsageMetrics(Base):
    """Daily usage metrics and quotas."""
    __tablename__ = "usage_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Scope
    repo_full_name = Column(String(255), nullable=True, index=True)  # Null for global metrics
    date = Column(DateTime, nullable=False, index=True)
    
    # Action counts
    total_actions = Column(Integer, default=0)
    issue_comments = Column(Integer, default=0)
    issue_analyses = Column(Integer, default=0)
    pr_analyses = Column(Integer, default=0)
    indexing_jobs = Column(Integer, default=0)
    
    # API usage
    tokens_used = Column(Integer, default=0)
    api_calls_made = Column(Integer, default=0)
    github_api_calls = Column(Integer, default=0)
    gemini_api_calls = Column(Integer, default=0)
    
    # Performance metrics
    avg_response_time_ms = Column(Float, nullable=True)
    success_rate = Column(Float, nullable=True)  # 0.0 to 1.0
    error_count = Column(Integer, default=0)
    
    # Storage metrics
    documents_stored = Column(Integer, nullable=True)
    embeddings_stored = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_metrics_repo_date', 'repo_full_name', 'date'),
        Index('idx_metrics_date', 'date'),
    )

class SystemHealth(Base):
    """System health and status tracking."""
    __tablename__ = "system_health"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Health check details
    component = Column(String(50), nullable=False, index=True)  # 'database', 'gemini_api', 'github_api', 'chromadb'
    status = Column(String(20), nullable=False)  # 'healthy', 'degraded', 'unhealthy'
    
    # Metrics
    response_time_ms = Column(Float, nullable=True)
    error_rate = Column(Float, nullable=True)  # 0.0 to 1.0
    
    # Details
    message = Column(Text, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    
    # Timestamps
    checked_at = Column(DateTime, default=func.now(), index=True)
    
    __table_args__ = (
        Index('idx_health_component_time', 'component', 'checked_at'),
        Index('idx_health_status', 'status'),
    ) 