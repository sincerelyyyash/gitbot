"""
Analytics Service

Provides comprehensive tracking and analytics for gitbot actions including:
- Action logging and tracking
- Performance metrics calculation
- Usage analytics and quotas
- Repository insights
- System health monitoring
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base import BaseService
from .repository import repository_service
from app.models.analytics import (
    ActionLog, IssueAnalysis, PRAnalysis, PRSummary,
    IndexingJob, UsageMetrics, SystemHealth
)

@dataclass
class ActionMetrics:
    """Action metrics data class."""
    total_actions: int
    successful_actions: int
    failed_actions: int
    success_rate: float
    avg_duration_ms: float
    recent_activity_count: int

@dataclass
class RepositoryStats:
    """Repository statistics data class."""
    repo_full_name: str
    actions: Dict[str, ActionMetrics]
    issues: Optional[Dict[str, Any]]
    pull_requests: Optional[Dict[str, Any]]
    pr_summaries: Optional[Dict[str, Any]]
    recent_activity_count: int

class AnalyticsService(BaseService[RepositoryStats]):
    """Service for tracking and analyzing gitbot actions."""
    
    def __init__(self):
        super().__init__("AnalyticsService")
    
    async def register_repository(
        self, 
        full_name: str, 
        installation_id: int,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Register a new repository or update existing one.
        
        Args:
            full_name: Repository full name
            installation_id: GitHub installation ID
            metadata: Optional repository metadata
            
        Returns:
            Repository ID
        """
        operation = "register_repository"
        start_time = self.log_operation_start(operation, repo=full_name)
        
        try:
            # Use repository service to handle registration
            repo_id = await repository_service.register_repository(
                full_name=full_name,
                installation_id=installation_id,
                metadata=metadata
            )
            
            self.log_operation_complete(operation, start_time, success=True, repo_id=repo_id)
            return repo_id
                
        except Exception as e:
            self.log_error(operation, e, repo=full_name)
            raise
    
    async def log_action_start(
        self,
        action_type: str,
        repo_full_name: Optional[str] = None,
        action_subtype: Optional[str] = None,
        github_event_type: Optional[str] = None,
        github_event_action: Optional[str] = None,
        target_type: Optional[str] = None,
        target_number: Optional[int] = None,
        target_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Log the start of an action and return the action log ID.
        
        Args:
            action_type: Type of action being performed
            repo_full_name: Repository full name
            action_subtype: Subtype of action
            github_event_type: GitHub event type
            github_event_action: GitHub event action
            target_type: Type of target (issue, pr, etc.)
            target_number: Target number
            target_id: Target ID
            metadata: Additional metadata
            
        Returns:
            Action log ID
        """
        operation = "log_action_start"
        start_time = self.log_operation_start(
            operation, 
            action_type=action_type,
            repo=repo_full_name
        )
        
        try:
            async with self.get_db_session() as db:
                action_log = ActionLog(
                    repo_full_name=repo_full_name,
                    action_type=action_type,
                    action_subtype=action_subtype,
                    github_event_type=github_event_type,
                    github_event_action=github_event_action,
                    target_type=target_type,
                    target_number=target_number,
                    target_id=target_id,
                    status='started',
                    metadata_json=metadata
                )
                db.add(action_log)
                await db.commit()
                await db.refresh(action_log)
                
                # Update repository activity if applicable
                if repo_full_name:
                    await self._update_repository_activity(db, repo_full_name)
                
                self.log_operation_complete(operation, start_time, success=True, action_id=action_log.id)
                return action_log.id
                
        except Exception as e:
            self.log_error(operation, e, action_type=action_type, repo=repo_full_name)
            raise
    
    async def log_action_complete(
        self,
        action_log_id: int,
        success: bool,
        error_message: Optional[str] = None,
        response_posted: bool = False,
        tokens_used: Optional[int] = None,
        api_calls_made: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log the completion of an action.
        
        Args:
            action_log_id: Action log ID
            success: Whether the action was successful
            error_message: Error message if failed
            response_posted: Whether a response was posted
            tokens_used: Number of tokens used
            api_calls_made: Number of API calls made
            metadata: Additional metadata
        """
        operation = "log_action_complete"
        start_time = self.log_operation_start(operation, action_id=action_log_id)
        
        try:
            async with self.get_db_session() as db:
                result = await db.execute(
                    "SELECT * FROM action_logs WHERE id = :action_id",
                    {"action_id": action_log_id}
                )
                action_log = result.fetchone()
                
                if action_log:
                    await db.execute(
                        """
                        UPDATE action_logs 
                        SET completed_at = :completed_at,
                            success = :success,
                            status = :status,
                            error_message = :error_message,
                            response_posted = :response_posted,
                            tokens_used = :tokens_used,
                            api_calls_made = :api_calls_made,
                            duration_ms = :duration_ms,
                            metadata_json = COALESCE(:metadata, metadata_json)
                        WHERE id = :action_id
                        """,
                        {
                            "completed_at": datetime.utcnow(),
                            "success": success,
                            "status": 'completed' if success else 'failed',
                            "error_message": error_message,
                            "response_posted": response_posted,
                            "tokens_used": tokens_used,
                            "api_calls_made": api_calls_made,
                            "duration_ms": self._calculate_duration_ms(action_log.started_at),
                            "metadata": metadata,
                            "action_id": action_log_id
                        }
                    )
                    await db.commit()
                
                self.log_operation_complete(operation, start_time, success=True)
                
        except Exception as e:
            self.log_error(operation, e, action_id=action_log_id)
    
    async def log_issue_analysis(
        self,
        action_log_id: Optional[int],
        repo_full_name: str,
        issue_number: int,
        issue_title: Optional[str] = None,
        issue_body_length: Optional[int] = None,
        is_question: Optional[bool] = None,
        is_bug_report: Optional[bool] = None,
        is_feature_request: Optional[bool] = None,
        is_duplicate: Optional[bool] = None,
        is_invalid: Optional[bool] = None,
        similar_issues_found: int = 0,
        highest_similarity_score: Optional[float] = None,
        similar_issue_numbers: Optional[List[int]] = None,
        response_generated: bool = False,
        response_length: Optional[int] = None,
        response_type: Optional[str] = None
    ) -> int:
        """
        Log detailed issue analysis results.
        
        Args:
            action_log_id: Associated action log ID
            repo_full_name: Repository full name
            issue_number: Issue number
            issue_title: Issue title
            issue_body_length: Length of issue body
            is_question: Whether issue is a question
            is_bug_report: Whether issue is a bug report
            is_feature_request: Whether issue is a feature request
            is_duplicate: Whether issue is a duplicate
            is_invalid: Whether issue is invalid
            similar_issues_found: Number of similar issues found
            highest_similarity_score: Highest similarity score
            similar_issue_numbers: List of similar issue numbers
            response_generated: Whether response was generated
            response_length: Length of response
            response_type: Type of response
            
        Returns:
            Issue analysis ID
        """
        operation = "log_issue_analysis"
        start_time = self.log_operation_start(
            operation, 
            repo=repo_full_name,
            issue=issue_number
        )
        
        try:
            async with self.get_db_session() as db:
                analysis = IssueAnalysis(
                    action_log_id=action_log_id,
                    repo_full_name=repo_full_name,
                    issue_number=issue_number,
                    issue_title=issue_title,
                    issue_body_length=issue_body_length,
                    is_question=is_question,
                    is_bug_report=is_bug_report,
                    is_feature_request=is_feature_request,
                    is_duplicate=is_duplicate,
                    is_invalid=is_invalid,
                    similar_issues_found=similar_issues_found,
                    highest_similarity_score=highest_similarity_score,
                    similar_issue_numbers=similar_issue_numbers,
                    response_generated=response_generated,
                    response_length=response_length,
                    response_type=response_type
                )
                db.add(analysis)
                await db.commit()
                await db.refresh(analysis)
                
                self.log_operation_complete(operation, start_time, success=True, analysis_id=analysis.id)
                return analysis.id
                
        except Exception as e:
            self.log_error(operation, e, repo=repo_full_name, issue=issue_number)
            raise
    
    async def get_repository_stats(self, repo_full_name: Optional[str] = None) -> RepositoryStats:
        """
        Get comprehensive repository statistics.
        
        Args:
            repo_full_name: Repository full name (optional for global stats)
            
        Returns:
            RepositoryStats object
        """
        operation = "get_repository_stats"
        start_time = self.log_operation_start(operation, repo=repo_full_name)
        
        try:
            async with self.get_db_session() as db:
                # Base query filter
                repo_filter = "WHERE repo_full_name = :repo_full_name" if repo_full_name else ""
                params = {"repo_full_name": repo_full_name} if repo_full_name else {}
                
                # Action counts by type
                result = await db.execute(
                    f"""
                    SELECT 
                        action_type,
                        COUNT(*) as count,
                        SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as successful,
                        AVG(duration_ms) as avg_duration_ms
                    FROM action_logs 
                    {repo_filter}
                    GROUP BY action_type
                    """,
                    params
                )
                
                action_stats = {}
                for row in result.fetchall():
                    action_stats[row.action_type] = ActionMetrics(
                        total_actions=row.count,
                        successful_actions=row.successful or 0,
                        failed_actions=row.count - (row.successful or 0),
                        success_rate=(row.successful or 0) / row.count if row.count > 0 else 0,
                        avg_duration_ms=float(row.avg_duration_ms) if row.avg_duration_ms else 0,
                        recent_activity_count=0  # Will be calculated separately
                    )
                
                # Recent activity (last 30 days)
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                recent_params = {**params, "thirty_days_ago": thirty_days_ago}
                recent_filter = f"AND started_at >= :thirty_days_ago" if repo_full_name else "WHERE started_at >= :thirty_days_ago"
                
                result = await db.execute(
                    f"SELECT COUNT(*) FROM action_logs {repo_filter} {recent_filter}",
                    recent_params
                )
                recent_activity_count = result.fetchone()[0] or 0
                
                # Update recent activity count in action stats
                for action_metrics in action_stats.values():
                    action_metrics.recent_activity_count = recent_activity_count
                
                # Issue analysis stats (only for specific repo)
                issues_stats = None
                if repo_full_name:
                    result = await db.execute(
                        """
                        SELECT 
                            COUNT(*) as total_issues,
                            SUM(CASE WHEN is_question = true THEN 1 ELSE 0 END) as questions,
                            SUM(CASE WHEN is_bug_report = true THEN 1 ELSE 0 END) as bug_reports,
                            SUM(CASE WHEN is_duplicate = true THEN 1 ELSE 0 END) as duplicates,
                            AVG(similar_issues_found) as avg_similar_found
                        FROM issue_analyses 
                        WHERE repo_full_name = :repo_full_name
                        """,
                        {"repo_full_name": repo_full_name}
                    )
                    issue_row = result.fetchone()
                    if issue_row:
                        issues_stats = {
                            'total': issue_row.total_issues or 0,
                            'questions': issue_row.questions or 0,
                            'bug_reports': issue_row.bug_reports or 0,
                            'duplicates': issue_row.duplicates or 0,
                            'avg_similar_found': float(issue_row.avg_similar_found) if issue_row.avg_similar_found else 0
                        }
                
                # PR analysis stats (only for specific repo)
                pr_stats = None
                if repo_full_name:
                    result = await db.execute(
                        """
                        SELECT 
                            COUNT(*) as total_prs,
                            AVG(overall_score) as avg_score,
                            SUM(security_issues_found) as total_security_issues,
                            SUM(quality_issues_found) as total_quality_issues
                        FROM pr_analyses 
                        WHERE repo_full_name = :repo_full_name
                        """,
                        {"repo_full_name": repo_full_name}
                    )
                    pr_row = result.fetchone()
                    if pr_row:
                        pr_stats = {
                            'total': pr_row.total_prs or 0,
                            'avg_score': float(pr_row.avg_score) if pr_row.avg_score else 0,
                            'security_issues_found': pr_row.total_security_issues or 0,
                            'quality_issues_found': pr_row.total_quality_issues or 0
                        }
                
                # PR summary stats (only for specific repo)
                pr_summary_stats = None
                if repo_full_name:
                    result = await db.execute(
                        """
                        SELECT 
                            COUNT(*) as total_summaries,
                            SUM(CASE WHEN summary_generated = true THEN 1 ELSE 0 END) as successful_summaries,
                            AVG(summary_length) as avg_summary_length,
                            SUM(CASE WHEN rag_system_available = true THEN 1 ELSE 0 END) as rag_generated,
                            SUM(CASE WHEN summary_type = 'fallback' THEN 1 ELSE 0 END) as fallback_summaries
                        FROM pr_summaries 
                        WHERE repo_full_name = :repo_full_name
                        """,
                        {"repo_full_name": repo_full_name}
                    )
                    summary_row = result.fetchone()
                    if summary_row:
                        pr_summary_stats = {
                            'total': summary_row.total_summaries or 0,
                            'successful': summary_row.successful_summaries or 0,
                            'success_rate': (summary_row.successful_summaries or 0) / (summary_row.total_summaries or 1),
                            'avg_summary_length': float(summary_row.avg_summary_length) if summary_row.avg_summary_length else 0,
                            'rag_generated': summary_row.rag_generated or 0,
                            'fallback_summaries': summary_row.fallback_summaries or 0
                        }
                
                stats = RepositoryStats(
                    repo_full_name=repo_full_name or "global",
                    actions=action_stats,
                    issues=issues_stats,
                    pull_requests=pr_stats,
                    pr_summaries=pr_summary_stats,
                    recent_activity_count=recent_activity_count
                )
                
                self.log_operation_complete(operation, start_time, success=True)
                return stats
                
        except Exception as e:
            self.log_error(operation, e, repo=repo_full_name)
            return RepositoryStats(
                repo_full_name=repo_full_name or "global",
                actions={},
                issues=None,
                pull_requests=None,
                pr_summaries=None,
                recent_activity_count=0
            )
    
    async def _update_repository_activity(self, db, repo_full_name: str) -> None:
        """Update repository last activity timestamp."""
        try:
            await repository_service.update_repository_activity(repo_full_name)
        except Exception as e:
            self.logger.error(f"Error updating repository activity {repo_full_name}: {e}")
    
    def _calculate_duration_ms(self, started_at: Optional[datetime]) -> Optional[int]:
        """Calculate duration in milliseconds."""
        if not started_at:
            return None
        
        duration = datetime.utcnow() - started_at
        return int(duration.total_seconds() * 1000)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the analytics service."""
        try:
            # Test database connection and basic query
            async with self.get_db_session() as db:
                result = await db.execute("SELECT COUNT(*) FROM action_logs")
                count = result.fetchone()[0]
            
            return {
                "status": "healthy",
                "total_action_logs": count,
                "database_accessible": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_accessible": False
            }
