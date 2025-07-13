"""
Analytics Service for GitBot Dashboard

Provides comprehensive tracking and analytics for gitbot actions including:
- Action logging and tracking
- Performance metrics calculation
- Usage analytics and quotas
- Repository insights
- System health monitoring
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.analytics import (
    Repository, ActionLog, IssueAnalysis, PRAnalysis, 
    IndexingJob, UsageMetrics, SystemHealth
)

logger = logging.getLogger("analytics_service")

class AnalyticsService:
    """Service for tracking and analyzing gitbot actions."""
    
    def __init__(self):
        self.logger = logging.getLogger("analytics_service")

    # Repository Management
    async def register_repository(
        self, 
        full_name: str, 
        installation_id: int,
        metadata: Optional[Dict] = None
    ) -> int:
        """Register a new repository or update existing one."""
        async for db in get_db():
            try:
                # Check if repository exists
                result = await db.execute(
                    select(Repository).where(Repository.full_name == full_name)
                )
                repo = result.scalar_one_or_none()
                
                if repo:
                    # Update existing
                    repo.installation_id = installation_id
                    repo.is_active = True
                    repo.last_activity = datetime.utcnow()
                    repo.updated_at = datetime.utcnow()
                    if metadata:
                        repo.metadata_json = metadata
                    await db.commit()
                    return repo.id
                else:
                    # Create new
                    owner, name = full_name.split('/', 1)
                    repo = Repository(
                        full_name=full_name,
                        installation_id=installation_id,
                        owner=owner,
                        name=name,
                        metadata_json=metadata
                    )
                    db.add(repo)
                    await db.commit()
                    await db.refresh(repo)
                    return repo.id
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Error registering repository {full_name}: {e}")
                raise

    async def update_repository_activity(self, full_name: str) -> None:
        """Update repository last activity timestamp."""
        async for db in get_db():
            try:
                result = await db.execute(
                    select(Repository).where(Repository.full_name == full_name)
                )
                repo = result.scalar_one_or_none()
                if repo:
                    repo.last_activity = datetime.utcnow()
                    await db.commit()
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Error updating repository activity {full_name}: {e}")

    async def mark_repository_indexed(self, full_name: str) -> None:
        """Mark repository as indexed."""
        async for db in get_db():
            try:
                result = await db.execute(
                    select(Repository).where(Repository.full_name == full_name)
                )
                repo = result.scalar_one_or_none()
                if repo:
                    repo.indexed_at = datetime.utcnow()
                    await db.commit()
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Error marking repository indexed {full_name}: {e}")

    # Action Logging
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
        """Log the start of an action and return the action log ID."""
        async for db in get_db():
            try:
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
                    await self.update_repository_activity(repo_full_name)
                
                return action_log.id
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Error logging action start: {e}")
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
        """Log the completion of an action."""
        async for db in get_db():
            try:
                result = await db.execute(
                    select(ActionLog).where(ActionLog.id == action_log_id)
                )
                action_log = result.scalar_one_or_none()
                
                if action_log:
                    action_log.completed_at = datetime.utcnow()
                    action_log.success = success
                    action_log.status = 'completed' if success else 'failed'
                    action_log.error_message = error_message
                    action_log.response_posted = response_posted
                    action_log.tokens_used = tokens_used
                    action_log.api_calls_made = api_calls_made
                    
                    # Calculate duration
                    if action_log.started_at:
                        duration = action_log.completed_at - action_log.started_at
                        action_log.duration_ms = int(duration.total_seconds() * 1000)
                    
                    # Update metadata
                    if metadata:
                        if action_log.metadata_json:
                            action_log.metadata_json.update(metadata)
                        else:
                            action_log.metadata_json = metadata
                    
                    await db.commit()
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Error logging action completion: {e}")

    # Issue Analysis Tracking
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
        """Log detailed issue analysis results."""
        async for db in get_db():
            try:
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
                return analysis.id
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Error logging issue analysis: {e}")
                raise

    # PR Analysis Tracking
    async def log_pr_analysis(
        self,
        action_log_id: Optional[int],
        repo_full_name: str,
        pr_number: int,
        pr_title: Optional[str] = None,
        files_changed: int = 0,
        lines_added: int = 0,
        lines_deleted: int = 0,
        security_issues_found: int = 0,
        quality_issues_found: int = 0,
        complexity_issues_found: int = 0,
        potential_bugs_found: int = 0,
        duplicate_functionality_found: int = 0,
        overall_score: Optional[int] = None,
        review_priority: Optional[str] = None,
        languages_detected: Optional[List[str]] = None,
        review_posted: bool = False,
        suggestions_count: int = 0,
        labels_added: Optional[List[str]] = None
    ) -> int:
        """Log detailed PR analysis results."""
        async for db in get_db():
            try:
                analysis = PRAnalysis(
                    action_log_id=action_log_id,
                    repo_full_name=repo_full_name,
                    pr_number=pr_number,
                    pr_title=pr_title,
                    files_changed=files_changed,
                    lines_added=lines_added,
                    lines_deleted=lines_deleted,
                    security_issues_found=security_issues_found,
                    quality_issues_found=quality_issues_found,
                    complexity_issues_found=complexity_issues_found,
                    potential_bugs_found=potential_bugs_found,
                    duplicate_functionality_found=duplicate_functionality_found,
                    overall_score=overall_score,
                    review_priority=review_priority,
                    languages_detected=languages_detected,
                    review_posted=review_posted,
                    suggestions_count=suggestions_count,
                    labels_added=labels_added
                )
                db.add(analysis)
                await db.commit()
                await db.refresh(analysis)
                return analysis.id
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Error logging PR analysis: {e}")
                raise

    # Indexing Job Tracking
    async def log_indexing_job(
        self,
        action_log_id: Optional[int],
        repo_full_name: str,
        installation_id: int,
        job_type: str = 'initial',
        trigger: Optional[str] = None,
        priority: int = 1,
        force_refresh: bool = False,
        status: str = 'queued'
    ) -> int:
        """Log indexing job details."""
        async for db in get_db():
            try:
                job = IndexingJob(
                    action_log_id=action_log_id,
                    repo_full_name=repo_full_name,
                    installation_id=installation_id,
                    job_type=job_type,
                    trigger=trigger,
                    priority=priority,
                    force_refresh=force_refresh,
                    status=status
                )
                db.add(job)
                await db.commit()
                await db.refresh(job)
                return job.id
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Error logging indexing job: {e}")
                raise

    async def update_indexing_job(
        self,
        job_id: int,
        status: Optional[str] = None,
        files_processed: Optional[int] = None,
        documents_created: Optional[int] = None,
        embeddings_generated: Optional[int] = None,
        error_message: Optional[str] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None
    ) -> None:
        """Update indexing job status and metrics."""
        async for db in get_db():
            try:
                result = await db.execute(
                    select(IndexingJob).where(IndexingJob.id == job_id)
                )
                job = result.scalar_one_or_none()
                
                if job:
                    if status:
                        job.status = status
                    if files_processed is not None:
                        job.files_processed = files_processed
                    if documents_created is not None:
                        job.documents_created = documents_created
                    if embeddings_generated is not None:
                        job.embeddings_generated = embeddings_generated
                    if error_message:
                        job.error_message = error_message
                        job.last_error_at = datetime.utcnow()
                    if started_at:
                        job.started_at = started_at
                    if completed_at:
                        job.completed_at = completed_at
                        # Calculate duration
                        if job.started_at:
                            duration = completed_at - job.started_at
                            job.duration_seconds = int(duration.total_seconds())
                    
                    await db.commit()
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Error updating indexing job: {e}")

    # Analytics and Metrics
    async def get_repository_stats(self, repo_full_name: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive repository statistics."""
        async for db in get_db():
            try:
                stats = {}
                
                # Base query filter
                repo_filter = ActionLog.repo_full_name == repo_full_name if repo_full_name else True
                
                # Action counts by type
                result = await db.execute(
                    select(
                        ActionLog.action_type,
                        func.count(ActionLog.id).label('count'),
                        func.sum(func.case((ActionLog.success == True, 1), else_=0)).label('successful'),
                        func.avg(ActionLog.duration_ms).label('avg_duration_ms')
                    )
                    .where(repo_filter)
                    .group_by(ActionLog.action_type)
                )
                
                action_stats = {}
                for row in result:
                    action_stats[row.action_type] = {
                        'total': row.count,
                        'successful': row.successful or 0,
                        'success_rate': (row.successful or 0) / row.count if row.count > 0 else 0,
                        'avg_duration_ms': float(row.avg_duration_ms) if row.avg_duration_ms else 0
                    }
                
                stats['actions'] = action_stats
                
                # Recent activity (last 30 days)
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                result = await db.execute(
                    select(func.count(ActionLog.id))
                    .where(and_(repo_filter, ActionLog.started_at >= thirty_days_ago))
                )
                stats['recent_activity_count'] = result.scalar() or 0
                
                # Issue analysis stats
                if repo_full_name:
                    result = await db.execute(
                        select(
                            func.count(IssueAnalysis.id).label('total_issues'),
                            func.sum(func.case((IssueAnalysis.is_question == True, 1), else_=0)).label('questions'),
                            func.sum(func.case((IssueAnalysis.is_bug_report == True, 1), else_=0)).label('bug_reports'),
                            func.sum(func.case((IssueAnalysis.is_duplicate == True, 1), else_=0)).label('duplicates'),
                            func.avg(IssueAnalysis.similar_issues_found).label('avg_similar_found')
                        )
                        .where(IssueAnalysis.repo_full_name == repo_full_name)
                    )
                    issue_row = result.first()
                    if issue_row:
                        stats['issues'] = {
                            'total': issue_row.total_issues or 0,
                            'questions': issue_row.questions or 0,
                            'bug_reports': issue_row.bug_reports or 0,
                            'duplicates': issue_row.duplicates or 0,
                            'avg_similar_found': float(issue_row.avg_similar_found) if issue_row.avg_similar_found else 0
                        }
                
                # PR analysis stats
                if repo_full_name:
                    result = await db.execute(
                        select(
                            func.count(PRAnalysis.id).label('total_prs'),
                            func.avg(PRAnalysis.overall_score).label('avg_score'),
                            func.sum(PRAnalysis.security_issues_found).label('total_security_issues'),
                            func.sum(PRAnalysis.quality_issues_found).label('total_quality_issues')
                        )
                        .where(PRAnalysis.repo_full_name == repo_full_name)
                    )
                    pr_row = result.first()
                    if pr_row:
                        stats['pull_requests'] = {
                            'total': pr_row.total_prs or 0,
                            'avg_score': float(pr_row.avg_score) if pr_row.avg_score else 0,
                            'security_issues_found': pr_row.total_security_issues or 0,
                            'quality_issues_found': pr_row.total_quality_issues or 0
                        }
                
                return stats
                
            except Exception as e:
                self.logger.error(f"Error getting repository stats: {e}")
                return {}

    async def get_activity_timeline(
        self, 
        repo_full_name: Optional[str] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get activity timeline for the dashboard."""
        async for db in get_db():
            try:
                start_date = datetime.utcnow() - timedelta(days=days)
                repo_filter = ActionLog.repo_full_name == repo_full_name if repo_full_name else True
                
                result = await db.execute(
                    select(
                        func.date(ActionLog.started_at).label('date'),
                        ActionLog.action_type,
                        func.count(ActionLog.id).label('count'),
                        func.sum(func.case((ActionLog.success == True, 1), else_=0)).label('successful')
                    )
                    .where(and_(repo_filter, ActionLog.started_at >= start_date))
                    .group_by(func.date(ActionLog.started_at), ActionLog.action_type)
                    .order_by(asc(func.date(ActionLog.started_at)))
                )
                
                timeline = []
                for row in result:
                    timeline.append({
                        'date': row.date.isoformat(),
                        'action_type': row.action_type,
                        'count': row.count,
                        'successful': row.successful or 0,
                        'success_rate': (row.successful or 0) / row.count if row.count > 0 else 0
                    })
                
                return timeline
                
            except Exception as e:
                self.logger.error(f"Error getting activity timeline: {e}")
                return []

    async def get_recent_actions(
        self, 
        repo_full_name: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent actions for the dashboard."""
        async for db in get_db():
            try:
                repo_filter = ActionLog.repo_full_name == repo_full_name if repo_full_name else True
                
                result = await db.execute(
                    select(ActionLog)
                    .where(repo_filter)
                    .order_by(desc(ActionLog.started_at))
                    .limit(limit)
                )
                
                actions = []
                for action in result.scalars():
                    actions.append({
                        'id': action.id,
                        'repo_full_name': action.repo_full_name,
                        'action_type': action.action_type,
                        'action_subtype': action.action_subtype,
                        'target_type': action.target_type,
                        'target_number': action.target_number,
                        'status': action.status,
                        'success': action.success,
                        'started_at': action.started_at.isoformat() if action.started_at else None,
                        'completed_at': action.completed_at.isoformat() if action.completed_at else None,
                        'duration_ms': action.duration_ms,
                        'response_posted': action.response_posted,
                        'error_message': action.error_message
                    })
                
                return actions
                
            except Exception as e:
                self.logger.error(f"Error getting recent actions: {e}")
                return []

    async def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        async for db in get_db():
            try:
                # Get latest health check for each component
                result = await db.execute(
                    select(SystemHealth)
                    .where(
                        SystemHealth.checked_at >= datetime.utcnow() - timedelta(hours=1)
                    )
                    .order_by(desc(SystemHealth.checked_at))
                )
                
                health_data = {}
                for health in result.scalars():
                    if health.component not in health_data:
                        health_data[health.component] = {
                            'status': health.status,
                            'response_time_ms': health.response_time_ms,
                            'error_rate': health.error_rate,
                            'message': health.message,
                            'checked_at': health.checked_at.isoformat()
                        }
                
                # Calculate overall system status
                if not health_data:
                    overall_status = 'unknown'
                elif any(h['status'] == 'unhealthy' for h in health_data.values()):
                    overall_status = 'unhealthy'
                elif any(h['status'] == 'degraded' for h in health_data.values()):
                    overall_status = 'degraded'
                else:
                    overall_status = 'healthy'
                
                return {
                    'overall_status': overall_status,
                    'components': health_data,
                    'last_check': datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error getting system health: {e}")
                return {
                    'overall_status': 'error',
                    'components': {},
                    'last_check': datetime.utcnow().isoformat(),
                    'error': str(e)
                }

    async def log_system_health(
        self,
        component: str,
        status: str,
        response_time_ms: Optional[float] = None,
        error_rate: Optional[float] = None,
        message: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Log system health check result."""
        async for db in get_db():
            try:
                health = SystemHealth(
                    component=component,
                    status=status,
                    response_time_ms=response_time_ms,
                    error_rate=error_rate,
                    message=message,
                    metadata_json=metadata
                )
                db.add(health)
                await db.commit()
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Error logging system health: {e}")

# Global analytics service instance
analytics_service = AnalyticsService() 