"""
Dashboard API endpoints for GitBot analytics and metrics.

Provides comprehensive dashboard APIs for tracking gitbot actions including:
- Repository statistics and metrics
- Activity timelines and trends
- System health monitoring
- Performance analytics
- Usage insights
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.analytics_service import analytics_service
from app.config import settings

router = APIRouter()
logger = logging.getLogger("dashboard_api")

# Pydantic models for request/response

class DashboardOverview(BaseModel):
    """Dashboard overview response model."""
    total_repositories: int
    active_repositories: int
    total_actions_today: int
    total_actions_this_week: int
    success_rate_today: float
    avg_response_time_ms: float
    system_health: str
    last_updated: str

class RepositoryStats(BaseModel):
    """Repository statistics response model."""
    repo_full_name: str
    total_actions: int
    recent_activity_count: int
    success_rate: float
    avg_response_time_ms: float
    issues_analyzed: int
    prs_analyzed: int
    pr_summaries_generated: int
    duplicates_found: int
    security_issues_found: int
    last_activity: Optional[str]
    indexed_at: Optional[str]

class ActivityTimelineItem(BaseModel):
    """Activity timeline item model."""
    date: str
    action_type: str
    count: int
    successful: int
    success_rate: float

class RecentAction(BaseModel):
    """Recent action model."""
    id: int
    repo_full_name: Optional[str]
    action_type: str
    action_subtype: Optional[str]
    target_type: Optional[str]
    target_number: Optional[int]
    status: str
    success: Optional[bool]
    started_at: Optional[str]
    completed_at: Optional[str]
    duration_ms: Optional[int]
    response_posted: bool
    error_message: Optional[str]

class SystemHealthComponent(BaseModel):
    """System health component model."""
    status: str
    response_time_ms: Optional[float]
    error_rate: Optional[float]
    message: Optional[str]
    checked_at: str

class SystemHealth(BaseModel):
    """System health response model."""
    overall_status: str
    components: Dict[str, SystemHealthComponent]
    last_check: str

# Authentication helper
def verify_admin_access(admin_token: Optional[str] = None):
    """Verify admin access for dashboard endpoints."""
    if settings.admin_token and admin_token != settings.admin_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token"
        )

# Dashboard Overview Endpoints

@router.get("/overview", response_model=DashboardOverview)
async def get_dashboard_overview(admin_token: Optional[str] = None):
    """Get dashboard overview with key metrics."""
    verify_admin_access(admin_token)
    
    try:
        # Get system-wide statistics
        stats = await analytics_service.get_repository_stats()
        health = await analytics_service.get_system_health()
        
        # Calculate overview metrics
        total_actions_today = 0
        total_actions_week = 0
        success_rate_today = 0.0
        avg_response_time = 0.0
        
        # Get today's activity
        today_timeline = await analytics_service.get_activity_timeline(days=1)
        if today_timeline:
            total_actions_today = sum(item['count'] for item in today_timeline)
            successful_today = sum(item['successful'] for item in today_timeline)
            success_rate_today = successful_today / total_actions_today if total_actions_today > 0 else 0.0
        
        # Get week's activity
        week_timeline = await analytics_service.get_activity_timeline(days=7)
        if week_timeline:
            total_actions_week = sum(item['count'] for item in week_timeline)
        
        # Calculate average response time from actions
        if 'actions' in stats:
            total_duration = 0
            total_count = 0
            for action_type, action_stats in stats['actions'].items():
                if action_stats['avg_duration_ms'] > 0:
                    total_duration += action_stats['avg_duration_ms'] * action_stats['total']
                    total_count += action_stats['total']
            avg_response_time = total_duration / total_count if total_count > 0 else 0.0
        
        return DashboardOverview(
            total_repositories=0,  # TODO: Add repository count
            active_repositories=0,  # TODO: Add active repository count
            total_actions_today=total_actions_today,
            total_actions_this_week=total_actions_week,
            success_rate_today=success_rate_today,
            avg_response_time_ms=avg_response_time,
            system_health=health.get('overall_status', 'unknown'),
            last_updated=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.exception("Error getting dashboard overview")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dashboard overview: {str(e)}"
        )

# Repository Analytics

@router.get("/repositories", response_model=List[RepositoryStats])
async def get_repositories_stats(
    admin_token: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100)
):
    """Get statistics for all repositories."""
    verify_admin_access(admin_token)
    
    try:
        # TODO: Implement repository listing with stats
        # For now, return empty list
        return []
        
    except Exception as e:
        logger.exception("Error getting repositories stats")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get repositories stats: {str(e)}"
        )

@router.get("/repositories/{owner}/{repo}", response_model=RepositoryStats)
async def get_repository_stats(
    owner: str,
    repo: str,
    admin_token: Optional[str] = None
):
    """Get detailed statistics for a specific repository."""
    verify_admin_access(admin_token)
    
    try:
        repo_full_name = f"{owner}/{repo}"
        stats = await analytics_service.get_repository_stats(repo_full_name)
        
        # Extract metrics from stats
        total_actions = sum(action['total'] for action in stats.get('actions', {}).values())
        success_count = sum(action['successful'] for action in stats.get('actions', {}).values())
        success_rate = success_count / total_actions if total_actions > 0 else 0.0
        
        # Calculate average response time
        total_duration = 0
        total_count = 0
        for action_stats in stats.get('actions', {}).values():
            if action_stats['avg_duration_ms'] > 0:
                total_duration += action_stats['avg_duration_ms'] * action_stats['total']
                total_count += action_stats['total']
        avg_response_time = total_duration / total_count if total_count > 0 else 0.0
        
        return RepositoryStats(
            repo_full_name=repo_full_name,
            total_actions=total_actions,
            recent_activity_count=stats.get('recent_activity_count', 0),
            success_rate=success_rate,
            avg_response_time_ms=avg_response_time,
            issues_analyzed=stats.get('issues', {}).get('total', 0),
            prs_analyzed=stats.get('pull_requests', {}).get('total', 0),
            pr_summaries_generated=stats.get('pr_summaries', {}).get('total', 0),
            duplicates_found=stats.get('issues', {}).get('duplicates', 0),
            security_issues_found=stats.get('pull_requests', {}).get('security_issues_found', 0),
            last_activity=None,  # TODO: Add from repository model
            indexed_at=None  # TODO: Add from repository model
        )
        
    except Exception as e:
        logger.exception(f"Error getting repository stats for {owner}/{repo}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get repository stats: {str(e)}"
        )

@router.get("/repositories/{owner}/{repo}/pr-summaries")
async def get_repository_pr_summaries(
    owner: str,
    repo: str,
    admin_token: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100)
):
    """Get detailed PR summary statistics for a specific repository."""
    verify_admin_access(admin_token)
    
    try:
        repo_full_name = f"{owner}/{repo}"
        stats = await analytics_service.get_repository_stats(repo_full_name)
        
        pr_summaries = stats.get('pr_summaries', {})
        
        return {
            "repo_full_name": repo_full_name,
            "total_summaries": pr_summaries.get('total', 0),
            "successful_summaries": pr_summaries.get('successful', 0),
            "success_rate": pr_summaries.get('success_rate', 0.0),
            "avg_summary_length": pr_summaries.get('avg_summary_length', 0),
            "rag_generated": pr_summaries.get('rag_generated', 0),
            "fallback_summaries": pr_summaries.get('fallback_summaries', 0),
            "rag_usage_rate": pr_summaries.get('rag_generated', 0) / max(pr_summaries.get('total', 1), 1)
        }
        
    except Exception as e:
        logger.exception(f"Error getting PR summary stats for {owner}/{repo}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get PR summary stats: {str(e)}"
        )

# Activity and Timeline

@router.get("/activity/timeline", response_model=List[ActivityTimelineItem])
async def get_activity_timeline(
    repo_full_name: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365),
    admin_token: Optional[str] = None
):
    """Get activity timeline for dashboard charts."""
    verify_admin_access(admin_token)
    
    try:
        timeline = await analytics_service.get_activity_timeline(
            repo_full_name=repo_full_name,
            days=days
        )
        
        return [ActivityTimelineItem(**item) for item in timeline]
        
    except Exception as e:
        logger.exception("Error getting activity timeline")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get activity timeline: {str(e)}"
        )

@router.get("/activity/recent", response_model=List[RecentAction])
async def get_recent_actions(
    repo_full_name: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    admin_token: Optional[str] = None
):
    """Get recent actions for the dashboard."""
    verify_admin_access(admin_token)
    
    try:
        actions = await analytics_service.get_recent_actions(
            repo_full_name=repo_full_name,
            limit=limit
        )
        
        return [RecentAction(**action) for action in actions]
        
    except Exception as e:
        logger.exception("Error getting recent actions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recent actions: {str(e)}"
        )

# System Health and Monitoring

@router.get("/health", response_model=SystemHealth)
async def get_system_health(admin_token: Optional[str] = None):
    """Get current system health status."""
    verify_admin_access(admin_token)
    
    try:
        health_data = await analytics_service.get_system_health()
        
        # Convert components to proper model format
        components = {}
        for component, data in health_data.get('components', {}).items():
            components[component] = SystemHealthComponent(**data)
        
        return SystemHealth(
            overall_status=health_data['overall_status'],
            components=components,
            last_check=health_data['last_check']
        )
        
    except Exception as e:
        logger.exception("Error getting system health")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system health: {str(e)}"
        )

# Analytics and Insights

@router.get("/analytics/pr-summaries")
async def get_pr_summary_analytics(
    repo_full_name: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365),
    admin_token: Optional[str] = None
):
    """Get PR summary analytics and insights."""
    verify_admin_access(admin_token)
    
    try:
        stats = await analytics_service.get_repository_stats(repo_full_name)
        timeline = await analytics_service.get_activity_timeline(repo_full_name, days)
        
        # Get PR summary specific stats
        pr_summaries = stats.get('pr_summaries', {})
        
        # Filter timeline for PR summary actions
        pr_summary_timeline = [
            item for item in timeline 
            if item['action_type'] == 'pr_summary'
        ]
        
        # Calculate PR summary trends
        total_summaries = sum(item['count'] for item in pr_summary_timeline)
        successful_summaries = sum(item['successful'] for item in pr_summary_timeline)
        overall_success_rate = successful_summaries / total_summaries if total_summaries > 0 else 0.0
        
        # Calculate RAG vs fallback usage
        rag_generated = pr_summaries.get('rag_generated', 0)
        fallback_summaries = pr_summaries.get('fallback_summaries', 0)
        total_generated = rag_generated + fallback_summaries
        rag_usage_rate = rag_generated / total_generated if total_generated > 0 else 0.0
        
        return JSONResponse({
            "pr_summary_analytics": {
                "period_days": days,
                "total_summaries": total_summaries,
                "successful_summaries": successful_summaries,
                "overall_success_rate": overall_success_rate,
                "rag_generated": rag_generated,
                "fallback_summaries": fallback_summaries,
                "rag_usage_rate": rag_usage_rate,
                "avg_summary_length": pr_summaries.get('avg_summary_length', 0),
                "repository": repo_full_name
            },
            "timeline_data": pr_summary_timeline,
            "generated_at": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.exception("Error getting PR summary analytics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get PR summary analytics: {str(e)}"
        )

@router.get("/analytics/summary")
async def get_analytics_summary(
    repo_full_name: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365),
    admin_token: Optional[str] = None
):
    """Get analytics summary with key insights."""
    verify_admin_access(admin_token)
    
    try:
        stats = await analytics_service.get_repository_stats(repo_full_name)
        timeline = await analytics_service.get_activity_timeline(repo_full_name, days)
        
        # Calculate trends and insights
        total_actions = sum(item['count'] for item in timeline)
        total_successful = sum(item['successful'] for item in timeline)
        overall_success_rate = total_successful / total_actions if total_actions > 0 else 0.0
        
        # Group by action type for insights
        action_breakdown = {}
        for item in timeline:
            action_type = item['action_type']
            if action_type not in action_breakdown:
                action_breakdown[action_type] = {'count': 0, 'successful': 0}
            action_breakdown[action_type]['count'] += item['count']
            action_breakdown[action_type]['successful'] += item['successful']
        
        # Calculate success rates per action type
        for action_type, data in action_breakdown.items():
            data['success_rate'] = data['successful'] / data['count'] if data['count'] > 0 else 0.0
        
        return JSONResponse({
            "summary": {
                "period_days": days,
                "total_actions": total_actions,
                "total_successful": total_successful,
                "overall_success_rate": overall_success_rate,
                "repository": repo_full_name
            },
            "action_breakdown": action_breakdown,
            "repository_stats": stats,
            "generated_at": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.exception("Error getting analytics summary")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics summary: {str(e)}"
        )

@router.get("/analytics/performance")
async def get_performance_metrics(
    repo_full_name: Optional[str] = Query(None),
    days: int = Query(7, ge=1, le=30),
    admin_token: Optional[str] = None
):
    """Get performance metrics and trends."""
    verify_admin_access(admin_token)
    
    try:
        # Get recent actions for performance analysis
        actions = await analytics_service.get_recent_actions(
            repo_full_name=repo_full_name,
            limit=1000  # Get more actions for better analysis
        )
        
        # Filter actions by date range
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_actions = [
            action for action in actions 
            if action.get('started_at') and 
            datetime.fromisoformat(action['started_at'].replace('Z', '+00:00')) >= cutoff_date
        ]
        
        # Calculate performance metrics
        total_actions = len(recent_actions)
        successful_actions = len([a for a in recent_actions if a.get('success') is True])
        failed_actions = len([a for a in recent_actions if a.get('success') is False])
        
        # Response time analysis
        response_times = [a['duration_ms'] for a in recent_actions if a.get('duration_ms')]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Performance by action type
        action_performance = {}
        for action in recent_actions:
            action_type = action['action_type']
            if action_type not in action_performance:
                action_performance[action_type] = {
                    'count': 0,
                    'successful': 0,
                    'total_duration_ms': 0,
                    'durations': []
                }
            
            action_performance[action_type]['count'] += 1
            if action.get('success'):
                action_performance[action_type]['successful'] += 1
            if action.get('duration_ms'):
                action_performance[action_type]['total_duration_ms'] += action['duration_ms']
                action_performance[action_type]['durations'].append(action['duration_ms'])
        
        # Calculate averages and percentiles
        for action_type, perf in action_performance.items():
            perf['success_rate'] = perf['successful'] / perf['count'] if perf['count'] > 0 else 0
            perf['avg_duration_ms'] = perf['total_duration_ms'] / perf['count'] if perf['count'] > 0 else 0
            
            # Calculate percentiles
            if perf['durations']:
                sorted_durations = sorted(perf['durations'])
                count = len(sorted_durations)
                perf['p50_duration_ms'] = sorted_durations[count // 2] if count > 0 else 0
                perf['p95_duration_ms'] = sorted_durations[int(count * 0.95)] if count > 0 else 0
                perf['p99_duration_ms'] = sorted_durations[int(count * 0.99)] if count > 0 else 0
            
            # Clean up temporary data
            del perf['total_duration_ms']
            del perf['durations']
        
        return JSONResponse({
            "performance_summary": {
                "period_days": days,
                "total_actions": total_actions,
                "successful_actions": successful_actions,
                "failed_actions": failed_actions,
                "success_rate": successful_actions / total_actions if total_actions > 0 else 0,
                "avg_response_time_ms": avg_response_time,
                "repository": repo_full_name
            },
            "action_performance": action_performance,
            "generated_at": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.exception("Error getting performance metrics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )

# Export and Reporting

@router.get("/export/actions")
async def export_actions(
    repo_full_name: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    format: str = Query("json", regex="^(json|csv)$"),
    admin_token: Optional[str] = None
):
    """Export action logs for external analysis."""
    verify_admin_access(admin_token)
    
    try:
        # Get actions with date filtering
        actions = await analytics_service.get_recent_actions(
            repo_full_name=repo_full_name,
            limit=10000  # Large limit for export
        )
        
        # Apply date filtering if provided
        if start_date or end_date:
            filtered_actions = []
            for action in actions:
                if not action.get('started_at'):
                    continue
                    
                action_date = datetime.fromisoformat(action['started_at'].replace('Z', '+00:00'))
                
                if start_date:
                    start_dt = datetime.fromisoformat(start_date)
                    if action_date < start_dt:
                        continue
                
                if end_date:
                    end_dt = datetime.fromisoformat(end_date)
                    if action_date > end_dt:
                        continue
                
                filtered_actions.append(action)
            
            actions = filtered_actions
        
        if format == "json":
            return JSONResponse({
                "actions": actions,
                "total_count": len(actions),
                "exported_at": datetime.utcnow().isoformat(),
                "filters": {
                    "repo_full_name": repo_full_name,
                    "start_date": start_date,
                    "end_date": end_date
                }
            })
        else:
            # TODO: Implement CSV export
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="CSV export not yet implemented"
            )
        
    except Exception as e:
        logger.exception("Error exporting actions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export actions: {str(e)}"
        ) 