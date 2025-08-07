"""
Quota Manager

Provides comprehensive quota management for external API services including:
- Token usage tracking and monitoring
- Quota limit enforcement
- Usage analytics and reporting
- Quota alerts and notifications
- Multi-service quota management
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum
import aiofiles
from .base import BaseCore, core_operation
from app.config import settings

class QuotaService(Enum):
    """Supported quota services."""
    GEMINI = "gemini"
    GITHUB = "github"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

@dataclass
class QuotaConfig:
    """Configuration for quota management."""
    daily_limit: int
    hourly_limit: Optional[int] = None
    minute_limit: Optional[int] = None
    alert_threshold: float = 0.8  # 80% of quota
    critical_threshold: float = 0.95  # 95% of quota
    reset_time: str = "00:00"  # Daily reset time (UTC)

@dataclass
class UsageEntry:
    """Usage entry for tracking token consumption."""
    timestamp: datetime
    tokens_used: int
    operation: str
    service: QuotaService
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QuotaStatus:
    """Current quota status for a service."""
    service: QuotaService
    current_usage: int
    daily_limit: int
    hourly_limit: Optional[int]
    minute_limit: Optional[int]
    usage_percent: float
    remaining: int
    reset_time: datetime
    status: str  # "normal", "warning", "critical", "exceeded"
    last_updated: datetime

class QuotaManager(BaseCore):
    """
    Quota manager for tracking and managing API usage limits.
    
    Features:
    - Multi-service quota tracking
    - Real-time usage monitoring
    - Quota alerts and notifications
    - Usage analytics and reporting
    - Automatic quota enforcement
    """
    
    def __init__(self, quota_file_path: str = "quota_data.json"):
        super().__init__("quota_manager")
        
        # Quota file configuration
        self.quota_file_path = os.path.abspath(quota_file_path)
        
        # Default quota configurations
        self._quota_configs = {
            QuotaService.GEMINI: QuotaConfig(
                daily_limit=60000,
                hourly_limit=5000,
                minute_limit=100,
                alert_threshold=0.8,
                critical_threshold=0.95
            ),
            QuotaService.GITHUB: QuotaConfig(
                daily_limit=5000,  # GitHub API calls
                hourly_limit=500,
                minute_limit=10,
                alert_threshold=0.8,
                critical_threshold=0.95
            ),
            QuotaService.OPENAI: QuotaConfig(
                daily_limit=100000,
                hourly_limit=10000,
                minute_limit=200,
                alert_threshold=0.8,
                critical_threshold=0.95
            ),
            QuotaService.ANTHROPIC: QuotaConfig(
                daily_limit=100000,
                hourly_limit=10000,
                minute_limit=200,
                alert_threshold=0.8,
                critical_threshold=0.95
            )
        }
        
        # Usage data storage
        self._usage_data: Dict[str, Dict[str, Any]] = {}
        
        # Quota metrics
        self._quota_metrics = {
            "total_usage_checks": 0,
            "quota_exceeded_count": 0,
            "quota_warnings": 0,
            "quota_critical_alerts": 0,
            "usage_updates": 0
        }
        
        # Ensure quota file exists
        self._ensure_quota_file_exists()
        
        # Load existing data
        self._load_usage_data()
    
    def _ensure_quota_file_exists(self):
        """Ensure the quota file exists and is properly initialized."""
        os.makedirs(os.path.dirname(self.quota_file_path), exist_ok=True)
        if not os.path.exists(self.quota_file_path):
            with open(self.quota_file_path, 'w') as f:
                json.dump({}, f)
    
    def _load_usage_data(self):
        """Load usage data from file."""
        try:
            with open(self.quota_file_path, 'r') as f:
                content = f.read()
                self._usage_data = json.loads(content) if content else {}
        except Exception as error:
            self.logger.error(f"Error loading quota data: {error}")
            self._usage_data = {}
    
    @core_operation("save_usage_data")
    async def _save_usage_data(self):
        """Save usage data to file."""
        try:
            async with aiofiles.open(self.quota_file_path, 'w') as f:
                await f.write(json.dumps(self._usage_data, indent=2, default=str))
        except Exception as error:
            self.logger.error(f"Error saving quota usage: {error}")
            raise
    
    def _clean_old_data(self, repo_data: Dict[str, Any], service: QuotaService):
        """Remove usage data older than 24 hours."""
        cutoff_time = (datetime.utcnow() - timedelta(days=1)).isoformat()
        
        if 'usage_history' in repo_data:
            repo_data['usage_history'] = [
                entry for entry in repo_data.get('usage_history', [])
                if entry.get('timestamp', '') > cutoff_time
            ]
    
    def _get_quota_config(self, service: QuotaService) -> QuotaConfig:
        """Get quota configuration for a service."""
        return self._quota_configs.get(service, self._quota_configs[QuotaService.GEMINI])
    
    @core_operation("check_quota")
    async def check_quota(self, repo_full_name: str, service: QuotaService = QuotaService.GEMINI) -> bool:
        """Check if the repository has remaining quota for a service."""
        self._quota_metrics["total_usage_checks"] += 1
        
        # Load latest data
        self._load_usage_data()
        
        # Get quota config
        quota_config = self._get_quota_config(service)
        
        # Check if repository exists in data
        if repo_full_name not in self._usage_data:
            return True
        
        repo_data = self._usage_data[repo_full_name]
        self._clean_old_data(repo_data, service)
        
        # Calculate total tokens used in the last 24 hours
        total_tokens = sum(
            entry.get('tokens', 0) for entry in repo_data.get('usage_history', [])
            if entry.get('service', service.value) == service.value
        )
        
        # Check daily limit
        if total_tokens >= quota_config.daily_limit:
            self._quota_metrics["quota_exceeded_count"] += 1
            return False
        
        # Check hourly limit
        if quota_config.hourly_limit:
            hour_ago = datetime.utcnow() - timedelta(hours=1)
            hourly_tokens = sum(
                entry.get('tokens', 0) for entry in repo_data.get('usage_history', [])
                if (entry.get('service', service.value) == service.value and
                    datetime.fromisoformat(entry.get('timestamp', '')) > hour_ago)
            )
            if hourly_tokens >= quota_config.hourly_limit:
                self._quota_metrics["quota_exceeded_count"] += 1
                return False
        
        # Check minute limit
        if quota_config.minute_limit:
            minute_ago = datetime.utcnow() - timedelta(minutes=1)
            minute_tokens = sum(
                entry.get('tokens', 0) for entry in repo_data.get('usage_history', [])
                if (entry.get('service', service.value) == service.value and
                    datetime.fromisoformat(entry.get('timestamp', '')) > minute_ago)
            )
            if minute_tokens >= quota_config.minute_limit:
                self._quota_metrics["quota_exceeded_count"] += 1
                return False
        
        return True
    
    @core_operation("update_usage")
    async def update_usage(self, repo_full_name: str, tokens_used: int, 
                          service: QuotaService = QuotaService.GEMINI,
                          operation: str = "unknown",
                          metadata: Optional[Dict[str, Any]] = None):
        """Update token usage for a repository and service."""
        self._quota_metrics["usage_updates"] += 1
        
        # Load latest data
        self._load_usage_data()
        
        # Initialize repository data if not exists
        if repo_full_name not in self._usage_data:
            self._usage_data[repo_full_name] = {'usage_history': []}
        
        repo_data = self._usage_data[repo_full_name]
        self._clean_old_data(repo_data, service)
        
        # Create usage entry
        usage_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'tokens': tokens_used,
            'service': service.value,
            'operation': operation,
            'metadata': metadata or {}
        }
        
        # Add to usage history
        repo_data['usage_history'].append(usage_entry)
        
        # Check for quota alerts
        await self._check_quota_alerts(repo_full_name, service)
        
        # Save data
        await self._save_usage_data()
        
        self.logger.debug(f"Updated usage for {repo_full_name}: {tokens_used} tokens for {service.value}")
    
    async def _check_quota_alerts(self, repo_full_name: str, service: QuotaService):
        """Check for quota alerts and trigger notifications."""
        quota_config = self._get_quota_config(service)
        repo_data = self._usage_data[repo_full_name]
        
        # Calculate current usage
        total_tokens = sum(
            entry.get('tokens', 0) for entry in repo_data.get('usage_history', [])
            if entry.get('service', service.value) == service.value
        )
        
        usage_percent = total_tokens / quota_config.daily_limit
        
        # Check alert thresholds
        if usage_percent >= quota_config.critical_threshold:
            self._quota_metrics["quota_critical_alerts"] += 1
            self.logger.critical(
                f"CRITICAL: Quota usage for {repo_full_name} ({service.value}) is at {usage_percent:.1%}"
            )
        elif usage_percent >= quota_config.alert_threshold:
            self._quota_metrics["quota_warnings"] += 1
            self.logger.warning(
                f"WARNING: Quota usage for {repo_full_name} ({service.value}) is at {usage_percent:.1%}"
            )
    
    @core_operation("get_usage_stats")
    async def get_usage_stats(self, repo_full_name: str, service: QuotaService = QuotaService.GEMINI) -> Dict[str, Any]:
        """Get usage statistics for a repository and service."""
        self._load_usage_data()
        
        quota_config = self._get_quota_config(service)
        
        if repo_full_name not in self._usage_data:
            return {
                'tokens_used_today': 0,
                'tokens_used_hour': 0,
                'tokens_used_minute': 0,
                'quota_remaining': quota_config.daily_limit,
                'usage_percent': 0.0,
                'quota_reset_in': '24 hours',
                'status': 'normal'
            }
        
        repo_data = self._usage_data[repo_full_name]
        self._clean_old_data(repo_data, service)
        
        # Calculate usage for different time periods
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        minute_ago = now - timedelta(minutes=1)
        
        daily_tokens = sum(
            entry.get('tokens', 0) for entry in repo_data.get('usage_history', [])
            if entry.get('service', service.value) == service.value
        )
        
        hourly_tokens = sum(
            entry.get('tokens', 0) for entry in repo_data.get('usage_history', [])
            if (entry.get('service', service.value) == service.value and
                datetime.fromisoformat(entry.get('timestamp', '')) > hour_ago)
        )
        
        minute_tokens = sum(
            entry.get('tokens', 0) for entry in repo_data.get('usage_history', [])
            if (entry.get('service', service.value) == service.value and
                datetime.fromisoformat(entry.get('timestamp', '')) > minute_ago)
        )
        
        # Calculate usage percentage and status
        usage_percent = daily_tokens / quota_config.daily_limit
        remaining = max(0, quota_config.daily_limit - daily_tokens)
        
        # Determine status
        if usage_percent >= quota_config.critical_threshold:
            status = 'critical'
        elif usage_percent >= quota_config.alert_threshold:
            status = 'warning'
        else:
            status = 'normal'
        
        # Calculate reset time
        reset_time = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        reset_in = reset_time - now
        reset_hours = round(reset_in.total_seconds() / 3600, 1)
        
        return {
            'tokens_used_today': daily_tokens,
            'tokens_used_hour': hourly_tokens,
            'tokens_used_minute': minute_tokens,
            'quota_remaining': remaining,
            'usage_percent': round(usage_percent * 100, 2),
            'quota_reset_in': f"{reset_hours} hours",
            'status': status,
            'daily_limit': quota_config.daily_limit,
            'hourly_limit': quota_config.hourly_limit,
            'minute_limit': quota_config.minute_limit
        }
    
    def get_quota_status(self, service: QuotaService) -> QuotaStatus:
        """Get current quota status for a service across all repositories."""
        quota_config = self._get_quota_config(service)
        self._load_usage_data()
        
        # Calculate total usage across all repositories
        total_usage = 0
        for repo_data in self._usage_data.values():
            self._clean_old_data(repo_data, service)
            total_usage += sum(
                entry.get('tokens', 0) for entry in repo_data.get('usage_history', [])
                if entry.get('service', service.value) == service.value
            )
        
        usage_percent = total_usage / quota_config.daily_limit
        remaining = max(0, quota_config.daily_limit - total_usage)
        
        # Determine status
        if usage_percent >= quota_config.critical_threshold:
            status = 'critical'
        elif usage_percent >= quota_config.alert_threshold:
            status = 'warning'
        elif total_usage >= quota_config.daily_limit:
            status = 'exceeded'
        else:
            status = 'normal'
        
        # Calculate reset time
        now = datetime.utcnow()
        reset_time = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        return QuotaStatus(
            service=service,
            current_usage=total_usage,
            daily_limit=quota_config.daily_limit,
            hourly_limit=quota_config.hourly_limit,
            minute_limit=quota_config.minute_limit,
            usage_percent=round(usage_percent * 100, 2),
            remaining=remaining,
            reset_time=reset_time,
            status=status,
            last_updated=now
        )
    
    def get_all_quota_statuses(self) -> Dict[str, QuotaStatus]:
        """Get quota status for all services."""
        return {
            service.value: self.get_quota_status(service)
            for service in QuotaService
        }
    
    def get_usage_analytics(self, repo_full_name: str, service: QuotaService = QuotaService.GEMINI) -> Dict[str, Any]:
        """Get detailed usage analytics for a repository."""
        self._load_usage_data()
        
        if repo_full_name not in self._usage_data:
            return {"error": "Repository not found"}
        
        repo_data = self._usage_data[repo_full_name]
        self._clean_old_data(repo_data, service)
        
        # Filter entries for the service
        service_entries = [
            entry for entry in repo_data.get('usage_history', [])
            if entry.get('service', service.value) == service.value
        ]
        
        if not service_entries:
            return {"error": "No usage data found"}
        
        # Calculate analytics
        total_tokens = sum(entry.get('tokens', 0) for entry in service_entries)
        avg_tokens_per_request = total_tokens / len(service_entries)
        
        # Group by operation
        operation_stats = {}
        for entry in service_entries:
            operation = entry.get('operation', 'unknown')
            tokens = entry.get('tokens', 0)
            if operation not in operation_stats:
                operation_stats[operation] = {'count': 0, 'total_tokens': 0}
            operation_stats[operation]['count'] += 1
            operation_stats[operation]['total_tokens'] += tokens
        
        # Calculate hourly usage for the last 24 hours
        hourly_usage = {}
        for i in range(24):
            hour_start = datetime.utcnow() - timedelta(hours=23-i)
            hour_end = hour_start + timedelta(hours=1)
            
            hour_tokens = sum(
                entry.get('tokens', 0) for entry in service_entries
                if hour_start <= datetime.fromisoformat(entry.get('timestamp', '')) < hour_end
            )
            hourly_usage[hour_start.strftime('%H:00')] = hour_tokens
        
        return {
            'total_requests': len(service_entries),
            'total_tokens': total_tokens,
            'avg_tokens_per_request': round(avg_tokens_per_request, 2),
            'operation_stats': operation_stats,
            'hourly_usage': hourly_usage,
            'last_24_hours': True
        }
    
    async def _basic_health_check(self):
        """Basic health check for quota manager."""
        try:
            # Test file access
            if not os.path.exists(self.quota_file_path):
                raise Exception("Quota file not accessible")
            
            # Test data loading
            self._load_usage_data()
            
        except Exception as error:
            raise Exception(f"Quota manager health check failed: {error}")
    
    def get_quota_metrics(self) -> Dict[str, Any]:
        """Get quota management metrics."""
        return {
            "component": self.component_name,
            "quota_metrics": self._quota_metrics.copy(),
            "quota_configs": {
                service.value: {
                    "daily_limit": config.daily_limit,
                    "hourly_limit": config.hourly_limit,
                    "minute_limit": config.minute_limit,
                    "alert_threshold": config.alert_threshold,
                    "critical_threshold": config.critical_threshold
                }
                for service, config in self._quota_configs.items()
            },
            "total_repositories": len(self._usage_data)
        }
    
    async def reset_metrics(self):
        """Reset quota metrics."""
        self._quota_metrics = {
            "total_usage_checks": 0,
            "quota_exceeded_count": 0,
            "quota_warnings": 0,
            "quota_critical_alerts": 0,
            "usage_updates": 0
        }
        self.logger.info("Quota metrics reset")
    
    def clear_usage_data(self, repo_full_name: Optional[str] = None):
        """Clear usage data for a repository or all repositories."""
        if repo_full_name:
            if repo_full_name in self._usage_data:
                del self._usage_data[repo_full_name]
                self.logger.info(f"Cleared usage data for {repo_full_name}")
        else:
            self._usage_data.clear()
            self.logger.info("Cleared all usage data")
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get quota manager configuration."""
        config = super().get_configuration()
        config.update({
            "quota_file_path": self.quota_file_path,
            "supported_services": [service.value for service in QuotaService],
            "quota_configs": {
                service.value: {
                    "daily_limit": config.daily_limit,
                    "hourly_limit": config.hourly_limit,
                    "minute_limit": config.minute_limit,
                    "alert_threshold": config.alert_threshold,
                    "critical_threshold": config.critical_threshold
                }
                for service, config in self._quota_configs.items()
            }
        })
        return config

# Legacy compatibility functions
async def check_quota(repo_full_name: str) -> bool:
    """Legacy function for backward compatibility."""
    return await quota_manager.check_quota(repo_full_name, QuotaService.GEMINI)

async def update_usage(repo_full_name: str, tokens_used: int):
    """Legacy function for backward compatibility."""
    await quota_manager.update_usage(repo_full_name, tokens_used, QuotaService.GEMINI)

async def get_usage_stats(repo_full_name: str) -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    return await quota_manager.get_usage_stats(repo_full_name, QuotaService.GEMINI)

# Global quota manager instance
quota_manager = QuotaManager() 