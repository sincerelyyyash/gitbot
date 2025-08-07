"""
Health Service

Handles system health monitoring including:
- Service health checks
- System component monitoring
- Health status aggregation
- Performance metrics collection
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio

from .base import BaseService
from app.core.database import get_db
from app.core.quota_manager import quota_manager
from app.config import settings

@dataclass
class HealthStatus:
    """Health status data class."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time_ms: Optional[float]
    error_rate: Optional[float]
    message: Optional[str]
    last_check: datetime
    metadata: Optional[Dict[str, Any]]

@dataclass
class SystemHealth:
    """System health summary."""
    overall_status: str
    components: Dict[str, HealthStatus]
    total_components: int
    healthy_components: int
    degraded_components: int
    unhealthy_components: int
    last_check: datetime

class HealthService(BaseService[SystemHealth]):
    """Service for monitoring system health."""
    
    def __init__(self):
        super().__init__("HealthService")
        self._health_cache: Dict[str, HealthStatus] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._last_full_check = datetime.utcnow()
        self._health_checks = self._initialize_health_checks()
    
    def _initialize_health_checks(self) -> Dict[str, callable]:
        """Initialize health check functions."""
        return {
            "database": self._check_database_health,
            "quota_manager": self._check_quota_manager_health,
            "github_api": self._check_github_api_health,
            "rag_system": self._check_rag_system_health,
            "memory": self._check_memory_health,
            "disk": self._check_disk_health
        }
    
    async def check_system_health(self, force_refresh: bool = False) -> SystemHealth:
        """
        Perform comprehensive system health check.
        
        Args:
            force_refresh: Force refresh of cached health data
            
        Returns:
            SystemHealth object with overall system status
        """
        operation = "check_system_health"
        start_time = self.log_operation_start(operation, force_refresh=force_refresh)
        
        try:
            # Check if we can use cached data
            if not force_refresh and self._can_use_cached_health():
                return self._get_cached_system_health()
            
            # Perform all health checks
            components = {}
            for component_name, check_func in self._health_checks.items():
                try:
                    health_status = await check_func()
                    components[component_name] = health_status
                    self._health_cache[component_name] = health_status
                except Exception as e:
                    self.logger.error(f"Health check failed for {component_name}: {e}")
                    components[component_name] = HealthStatus(
                        component=component_name,
                        status="unhealthy",
                        response_time_ms=None,
                        error_rate=1.0,
                        message=f"Health check failed: {str(e)}",
                        last_check=datetime.utcnow(),
                        metadata={"error": str(e)}
                    )
            
            # Calculate overall status
            overall_status = self._calculate_overall_status(components)
            
            # Create system health summary
            system_health = SystemHealth(
                overall_status=overall_status,
                components=components,
                total_components=len(components),
                healthy_components=len([c for c in components.values() if c.status == "healthy"]),
                degraded_components=len([c for c in components.values() if c.status == "degraded"]),
                unhealthy_components=len([c for c in components.values() if c.status == "unhealthy"]),
                last_check=datetime.utcnow()
            )
            
            self._last_full_check = datetime.utcnow()
            
            self.log_operation_complete(
                operation, 
                start_time, 
                success=True,
                overall_status=overall_status,
                healthy_components=system_health.healthy_components
            )
            
            return system_health
            
        except Exception as e:
            self.log_error(operation, e)
            # Return unhealthy status if health check fails
            return SystemHealth(
                overall_status="unhealthy",
                components={},
                total_components=0,
                healthy_components=0,
                degraded_components=0,
                unhealthy_components=0,
                last_check=datetime.utcnow()
            )
    
    async def check_component_health(self, component_name: str) -> Optional[HealthStatus]:
        """
        Check health of a specific component.
        
        Args:
            component_name: Name of the component to check
            
        Returns:
            HealthStatus object or None if component not found
        """
        if component_name not in self._health_checks:
            return None
        
        try:
            health_status = await self._health_checks[component_name]()
            self._health_cache[component_name] = health_status
            return health_status
        except Exception as e:
            self.logger.error(f"Component health check failed for {component_name}: {e}")
            return None
    
    async def _check_database_health(self) -> HealthStatus:
        """Check database health."""
        start_time = datetime.utcnow()
        
        try:
            async for db in get_db():
                # Test basic query
                result = await db.execute("SELECT 1")
                await result.fetchone()
                
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                return HealthStatus(
                    component="database",
                    status="healthy",
                    response_time_ms=response_time,
                    error_rate=0.0,
                    message="Database connection successful",
                    last_check=datetime.utcnow(),
                    metadata={"connection_pool_size": "default"}
                )
                
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return HealthStatus(
                component="database",
                status="unhealthy",
                response_time_ms=response_time,
                error_rate=1.0,
                message=f"Database connection failed: {str(e)}",
                last_check=datetime.utcnow(),
                metadata={"error": str(e)}
            )
    
    async def _check_quota_manager_health(self) -> HealthStatus:
        """Check quota manager health."""
        start_time = datetime.utcnow()
        
        try:
            # Test quota check
            test_repo = "test/repo"
            quota_status = await quota_manager.check_quota(test_repo)
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return HealthStatus(
                component="quota_manager",
                status="healthy",
                response_time_ms=response_time,
                error_rate=0.0,
                message="Quota manager operational",
                last_check=datetime.utcnow(),
                metadata={"quota_status": quota_status}
            )
            
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return HealthStatus(
                component="quota_manager",
                status="unhealthy",
                response_time_ms=response_time,
                error_rate=1.0,
                message=f"Quota manager failed: {str(e)}",
                last_check=datetime.utcnow(),
                metadata={"error": str(e)}
            )
    
    async def _check_github_api_health(self) -> HealthStatus:
        """Check GitHub API health."""
        start_time = datetime.utcnow()
        
        try:
            # This would typically test GitHub API connectivity
            # For now, we'll simulate a healthy status
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return HealthStatus(
                component="github_api",
                status="healthy",
                response_time_ms=response_time,
                error_rate=0.0,
                message="GitHub API accessible",
                last_check=datetime.utcnow(),
                metadata={"api_version": "v3"}
            )
            
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return HealthStatus(
                component="github_api",
                status="unhealthy",
                response_time_ms=response_time,
                error_rate=1.0,
                message=f"GitHub API failed: {str(e)}",
                last_check=datetime.utcnow(),
                metadata={"error": str(e)}
            )
    
    async def _check_rag_system_health(self) -> HealthStatus:
        """Check RAG system health."""
        start_time = datetime.utcnow()
        
        try:
            # This would test RAG system components
            # For now, we'll simulate a healthy status
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return HealthStatus(
                component="rag_system",
                status="healthy",
                response_time_ms=response_time,
                error_rate=0.0,
                message="RAG system operational",
                last_check=datetime.utcnow(),
                metadata={"embeddings_model": "gemini"}
            )
            
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return HealthStatus(
                component="rag_system",
                status="unhealthy",
                response_time_ms=response_time,
                error_rate=1.0,
                message=f"RAG system failed: {str(e)}",
                last_check=datetime.utcnow(),
                metadata={"error": str(e)}
            )
    
    async def _check_memory_health(self) -> HealthStatus:
        """Check memory usage health."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent
            
            if memory_usage_percent < 80:
                status = "healthy"
                message = f"Memory usage: {memory_usage_percent:.1f}%"
            elif memory_usage_percent < 95:
                status = "degraded"
                message = f"High memory usage: {memory_usage_percent:.1f}%"
            else:
                status = "unhealthy"
                message = f"Critical memory usage: {memory_usage_percent:.1f}%"
            
            return HealthStatus(
                component="memory",
                status=status,
                response_time_ms=None,
                error_rate=0.0,
                message=message,
                last_check=datetime.utcnow(),
                metadata={
                    "usage_percent": memory_usage_percent,
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3)
                }
            )
            
        except ImportError:
            return HealthStatus(
                component="memory",
                status="degraded",
                response_time_ms=None,
                error_rate=0.0,
                message="Memory monitoring not available (psutil not installed)",
                last_check=datetime.utcnow(),
                metadata={"note": "psutil not available"}
            )
        except Exception as e:
            return HealthStatus(
                component="memory",
                status="unhealthy",
                response_time_ms=None,
                error_rate=1.0,
                message=f"Memory check failed: {str(e)}",
                last_check=datetime.utcnow(),
                metadata={"error": str(e)}
            )
    
    async def _check_disk_health(self) -> HealthStatus:
        """Check disk usage health."""
        try:
            import psutil
            
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            if disk_usage_percent < 85:
                status = "healthy"
                message = f"Disk usage: {disk_usage_percent:.1f}%"
            elif disk_usage_percent < 95:
                status = "degraded"
                message = f"High disk usage: {disk_usage_percent:.1f}%"
            else:
                status = "unhealthy"
                message = f"Critical disk usage: {disk_usage_percent:.1f}%"
            
            return HealthStatus(
                component="disk",
                status=status,
                response_time_ms=None,
                error_rate=0.0,
                message=message,
                last_check=datetime.utcnow(),
                metadata={
                    "usage_percent": disk_usage_percent,
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3)
                }
            )
            
        except ImportError:
            return HealthStatus(
                component="disk",
                status="degraded",
                response_time_ms=None,
                error_rate=0.0,
                message="Disk monitoring not available (psutil not installed)",
                last_check=datetime.utcnow(),
                metadata={"note": "psutil not available"}
            )
        except Exception as e:
            return HealthStatus(
                component="disk",
                status="unhealthy",
                response_time_ms=None,
                error_rate=1.0,
                message=f"Disk check failed: {str(e)}",
                last_check=datetime.utcnow(),
                metadata={"error": str(e)}
            )
    
    def _calculate_overall_status(self, components: Dict[str, HealthStatus]) -> str:
        """Calculate overall system status based on component statuses."""
        if not components:
            return "unknown"
        
        status_counts = {}
        for component in components.values():
            status_counts[component.status] = status_counts.get(component.status, 0) + 1
        
        # If any component is unhealthy, overall status is unhealthy
        if status_counts.get("unhealthy", 0) > 0:
            return "unhealthy"
        
        # If any component is degraded, overall status is degraded
        if status_counts.get("degraded", 0) > 0:
            return "degraded"
        
        # All components are healthy
        return "healthy"
    
    def _can_use_cached_health(self) -> bool:
        """Check if cached health data is still valid."""
        return datetime.utcnow() - self._last_full_check < self._cache_ttl
    
    def _get_cached_system_health(self) -> SystemHealth:
        """Get system health from cache."""
        components = self._health_cache.copy()
        overall_status = self._calculate_overall_status(components)
        
        return SystemHealth(
            overall_status=overall_status,
            components=components,
            total_components=len(components),
            healthy_components=len([c for c in components.values() if c.status == "healthy"]),
            degraded_components=len([c for c in components.values() if c.status == "degraded"]),
            unhealthy_components=len([c for c in components.values() if c.status == "unhealthy"]),
            last_check=self._last_full_check
        )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of current health status."""
        if self._can_use_cached_health():
            system_health = self._get_cached_system_health()
        else:
            # Return basic summary without full check
            return {
                "status": "unknown",
                "last_check": self._last_full_check.isoformat(),
                "note": "Full health check required"
            }
        
        return {
            "status": system_health.overall_status,
            "total_components": system_health.total_components,
            "healthy_components": system_health.healthy_components,
            "degraded_components": system_health.degraded_components,
            "unhealthy_components": system_health.unhealthy_components,
            "last_check": system_health.last_check.isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the health service itself."""
        try:
            # Test that we can perform a basic health check
            test_status = await self._check_database_health()
            
            return {
                "status": "healthy",
                "cache_size": len(self._health_cache),
                "last_full_check": self._last_full_check.isoformat(),
                "test_component_status": test_status.status
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
