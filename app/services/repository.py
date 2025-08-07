"""
Repository Service

Handles repository-related operations including:
- Repository registration and management
- Activity tracking
- Indexing status management
- Repository metadata handling
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base import BaseService
from app.models.analytics import Repository
from app.core.github_utils import get_github_app_installation_client
from app.config import settings

@dataclass
class RepositoryInfo:
    """Repository information data class."""
    full_name: str
    installation_id: int
    owner: str
    name: str
    is_active: bool
    indexed_at: Optional[datetime]
    last_activity: Optional[datetime]
    metadata: Optional[Dict[str, Any]]

class RepositoryService(BaseService[RepositoryInfo]):
    """Service for managing repository operations."""
    
    def __init__(self):
        super().__init__("RepositoryService")
        self._repo_cache: Dict[str, RepositoryInfo] = {}
        self._cache_ttl = timedelta(minutes=30)
        self._last_cache_cleanup = datetime.utcnow()
    
    async def register_repository(
        self, 
        full_name: str, 
        installation_id: int,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Register a new repository or update existing one.
        
        Args:
            full_name: Repository full name (owner/repo)
            installation_id: GitHub installation ID
            metadata: Optional repository metadata
            
        Returns:
            Repository ID
        """
        operation = "register_repository"
        start_time = self.log_operation_start(operation, repo=full_name)
        
        try:
            async with self.get_db_session() as db:
                # Check if repository exists
                result = await db.execute(
                    "SELECT * FROM repositories WHERE full_name = :full_name",
                    {"full_name": full_name}
                )
                repo = result.fetchone()
                
                if repo:
                    # Update existing
                    await db.execute(
                        """
                        UPDATE repositories 
                        SET installation_id = :installation_id, 
                            is_active = true, 
                            last_activity = :now, 
                            updated_at = :now,
                            metadata_json = COALESCE(:metadata, metadata_json)
                        WHERE full_name = :full_name
                        """,
                        {
                            "installation_id": installation_id,
                            "now": datetime.utcnow(),
                            "metadata": metadata,
                            "full_name": full_name
                        }
                    )
                    repo_id = repo.id
                else:
                    # Create new
                    owner, name = full_name.split('/', 1)
                    result = await db.execute(
                        """
                        INSERT INTO repositories 
                        (full_name, installation_id, owner, name, metadata_json, created_at, updated_at)
                        VALUES (:full_name, :installation_id, :owner, :name, :metadata, :now, :now)
                        RETURNING id
                        """,
                        {
                            "full_name": full_name,
                            "installation_id": installation_id,
                            "owner": owner,
                            "name": name,
                            "metadata": metadata,
                            "now": datetime.utcnow()
                        }
                    )
                    repo_id = result.fetchone().id
                
                await db.commit()
                
                # Update cache
                self._update_cache(full_name, installation_id, owner, name, metadata)
                
                self.log_operation_complete(operation, start_time, success=True, repo_id=repo_id)
                return repo_id
                
        except Exception as e:
            self.log_error(operation, e, repo=full_name)
            raise
    
    async def get_repository_info(self, full_name: str) -> Optional[RepositoryInfo]:
        """
        Get repository information from cache or database.
        
        Args:
            full_name: Repository full name
            
        Returns:
            RepositoryInfo object or None if not found
        """
        # Check cache first
        if full_name in self._repo_cache:
            cached_info = self._repo_cache[full_name]
            if datetime.utcnow() - cached_info.last_activity < self._cache_ttl:
                return cached_info
        
        # Fetch from database
        operation = "get_repository_info"
        start_time = self.log_operation_start(operation, repo=full_name)
        
        try:
            async with self.get_db_session() as db:
                result = await db.execute(
                    "SELECT * FROM repositories WHERE full_name = :full_name",
                    {"full_name": full_name}
                )
                repo = result.fetchone()
                
                if repo:
                    repo_info = RepositoryInfo(
                        full_name=repo.full_name,
                        installation_id=repo.installation_id,
                        owner=repo.owner,
                        name=repo.name,
                        is_active=repo.is_active,
                        indexed_at=repo.indexed_at,
                        last_activity=repo.last_activity,
                        metadata=repo.metadata_json
                    )
                    
                    # Update cache
                    self._repo_cache[full_name] = repo_info
                    
                    self.log_operation_complete(operation, start_time, success=True)
                    return repo_info
                
                self.log_operation_complete(operation, start_time, success=True, found=False)
                return None
                
        except Exception as e:
            self.log_error(operation, e, repo=full_name)
            return None
    
    async def update_repository_activity(self, full_name: str) -> bool:
        """
        Update repository last activity timestamp.
        
        Args:
            full_name: Repository full name
            
        Returns:
            True if updated successfully, False otherwise
        """
        operation = "update_repository_activity"
        start_time = self.log_operation_start(operation, repo=full_name)
        
        try:
            async with self.get_db_session() as db:
                result = await db.execute(
                    """
                    UPDATE repositories 
                    SET last_activity = :now 
                    WHERE full_name = :full_name
                    """,
                    {
                        "now": datetime.utcnow(),
                        "full_name": full_name
                    }
                )
                await db.commit()
                
                # Update cache
                if full_name in self._repo_cache:
                    self._repo_cache[full_name].last_activity = datetime.utcnow()
                
                updated = result.rowcount > 0
                self.log_operation_complete(operation, start_time, success=updated)
                return updated
                
        except Exception as e:
            self.log_error(operation, e, repo=full_name)
            return False
    
    async def mark_repository_indexed(self, full_name: str) -> bool:
        """
        Mark repository as indexed.
        
        Args:
            full_name: Repository full name
            
        Returns:
            True if updated successfully, False otherwise
        """
        operation = "mark_repository_indexed"
        start_time = self.log_operation_start(operation, repo=full_name)
        
        try:
            async with self.get_db_session() as db:
                result = await db.execute(
                    """
                    UPDATE repositories 
                    SET indexed_at = :now 
                    WHERE full_name = :full_name
                    """,
                    {
                        "now": datetime.utcnow(),
                        "full_name": full_name
                    }
                )
                await db.commit()
                
                # Update cache
                if full_name in self._repo_cache:
                    self._repo_cache[full_name].indexed_at = datetime.utcnow()
                
                updated = result.rowcount > 0
                self.log_operation_complete(operation, start_time, success=updated)
                return updated
                
        except Exception as e:
            self.log_error(operation, e, repo=full_name)
            return False
    
    async def get_active_repositories(self, limit: int = 100) -> List[RepositoryInfo]:
        """
        Get list of active repositories.
        
        Args:
            limit: Maximum number of repositories to return
            
        Returns:
            List of RepositoryInfo objects
        """
        operation = "get_active_repositories"
        start_time = self.log_operation_start(operation, limit=limit)
        
        try:
            async with self.get_db_session() as db:
                result = await db.execute(
                    """
                    SELECT * FROM repositories 
                    WHERE is_active = true 
                    ORDER BY last_activity DESC 
                    LIMIT :limit
                    """,
                    {"limit": limit}
                )
                
                repositories = []
                for row in result.fetchall():
                    repo_info = RepositoryInfo(
                        full_name=row.full_name,
                        installation_id=row.installation_id,
                        owner=row.owner,
                        name=row.name,
                        is_active=row.is_active,
                        indexed_at=row.indexed_at,
                        last_activity=row.last_activity,
                        metadata=row.metadata_json
                    )
                    repositories.append(repo_info)
                    
                    # Update cache
                    self._repo_cache[row.full_name] = repo_info
                
                self.log_operation_complete(operation, start_time, success=True, count=len(repositories))
                return repositories
                
        except Exception as e:
            self.log_error(operation, e)
            return []
    
    async def get_github_client(self, repo_full_name: str) -> Optional[Any]:
        """
        Get GitHub client for a repository.
        
        Args:
            repo_full_name: Repository full name
            
        Returns:
            GitHub client or None if not found
        """
        repo_info = await self.get_repository_info(repo_full_name)
        if not repo_info:
            return None
        
        return await get_github_app_installation_client(
            settings.github_app_id,
            settings.github_private_key,
            repo_info.installation_id
        )
    
    async def is_repository_indexed(self, full_name: str) -> bool:
        """
        Check if a repository is indexed.
        
        Args:
            full_name: Repository full name
            
        Returns:
            True if indexed, False otherwise
        """
        repo_info = await self.get_repository_info(full_name)
        return repo_info is not None and repo_info.indexed_at is not None
    
    async def get_repository_stats(self, full_name: str) -> Dict[str, Any]:
        """
        Get repository statistics.
        
        Args:
            full_name: Repository full name
            
        Returns:
            Dictionary containing repository statistics
        """
        operation = "get_repository_stats"
        start_time = self.log_operation_start(operation, repo=full_name)
        
        try:
            repo_info = await self.get_repository_info(full_name)
            if not repo_info:
                return {}
            
            stats = {
                "full_name": repo_info.full_name,
                "owner": repo_info.owner,
                "name": repo_info.name,
                "is_active": repo_info.is_active,
                "indexed": repo_info.indexed_at is not None,
                "indexed_at": repo_info.indexed_at.isoformat() if repo_info.indexed_at else None,
                "last_activity": repo_info.last_activity.isoformat() if repo_info.last_activity else None,
                "installation_id": repo_info.installation_id,
                "metadata": repo_info.metadata
            }
            
            self.log_operation_complete(operation, start_time, success=True)
            return stats
            
        except Exception as e:
            self.log_error(operation, e, repo=full_name)
            return {}
    
    def _update_cache(self, full_name: str, installation_id: int, owner: str, 
                     name: str, metadata: Optional[Dict]) -> None:
        """Update the repository cache."""
        self._repo_cache[full_name] = RepositoryInfo(
            full_name=full_name,
            installation_id=installation_id,
            owner=owner,
            name=name,
            is_active=True,
            indexed_at=None,
            last_activity=datetime.utcnow(),
            metadata=metadata
        )
    
    async def cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        now = datetime.utcnow()
        if now - self._last_cache_cleanup < timedelta(minutes=5):
            return
        
        expired_keys = [
            key for key, info in self._repo_cache.items()
            if now - info.last_activity > self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._repo_cache[key]
        
        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        self._last_cache_cleanup = now
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the repository service."""
        try:
            # Test database connection
            async with self.get_db_session() as db:
                await db.execute("SELECT 1")
            
            return {
                "status": "healthy",
                "cache_size": len(self._repo_cache),
                "last_cache_cleanup": self._last_cache_cleanup.isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
