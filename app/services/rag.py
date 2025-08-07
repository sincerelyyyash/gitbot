"""
RAG Service

Handles RAG (Retrieval Augmented Generation) operations including:
- Knowledge base initialization and management
- Document indexing and retrieval
- Query processing and response generation
- Vector database management
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio

from .base import BaseService
from app.core.rag_system import (
    initialize_rag_system, 
    query_rag_system,
    get_collection_info,
    delete_collection,
    list_collections,
    _sanitize_collection_name
)
from app.core.github_utils import (
    fetch_all_repository_content,
    fetch_repository_files
)
from app.config import settings

@dataclass
class KnowledgeBaseInfo:
    """Knowledge base information data class."""
    repo_full_name: str
    collection_name: str
    documents_count: int
    last_indexed: Optional[datetime]
    is_initialized: bool
    error_count: int
    last_error: Optional[str]

@dataclass
class QueryResult:
    """Query result data class."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time_ms: float
    tokens_used: Optional[int]

class RAGService(BaseService[QueryResult]):
    """Service for managing RAG operations."""
    
    def __init__(self):
        super().__init__("RAGService")
        self._knowledge_bases: Dict[str, Any] = {}
        self._repo_error_counts: Dict[str, int] = {}
        self._repo_circuit_breaker: Dict[str, datetime] = {}
        self._scheduled_tasks = set()
        self._max_error_count = 5
        self._circuit_breaker_duration = timedelta(minutes=30)
    
    async def get_or_init_repo_knowledge_base(
        self,
        repo_full_name: str, 
        installation_id: int,
        include_current_content: bool = True,
        current_documents_data: list = None,
        force_refresh: bool = False
    ) -> Union[Any, Dict[str, str]]:
        """
        Get or initialize knowledge base for a repository.
        
        Args:
            repo_full_name: Repository full name
            installation_id: GitHub installation ID
            include_current_content: Whether to include current content
            current_documents_data: Current documents data
            force_refresh: Force refresh of knowledge base
            
        Returns:
            Knowledge base object or error dictionary
        """
        operation = "get_or_init_repo_knowledge_base"
        start_time = self.log_operation_start(
            operation, 
            repo=repo_full_name,
            force_refresh=force_refresh
        )
        
        try:
            # Check circuit breaker
            if self._is_circuit_breaker_open(repo_full_name):
                error_msg = f"Circuit breaker open for {repo_full_name}"
                self.log_operation_complete(operation, start_time, success=False)
                return {"error": "circuit_breaker_open", "error_message": error_msg}
            
            # Check if knowledge base exists and is valid
            if repo_full_name in self._knowledge_bases and not force_refresh:
                kb = self._knowledge_bases[repo_full_name]
                if kb and not self._is_knowledge_base_expired(kb):
                    self.log_operation_complete(operation, start_time, success=True, cached=True)
                    return kb
            
            # Initialize new knowledge base
            collection_name = _sanitize_collection_name(repo_full_name)
            
            # Fetch repository content
            content_data = await self._fetch_repository_content(
                repo_full_name, installation_id, include_current_content, current_documents_data
            )
            
            if not content_data:
                error_msg = f"Failed to fetch content for {repo_full_name}"
                self._increment_error_count(repo_full_name)
                self.log_operation_complete(operation, start_time, success=False)
                return {"error": "content_fetch_failed", "error_message": error_msg}
            
            # Initialize RAG system
            rag_system = await initialize_rag_system(
                collection_name=collection_name,
                documents_data=content_data,
                force_refresh=force_refresh
            )
            
            if not rag_system:
                error_msg = f"Failed to initialize RAG system for {repo_full_name}"
                self._increment_error_count(repo_full_name)
                self.log_operation_complete(operation, start_time, success=False)
                return {"error": "rag_init_failed", "error_message": error_msg}
            
            # Store knowledge base
            self._knowledge_bases[repo_full_name] = rag_system
            self._reset_error_count(repo_full_name)
            
            self.log_operation_complete(
                operation, 
                start_time, 
                success=True,
                documents_count=len(content_data)
            )
            
            return rag_system
            
        except Exception as e:
            self._increment_error_count(repo_full_name)
            self.log_error(operation, e, repo=repo_full_name)
            return {"error": "exception", "error_message": str(e)}
    
    async def query_knowledge_base(
        self,
        repo_full_name: str,
        query: str,
        chat_history: List[Tuple[str, str]] = None
    ) -> QueryResult:
        """
        Query a repository's knowledge base.
        
        Args:
            repo_full_name: Repository full name
            query: Query text
            chat_history: Chat history for context
            
        Returns:
            QueryResult with answer and metadata
        """
        operation = "query_knowledge_base"
        start_time = self.log_operation_start(operation, repo=repo_full_name)
        
        try:
            # Get knowledge base
            kb = self._knowledge_bases.get(repo_full_name)
            if not kb:
                return QueryResult(
                    answer="Knowledge base not available for this repository.",
                    sources=[],
                    confidence_score=0.0,
                    processing_time_ms=0,
                    tokens_used=None
                )
            
            # Query RAG system
            result = await query_rag_system(kb, query, chat_history or [])
            
            if isinstance(result, tuple):
                answer, sources = result
            else:
                answer = result
                sources = []
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            query_result = QueryResult(
                answer=answer,
                sources=sources,
                confidence_score=self._calculate_confidence_score(answer, sources),
                processing_time_ms=processing_time,
                tokens_used=self._estimate_tokens_used(query, answer)
            )
            
            self.log_operation_complete(
                operation, 
                start_time, 
                success=True,
                answer_length=len(answer),
                sources_count=len(sources)
            )
            
            return query_result
            
        except Exception as e:
            self.log_error(operation, e, repo=repo_full_name)
            return QueryResult(
                answer="Sorry, I encountered an error while processing your query.",
                sources=[],
                confidence_score=0.0,
                processing_time_ms=0,
                tokens_used=None
            )
    
    async def refresh_repository_knowledge_base(
        self, 
        repo_full_name: str, 
        installation_id: int
    ) -> bool:
        """
        Refresh knowledge base for a repository.
        
        Args:
            repo_full_name: Repository full name
            installation_id: GitHub installation ID
            
        Returns:
            True if successful, False otherwise
        """
        operation = "refresh_repository_knowledge_base"
        start_time = self.log_operation_start(operation, repo=repo_full_name)
        
        try:
            # Force refresh
            result = await self.get_or_init_repo_knowledge_base(
                repo_full_name=repo_full_name,
                installation_id=installation_id,
                include_current_content=True,
                force_refresh=True
            )
            
            success = not isinstance(result, dict) or "error" not in result
            
            self.log_operation_complete(operation, start_time, success=success)
            return success
            
        except Exception as e:
            self.log_error(operation, e, repo=repo_full_name)
            return False
    
    async def schedule_reindex(
        self, 
        repo_full_name: str, 
        installation_id: int, 
        delay_minutes: int = 60
    ) -> bool:
        """
        Schedule reindexing of a repository.
        
        Args:
            repo_full_name: Repository full name
            installation_id: GitHub installation ID
            delay_minutes: Delay before reindexing
            
        Returns:
            True if scheduled successfully, False otherwise
        """
        operation = "schedule_reindex"
        start_time = self.log_operation_start(
            operation, 
            repo=repo_full_name,
            delay_minutes=delay_minutes
        )
        
        try:
            task_key = f"{repo_full_name}:{installation_id}"
            
            # Cancel existing task if any
            if task_key in self._scheduled_tasks:
                self._scheduled_tasks.discard(task_key)
            
            # Schedule new task
            asyncio.create_task(self._delayed_reindex(repo_full_name, installation_id, delay_minutes))
            self._scheduled_tasks.add(task_key)
            
            self.log_operation_complete(operation, start_time, success=True)
            return True
            
        except Exception as e:
            self.log_error(operation, e, repo=repo_full_name)
            return False
    
    async def get_repository_collection_info(self, repo_full_name: str) -> KnowledgeBaseInfo:
        """
        Get information about a repository's knowledge base.
        
        Args:
            repo_full_name: Repository full name
            
        Returns:
            KnowledgeBaseInfo object
        """
        operation = "get_repository_collection_info"
        start_time = self.log_operation_start(operation, repo=repo_full_name)
        
        try:
            collection_name = _sanitize_collection_name(repo_full_name)
            info = await get_collection_info(collection_name)
            
            kb_info = KnowledgeBaseInfo(
                repo_full_name=repo_full_name,
                collection_name=collection_name,
                documents_count=info.get("documents_count", 0),
                last_indexed=info.get("last_indexed"),
                is_initialized=repo_full_name in self._knowledge_bases,
                error_count=self._repo_error_counts.get(repo_full_name, 0),
                last_error=None  # Would need to track this
            )
            
            self.log_operation_complete(operation, start_time, success=True)
            return kb_info
            
        except Exception as e:
            self.log_error(operation, e, repo=repo_full_name)
            return KnowledgeBaseInfo(
                repo_full_name=repo_full_name,
                collection_name=_sanitize_collection_name(repo_full_name),
                documents_count=0,
                last_indexed=None,
                is_initialized=False,
                error_count=self._repo_error_counts.get(repo_full_name, 0),
                last_error=str(e)
            )
    
    async def delete_repository_collection(self, repo_full_name: str) -> bool:
        """
        Delete a repository's knowledge base.
        
        Args:
            repo_full_name: Repository full name
            
        Returns:
            True if deleted successfully, False otherwise
        """
        operation = "delete_repository_collection"
        start_time = self.log_operation_start(operation, repo=repo_full_name)
        
        try:
            collection_name = _sanitize_collection_name(repo_full_name)
            success = await delete_collection(collection_name)
            
            if success:
                # Remove from memory
                self._knowledge_bases.pop(repo_full_name, None)
                self._repo_error_counts.pop(repo_full_name, None)
                self._repo_circuit_breaker.pop(repo_full_name, None)
            
            self.log_operation_complete(operation, start_time, success=success)
            return success
            
        except Exception as e:
            self.log_error(operation, e, repo=repo_full_name)
            return False
    
    async def list_all_repository_collections(self) -> List[KnowledgeBaseInfo]:
        """
        List all repository knowledge bases.
        
        Returns:
            List of KnowledgeBaseInfo objects
        """
        operation = "list_all_repository_collections"
        start_time = self.log_operation_start(operation)
        
        try:
            collections = await list_collections()
            kb_infos = []
            
            for collection in collections:
                repo_full_name = collection.get("repo_full_name", "unknown")
                kb_info = KnowledgeBaseInfo(
                    repo_full_name=repo_full_name,
                    collection_name=collection.get("name", ""),
                    documents_count=collection.get("documents_count", 0),
                    last_indexed=collection.get("last_indexed"),
                    is_initialized=repo_full_name in self._knowledge_bases,
                    error_count=self._repo_error_counts.get(repo_full_name, 0),
                    last_error=None
                )
                kb_infos.append(kb_info)
            
            self.log_operation_complete(operation, start_time, success=True, count=len(kb_infos))
            return kb_infos
            
        except Exception as e:
            self.log_error(operation, e)
            return []
    
    async def cleanup_inactive_collections(self) -> int:
        """
        Clean up inactive knowledge bases.
        
        Returns:
            Number of collections cleaned up
        """
        operation = "cleanup_inactive_collections"
        start_time = self.log_operation_start(operation)
        
        try:
            # Get all collections
            collections = await list_collections()
            cleaned_count = 0
            
            for collection in collections:
                repo_full_name = collection.get("repo_full_name", "unknown")
                last_indexed = collection.get("last_indexed")
                
                # Check if collection is inactive (no activity for 30 days)
                if last_indexed:
                    days_since_indexed = (datetime.utcnow() - last_indexed).days
                    if days_since_indexed > 30:
                        success = await self.delete_repository_collection(repo_full_name)
                        if success:
                            cleaned_count += 1
            
            self.log_operation_complete(operation, start_time, success=True, cleaned_count=cleaned_count)
            return cleaned_count
            
        except Exception as e:
            self.log_error(operation, e)
            return 0
    
    async def _fetch_repository_content(
        self,
        repo_full_name: str,
        installation_id: int,
        include_current_content: bool,
        current_documents_data: list
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch repository content for indexing."""
        try:
            if current_documents_data:
                return current_documents_data
            
            # Fetch all repository content
            content_data = await fetch_all_repository_content(
                repo_full_name=repo_full_name,
                installation_id=installation_id
            )
            
            return content_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch repository content: {e}")
            return None
    
    async def _delayed_reindex(
        self, 
        repo_full_name: str, 
        installation_id: int, 
        delay_minutes: int
    ) -> None:
        """Perform delayed reindexing."""
        try:
            await asyncio.sleep(delay_minutes * 60)
            
            # Remove from scheduled tasks
            task_key = f"{repo_full_name}:{installation_id}"
            self._scheduled_tasks.discard(task_key)
            
            # Perform reindex
            await self.refresh_repository_knowledge_base(repo_full_name, installation_id)
            
        except Exception as e:
            self.logger.error(f"Delayed reindex failed for {repo_full_name}: {e}")
    
    def _is_circuit_breaker_open(self, repo_full_name: str) -> bool:
        """Check if circuit breaker is open for a repository."""
        if repo_full_name not in self._repo_circuit_breaker:
            return False
        
        last_error_time = self._repo_circuit_breaker[repo_full_name]
        return datetime.utcnow() - last_error_time < self._circuit_breaker_duration
    
    def _increment_error_count(self, repo_full_name: str) -> None:
        """Increment error count for a repository."""
        self._repo_error_counts[repo_full_name] = self._repo_error_counts.get(repo_full_name, 0) + 1
        
        # Open circuit breaker if error count exceeds threshold
        if self._repo_error_counts[repo_full_name] >= self._max_error_count:
            self._repo_circuit_breaker[repo_full_name] = datetime.utcnow()
    
    def _reset_error_count(self, repo_full_name: str) -> None:
        """Reset error count for a repository."""
        self._repo_error_counts[repo_full_name] = 0
        self._repo_circuit_breaker.pop(repo_full_name, None)
    
    def _is_knowledge_base_expired(self, kb: Any) -> bool:
        """Check if knowledge base is expired."""
        # This would check if the knowledge base needs refresh
        # For now, return False (never expired)
        return False
    
    def _calculate_confidence_score(self, answer: str, sources: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for a query result."""
        if not answer or answer.strip() == "":
            return 0.0
        
        # Base confidence on answer length and source count
        base_score = min(len(answer) / 100, 1.0)  # Normalize by length
        source_bonus = min(len(sources) * 0.1, 0.3)  # Bonus for sources
        
        return min(base_score + source_bonus, 1.0)
    
    def _estimate_tokens_used(self, query: str, answer: str) -> int:
        """Estimate tokens used for a query."""
        # Rough estimation: 1 token ≈ 4 characters
        query_tokens = len(query) // 4
        answer_tokens = len(answer) // 4
        return query_tokens + answer_tokens
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the RAG service."""
        try:
            # Test basic operations
            collections = await list_collections()
            
            return {
                "status": "healthy",
                "knowledge_bases_count": len(self._knowledge_bases),
                "collections_count": len(collections),
                "scheduled_tasks_count": len(self._scheduled_tasks),
                "error_counts": self._repo_error_counts.copy(),
                "circuit_breakers_count": len(self._repo_circuit_breaker)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
