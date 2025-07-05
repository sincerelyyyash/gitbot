import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from app.core.github_utils import get_github_app_installation_client
from app.services.rag_service import get_or_init_repo_knowledge_base, reset_error_count
from app.config import settings
from app.core.quota_manager import quota_manager

logger = logging.getLogger("indexing_service")

class IndexingStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"

@dataclass
class IndexingJob:
    repo_full_name: str
    installation_id: int
    status: IndexingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    priority: int = 1  # Lower number = higher priority
    force_refresh: bool = False
    
    def to_dict(self) -> dict:
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, IndexingStatus):
                data[key] = value.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'IndexingJob':
        # Convert ISO strings back to datetime objects
        for key in ['created_at', 'started_at', 'completed_at']:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        if 'status' in data:
            data['status'] = IndexingStatus(data['status'])
        return cls(**data)

class IndexingQueue:
    """Thread-safe indexing queue with persistence."""
    
    def __init__(self, persist_file: Optional[str] = None):
        self.persist_file = persist_file or os.path.join(
            settings.data_dir, "indexing_queue.json"
        )
        self._queue: List[IndexingJob] = []
        self._processing: Set[str] = set()
        self._lock = asyncio.Lock()
        self._load_queue()
    
    def _load_queue(self):
        """Load queue state from persistent storage."""
        try:
            if os.path.exists(self.persist_file):
                with open(self.persist_file, 'r') as f:
                    data = json.load(f)
                    self._queue = [IndexingJob.from_dict(job_data) for job_data in data]
                    
                    # Reset in-progress jobs to pending on startup
                    for job in self._queue:
                        if job.status == IndexingStatus.IN_PROGRESS:
                            job.status = IndexingStatus.PENDING
                            job.started_at = None
                    
                logger.info(f"Loaded {len(self._queue)} indexing jobs from {self.persist_file}")
        except Exception as e:
            logger.error(f"Error loading indexing queue: {e}")
            self._queue = []
    
    def _save_queue(self):
        """Save queue state to persistent storage."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.persist_file), exist_ok=True)
            
            with open(self.persist_file, 'w') as f:
                json.dump([job.to_dict() for job in self._queue], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving indexing queue: {e}")
    
    async def add_job(
        self, 
        repo_full_name: str, 
        installation_id: int, 
        priority: int = 1,
        force_refresh: bool = False
    ) -> bool:
        """Add a new indexing job to the queue."""
        async with self._lock:
            # Check if job already exists
            existing_job = next(
                (job for job in self._queue 
                 if job.repo_full_name == repo_full_name 
                 and job.status in [IndexingStatus.PENDING, IndexingStatus.IN_PROGRESS, IndexingStatus.RETRYING]),
                None
            )
            
            if existing_job:
                # Update priority if new job has higher priority (lower number)
                if priority < existing_job.priority:
                    existing_job.priority = priority
                    logger.info(f"Updated priority for {repo_full_name} to {priority}")
                    self._sort_queue()
                    self._save_queue()
                return False
            
            # Create new job
            job = IndexingJob(
                repo_full_name=repo_full_name,
                installation_id=installation_id,
                status=IndexingStatus.PENDING,
                created_at=datetime.utcnow(),
                priority=priority,
                force_refresh=force_refresh
            )
            
            self._queue.append(job)
            self._sort_queue()
            self._save_queue()
            
            logger.info(f"Added indexing job for {repo_full_name} with priority {priority}")
            return True
    
    def _sort_queue(self):
        """Sort queue by priority and creation time."""
        self._queue.sort(key=lambda x: (x.priority, x.created_at))
    
    async def get_next_job(self) -> Optional[IndexingJob]:
        """Get the next job to process."""
        async with self._lock:
            # Find next pending job that's not being processed
            for job in self._queue:
                if (job.status == IndexingStatus.PENDING 
                    and job.repo_full_name not in self._processing):
                    
                    job.status = IndexingStatus.IN_PROGRESS
                    job.started_at = datetime.utcnow()
                    self._processing.add(job.repo_full_name)
                    self._save_queue()
                    return job
            
            return None
    
    async def complete_job(self, repo_full_name: str, success: bool, error_message: Optional[str] = None):
        """Mark a job as completed or failed."""
        async with self._lock:
            job = next(
                (job for job in self._queue if job.repo_full_name == repo_full_name),
                None
            )
            
            if not job:
                return
            
            job.completed_at = datetime.utcnow()
            self._processing.discard(repo_full_name)
            
            if success:
                job.status = IndexingStatus.COMPLETED
                logger.info(f"Indexing completed for {repo_full_name}")
            else:
                job.error_message = error_message
                job.retry_count += 1
                
                if job.retry_count >= job.max_retries:
                    job.status = IndexingStatus.FAILED
                    logger.error(f"Indexing failed permanently for {repo_full_name}: {error_message}")
                else:
                    job.status = IndexingStatus.PENDING  # Will be retried
                    job.started_at = None
                    self._processing.discard(repo_full_name)
                    logger.warning(f"Indexing failed for {repo_full_name}, will retry ({job.retry_count}/{job.max_retries}): {error_message}")
            
            self._save_queue()
    
    async def get_status(self, repo_full_name: Optional[str] = None) -> Dict:
        """Get indexing status for a specific repo or all repos."""
        async with self._lock:
            if repo_full_name:
                job = next(
                    (job for job in self._queue if job.repo_full_name == repo_full_name),
                    None
                )
                return job.to_dict() if job else None
            
            return {
                "total_jobs": len(self._queue),
                "pending": len([j for j in self._queue if j.status == IndexingStatus.PENDING]),
                "in_progress": len([j for j in self._queue if j.status == IndexingStatus.IN_PROGRESS]),
                "completed": len([j for j in self._queue if j.status == IndexingStatus.COMPLETED]),
                "failed": len([j for j in self._queue if j.status == IndexingStatus.FAILED]),
                "processing": list(self._processing),
                "jobs": [job.to_dict() for job in self._queue]
            }
    
    async def cancel_job(self, repo_full_name: str) -> bool:
        """Cancel a pending job."""
        async with self._lock:
            job = next(
                (job for job in self._queue 
                 if job.repo_full_name == repo_full_name 
                 and job.status in [IndexingStatus.PENDING, IndexingStatus.RETRYING]),
                None
            )
            
            if job:
                job.status = IndexingStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                self._save_queue()
                logger.info(f"Cancelled indexing job for {repo_full_name}")
                return True
            
            return False
    
    async def cleanup_old_jobs(self, days: int = 7):
        """Clean up completed/failed jobs older than specified days."""
        async with self._lock:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            original_count = len(self._queue)
            
            self._queue = [
                job for job in self._queue
                if not (
                    job.status in [IndexingStatus.COMPLETED, IndexingStatus.FAILED, IndexingStatus.CANCELLED]
                    and job.completed_at
                    and job.completed_at < cutoff_date
                )
            ]
            
            cleaned_count = original_count - len(self._queue)
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old indexing jobs")
                self._save_queue()

class IndexingService:
    """Main indexing service that processes the queue."""
    
    def __init__(self, max_concurrent_jobs: int = 3):
        self.queue = IndexingQueue()
        self.max_concurrent_jobs = max_concurrent_jobs
        self._workers: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "started_at": datetime.utcnow()
        }
    
    async def start(self):
        """Start the indexing service workers."""
        logger.info(f"Starting indexing service with {self.max_concurrent_jobs} workers")
        
        # Start worker tasks
        for i in range(self.max_concurrent_jobs):
            worker = asyncio.create_task(self._worker(f"worker-{i+1}"))
            self._workers.append(worker)
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_worker())
        self._workers.append(cleanup_task)
        
        logger.info("Indexing service started successfully")
    
    async def stop(self):
        """Stop the indexing service."""
        logger.info("Stopping indexing service...")
        self._shutdown_event.set()
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        logger.info("Indexing service stopped")
    
    async def _worker(self, worker_name: str):
        """Worker process that handles indexing jobs."""
        logger.info(f"Started indexing worker: {worker_name}")
        
        while not self._shutdown_event.is_set():
            try:
                # Get next job
                job = await self.queue.get_next_job()
                
                if not job:
                    # No jobs available, wait a bit
                    await asyncio.sleep(5)
                    continue
                
                logger.info(f"{worker_name}: Processing {job.repo_full_name}")
                
                # Process the job
                success, error_message = await self._process_indexing_job(job)
                
                # Update job status
                await self.queue.complete_job(job.repo_full_name, success, error_message)
                
                # Update stats
                self._stats["total_processed"] += 1
                if success:
                    self._stats["successful"] += 1
                else:
                    self._stats["failed"] += 1
                
                # Small delay between jobs
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.exception(f"Error in indexing worker {worker_name}")
                await asyncio.sleep(10)  # Wait before retrying
        
        logger.info(f"Indexing worker {worker_name} stopped")
    
    async def _process_indexing_job(self, job: IndexingJob) -> tuple[bool, Optional[str]]:
        """Process a single indexing job."""
        try:
            logger.info(f"Starting indexing for {job.repo_full_name}")
            
            # Check quota before starting
            if not await quota_manager.check_quota(job.repo_full_name):
                return False, "API quota exceeded"
            
            # Initialize the knowledge base (this will do the indexing)
            rag_result = await get_or_init_repo_knowledge_base(
                repo_full_name=job.repo_full_name,
                installation_id=job.installation_id,
                include_current_content=False,  # Don't include current content for background indexing
                force_refresh=job.force_refresh
            )
            
            if not rag_result:
                return False, "Failed to initialize RAG system"
            
            if isinstance(rag_result, dict) and rag_result.get("error"):
                error_type = rag_result.get("error_type", "Unknown")
                error_message = rag_result.get("error_message", "Unknown error")
                return False, f"{error_type}: {error_message}"
            
            # Reset error count on successful indexing
            reset_error_count(job.repo_full_name)
            
            logger.info(f"Successfully indexed {job.repo_full_name}")
            return True, None
            
        except Exception as e:
            logger.exception(f"Error indexing {job.repo_full_name}")
            return False, str(e)
    
    async def _cleanup_worker(self):
        """Worker that periodically cleans up old jobs."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up old jobs every 24 hours
                await asyncio.sleep(24 * 60 * 60)  # 24 hours
                await self.queue.cleanup_old_jobs(days=7)
            except Exception as e:
                logger.exception("Error in cleanup worker")
                await asyncio.sleep(60 * 60)  # Retry in 1 hour
    
    async def add_repository(
        self, 
        repo_full_name: str, 
        installation_id: int, 
        priority: int = 1,
        force_refresh: bool = False
    ) -> bool:
        """Add a repository for indexing."""
        return await self.queue.add_job(repo_full_name, installation_id, priority, force_refresh)
    
    async def get_status(self, repo_full_name: Optional[str] = None) -> Dict:
        """Get indexing status."""
        queue_status = await self.queue.get_status(repo_full_name)
        
        if repo_full_name:
            return queue_status
        
        # Add service stats
        queue_status["service_stats"] = self._stats.copy()
        queue_status["service_stats"]["uptime_seconds"] = (
            datetime.utcnow() - self._stats["started_at"]
        ).total_seconds()
        
        return queue_status
    
    async def cancel_indexing(self, repo_full_name: str) -> bool:
        """Cancel indexing for a repository."""
        return await self.queue.cancel_job(repo_full_name)

# Global indexing service instance
indexing_service = IndexingService(max_concurrent_jobs=3) 