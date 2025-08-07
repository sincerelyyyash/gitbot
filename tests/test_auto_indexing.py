import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.indexing_service import IndexingService, IndexingQueue, IndexingJob, IndexingStatus
from app.models.github import InstallationPayload, InstallationRepositoriesPayload, Repository, User, Installation


class TestIndexingJob:
    """Test the IndexingJob dataclass."""
    
    def test_job_creation(self):
        job = IndexingJob(
            repo_full_name="owner/repo",
            installation_id=12345,
            status=IndexingStatus.PENDING,
            created_at=datetime.utcnow()
        )
        
        assert job.repo_full_name == "owner/repo"
        assert job.installation_id == 12345
        assert job.status == IndexingStatus.PENDING
        assert job.retry_count == 0
        assert job.max_retries == 3
        assert job.priority == 1
        assert job.force_refresh is False
    
    def test_job_serialization(self):
        job = IndexingJob(
            repo_full_name="owner/repo",
            installation_id=12345,
            status=IndexingStatus.PENDING,
            created_at=datetime.utcnow()
        )
        
        # Test to_dict
        job_dict = job.to_dict()
        assert job_dict["repo_full_name"] == "owner/repo"
        assert job_dict["status"] == "pending"
        assert isinstance(job_dict["created_at"], str)
        
        # Test from_dict
        restored_job = IndexingJob.from_dict(job_dict)
        assert restored_job.repo_full_name == job.repo_full_name
        assert restored_job.status == job.status
        assert restored_job.created_at == job.created_at


class TestIndexingQueue:
    """Test the IndexingQueue class."""
    
    @pytest.fixture
    def temp_queue_file(self):
        """Create a temporary file for queue persistence."""
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def queue(self, temp_queue_file):
        """Create an IndexingQueue with temporary storage."""
        return IndexingQueue(persist_file=temp_queue_file)
    
    @pytest.mark.asyncio
    async def test_add_job(self, queue):
        """Test adding jobs to the queue."""
        # Add first job
        added = await queue.add_job("owner/repo1", 12345, priority=1)
        assert added is True
        
        # Try to add duplicate job
        added = await queue.add_job("owner/repo1", 12345, priority=1)
        assert added is False  # Should not add duplicate
        
        # Add different repository
        added = await queue.add_job("owner/repo2", 12345, priority=2)
        assert added is True
    
    @pytest.mark.asyncio
    async def test_priority_sorting(self, queue):
        """Test that jobs are sorted by priority."""
        # Add jobs with different priorities
        await queue.add_job("owner/repo1", 12345, priority=3)
        await queue.add_job("owner/repo2", 12345, priority=1)  # Higher priority
        await queue.add_job("owner/repo3", 12345, priority=2)
        
        # Get jobs and verify order
        job1 = await queue.get_next_job()
        job2 = await queue.get_next_job()
        job3 = await queue.get_next_job()
        
        assert job1.repo_full_name == "owner/repo2"  # Priority 1
        assert job2.repo_full_name == "owner/repo3"  # Priority 2
        assert job3.repo_full_name == "owner/repo1"  # Priority 3
    
    @pytest.mark.asyncio
    async def test_job_completion(self, queue):
        """Test marking jobs as completed or failed."""
        await queue.add_job("owner/repo", 12345)
        
        # Complete successfully
        await queue.complete_job("owner/repo", success=True)
        
        status = await queue.get_status("owner/repo")
        assert status["status"] == "completed"
        assert status["completed_at"] is not None
    
    @pytest.mark.asyncio
    async def test_job_retry_logic(self, queue):
        """Test retry logic for failed jobs."""
        await queue.add_job("owner/repo", 12345)
        
        # Fail job (should retry)
        await queue.complete_job("owner/repo", success=False, error_message="Test error")
        
        status = await queue.get_status("owner/repo")
        assert status["status"] == "pending"  # Should be retrying
        assert status["retry_count"] == 1
        assert status["error_message"] == "Test error"
        
        # Fail again
        job = await queue.get_next_job()
        await queue.complete_job("owner/repo", success=False, error_message="Test error")
        
        # Fail third time (should reach max retries)
        job = await queue.get_next_job()
        await queue.complete_job("owner/repo", success=False, error_message="Test error")
        
        status = await queue.get_status("owner/repo")
        assert status["status"] == "failed"
        assert status["retry_count"] == 3
    
    @pytest.mark.asyncio
    async def test_queue_persistence(self, temp_queue_file):
        """Test that queue state is persisted to disk."""
        # Create queue and add jobs
        queue1 = IndexingQueue(persist_file=temp_queue_file)
        await queue1.add_job("owner/repo1", 12345, priority=1)
        await queue1.add_job("owner/repo2", 12345, priority=2)
        
        # Create new queue from same file
        queue2 = IndexingQueue(persist_file=temp_queue_file)
        
        # Should load existing jobs
        status = await queue2.get_status()
        assert status["total_jobs"] == 2
        assert status["pending"] == 2
        
        # Jobs should be in priority order
        job = await queue2.get_next_job()
        assert job.repo_full_name == "owner/repo1"
    
    @pytest.mark.asyncio
    async def test_cancel_job(self, queue):
        """Test cancelling pending jobs."""
        await queue.add_job("owner/repo", 12345)
        
        # Cancel the job
        cancelled = await queue.cancel_job("owner/repo")
        assert cancelled is True
        
        status = await queue.get_status("owner/repo")
        assert status["status"] == "cancelled"
        
        # Try to cancel non-existent job
        cancelled = await queue.cancel_job("owner/nonexistent")
        assert cancelled is False
    
    @pytest.mark.asyncio
    async def test_cleanup_old_jobs(self, queue):
        """Test cleanup of old completed jobs."""
        # Add and complete a job
        await queue.add_job("owner/repo", 12345)
        await queue.complete_job("owner/repo", success=True)
        
        # Manually set completed_at to old date
        async with queue._lock:
            job = queue._queue[0]
            job.completed_at = datetime.utcnow() - timedelta(days=8)
            queue._save_queue()
        
        # Run cleanup
        await queue.cleanup_old_jobs(days=7)
        
        # Job should be removed
        status = await queue.get_status()
        assert status["total_jobs"] == 0


class TestIndexingService:
    """Test the IndexingService class."""
    
    @pytest.fixture
    def temp_queue_file(self):
        """Create a temporary file for queue persistence."""
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def service(self, temp_queue_file):
        """Create an IndexingService with temporary storage."""
        with patch('app.services.indexing_service.IndexingQueue') as mock_queue_class:
            mock_queue = AsyncMock()
            mock_queue_class.return_value = mock_queue
            service = IndexingService(max_concurrent_jobs=2)
            service.queue = mock_queue
            return service, mock_queue
    
    @pytest.mark.asyncio
    async def test_add_repository(self, service):
        """Test adding repository for indexing."""
        service, mock_queue = service
        mock_queue.add_job.return_value = True
        
        result = await service.add_repository("owner/repo", 12345, priority=1)
        
        assert result is True
        mock_queue.add_job.assert_called_once_with("owner/repo", 12345, 1, False)
    
    @pytest.mark.asyncio
    async def test_get_status(self, service):
        """Test getting service status."""
        service, mock_queue = service
        mock_queue.get_status.return_value = {"total_jobs": 5}
        
        status = await service.get_status()
        
        assert "service_stats" in status
        assert "total_jobs" in status
        assert status["total_jobs"] == 5
    
    @pytest.mark.asyncio
    async def test_cancel_indexing(self, service):
        """Test cancelling repository indexing."""
        service, mock_queue = service
        mock_queue.cancel_job.return_value = True
        
        result = await service.cancel_indexing("owner/repo")
        
        assert result is True
        mock_queue.cancel_job.assert_called_once_with("owner/repo")


class TestWebhookIntegration:
    """Test webhook integration with indexing service."""
    
    def test_installation_payload_creation(self):
        """Test creating InstallationPayload from webhook data."""
        payload_data = {
            "action": "created",
            "installation": {
                "id": 12345,
                "node_id": "MDIzOkludGVncmF0aW9uSW5zdGFsbGF0aW9uMTIzNDU="
            },
            "repositories": [
                {
                    "id": 1,
                    "name": "repo1",
                    "full_name": "owner/repo1",
                    "private": False,
                    "owner": {
                        "login": "owner",
                        "id": 123,
                        "type": "User",
                        "site_admin": False
                    },
                    "default_branch": "main"
                }
            ],
            "sender": {
                "login": "owner",
                "id": 123,
                "type": "User",
                "site_admin": False
            }
        }
        
        payload = InstallationPayload(**payload_data)
        
        assert payload.action == "created"
        assert payload.installation.id == 12345
        assert len(payload.repositories) == 1
        assert payload.repositories[0].full_name == "owner/repo1"
        assert payload.sender.login == "owner"
    
    def test_installation_repositories_payload_creation(self):
        """Test creating InstallationRepositoriesPayload from webhook data."""
        payload_data = {
            "action": "added",
            "installation": {
                "id": 12345
            },
            "repositories_added": [
                {
                    "id": 2,
                    "name": "repo2",
                    "full_name": "owner/repo2",
                    "private": False,
                    "owner": {
                        "login": "owner",
                        "id": 123,
                        "type": "User",
                        "site_admin": False
                    },
                    "default_branch": "main"
                }
            ],
            "repositories_removed": [],
            "repository_selection": "selected",
            "sender": {
                "login": "owner",
                "id": 123,
                "type": "User",
                "site_admin": False
            }
        }
        
        payload = InstallationRepositoriesPayload(**payload_data)
        
        assert payload.action == "added"
        assert payload.installation.id == 12345
        assert len(payload.repositories_added) == 1
        assert payload.repositories_added[0].full_name == "owner/repo2"
        assert payload.repository_selection == "selected"
    
    @pytest.mark.asyncio
    async def test_installation_event_handling(self):
        """Test handling installation webhook events."""
        from app.api.webhook import handle_installation_event
        
        # Mock the indexing service
        with patch('app.api.webhook.indexing_service') as mock_service:
            mock_service.add_repository = AsyncMock(return_value=True)
            
            # Create installation payload
            payload = InstallationPayload(
                action="created",
                installation=Installation(id=12345),
                repositories=[
                    Repository(
                        id=1,
                        name="repo1",
                        full_name="owner/repo1",
                        private=False,
                        owner=User(login="owner", id=123, type="User"),
                        default_branch="main"
                    )
                ],
                sender=User(login="owner", id=123, type="User")
            )
            
            # Handle the event
            await handle_installation_event(payload)
            
            # Verify repository was queued for indexing
            mock_service.add_repository.assert_called_once_with(
                repo_full_name="owner/repo1",
                installation_id=12345,
                priority=0,  # Highest priority for new installations
                force_refresh=False
            )
    
    @pytest.mark.asyncio
    async def test_installation_repositories_event_handling(self):
        """Test handling installation_repositories webhook events."""
        from app.api.webhook import handle_installation_repositories_event
        
        # Mock the indexing service
        with patch('app.api.webhook.indexing_service') as mock_service:
            mock_service.add_repository = AsyncMock(return_value=True)
            mock_service.cancel_indexing = AsyncMock(return_value=True)
            
            # Test adding repositories
            payload = InstallationRepositoriesPayload(
                action="added",
                installation=Installation(id=12345),
                repositories_added=[
                    Repository(
                        id=2,
                        name="repo2",
                        full_name="owner/repo2",
                        private=False,
                        owner=User(login="owner", id=123, type="User"),
                        default_branch="main"
                    )
                ],
                repositories_removed=None,
                repository_selection="selected",
                sender=User(login="owner", id=123, type="User")
            )
            
            await handle_installation_repositories_event(payload)
            
            # Verify repository was queued for indexing
            mock_service.add_repository.assert_called_once_with(
                repo_full_name="owner/repo2",
                installation_id=12345,
                priority=1,  # High priority for newly added repos
                force_refresh=False
            )
            
            # Test removing repositories
            mock_service.reset_mock()
            payload.action = "removed"
            payload.repositories_added = None
            payload.repositories_removed = [
                Repository(
                    id=2,
                    name="repo2",
                    full_name="owner/repo2",
                    private=False,
                    owner=User(login="owner", id=123, type="User"),
                    default_branch="main"
                )
            ]
            
            await handle_installation_repositories_event(payload)
            
            # Verify indexing was cancelled
            mock_service.cancel_indexing.assert_called_once_with("owner/repo2")


class TestIndexingServiceIntegration:
    """Integration tests for the complete indexing system."""
    
    @pytest.mark.asyncio
    async def test_full_indexing_workflow(self):
        """Test complete workflow from webhook to indexing completion."""
        # Mock the RAG service
        with patch('app.services.rag_service.get_or_init_repo_knowledge_base') as mock_rag:
            with patch('app.services.indexing_service.quota_manager') as mock_quota:
                mock_quota.check_quota.return_value = True
                mock_rag.return_value = {"success": True}  # Successful indexing
                
                # Create temporary queue file
                fd, temp_file = tempfile.mkstemp(suffix='.json')
                os.close(fd)
                
                try:
                    # Create service with real queue
                    service = IndexingService(max_concurrent_jobs=1)
                    service.queue = IndexingQueue(persist_file=temp_file)
                    
                    # Add repository for indexing
                    await service.add_repository("owner/test-repo", 12345, priority=1)
                    
                    # Simulate processing one job manually
                    job = await service.queue.get_next_job()
                    assert job is not None
                    assert job.repo_full_name == "owner/test-repo"
                    assert job.status == IndexingStatus.IN_PROGRESS
                    
                    # Process the job
                    success, error_msg = await service._process_indexing_job(job)
                    assert success is True
                    assert error_msg is None
                    
                    # Complete the job
                    await service.queue.complete_job(job.repo_full_name, success, error_msg)
                    
                    # Verify job completion
                    status = await service.queue.get_status("owner/test-repo")
                    assert status["status"] == "completed"
                    assert status["retry_count"] == 0
                    
                finally:
                    try:
                        os.unlink(temp_file)
                    except FileNotFoundError:
                        pass
    
    @pytest.mark.asyncio 
    async def test_indexing_failure_and_retry(self):
        """Test indexing failure and retry mechanism."""
        # Mock the RAG service to fail
        with patch('app.services.rag_service.get_or_init_repo_knowledge_base') as mock_rag:
            with patch('app.services.indexing_service.quota_manager') as mock_quota:
                mock_quota.check_quota.return_value = True
                mock_rag.return_value = {
                    "error": True,
                    "error_type": "APIKeyRestricted",
                    "error_message": "IP address restriction"
                }
                
                # Create temporary queue file
                fd, temp_file = tempfile.mkstemp(suffix='.json')
                os.close(fd)
                
                try:
                    # Create service with real queue
                    service = IndexingService(max_concurrent_jobs=1)
                    service.queue = IndexingQueue(persist_file=temp_file)
                    
                    # Add repository for indexing
                    await service.add_repository("owner/test-repo", 12345, priority=1)
                    
                    # Process job - should fail
                    job = await service.queue.get_next_job()
                    success, error_msg = await service._process_indexing_job(job)
                    
                    assert success is False
                    assert "APIKeyRestricted" in error_msg
                    
                    # Complete with failure
                    await service.queue.complete_job(job.repo_full_name, success, error_msg)
                    
                    # Job should be retrying
                    status = await service.queue.get_status("owner/test-repo")
                    assert status["status"] == "pending"  # Ready for retry
                    assert status["retry_count"] == 1
                    assert "APIKeyRestricted" in status["error_message"]
                    
                finally:
                    try:
                        os.unlink(temp_file)
                    except FileNotFoundError:
                        pass


if __name__ == "__main__":
    pytest.main([__file__])