from fastapi.testclient import TestClient
from app.main import app
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.core.github_utils import (
    fetch_repository_files,
    fetch_repository_issues,
    fetch_repository_pull_requests,
    fetch_repository_metadata,
    fetch_all_repository_content,
    _should_exclude_file,
    _is_text_file,
    _get_file_extension
)
from app.services.rag_service import get_or_init_repo_knowledge_base
import base64
from app.models.github import InstallationPayload

client = TestClient(app)

def test_webhook_missing_signature():
    response = client.post("/webhook", json={})
    assert response.status_code == 401
    assert "Missing X-Hub-Signature-256 header" in response.text

# More tests can be added for valid/invalid payloads and signature logic 

class TestRepositoryContentFetching:
    
    def test_should_exclude_file(self):
        """Test file exclusion logic."""
        assert _should_exclude_file("node_modules/package.json") == True
        assert _should_exclude_file("src/components/Button.tsx") == False
        assert _should_exclude_file(".git/config") == True
        assert _should_exclude_file("dist/main.js") == True
        assert _should_exclude_file("README.md") == False
    
    def test_is_text_file(self):
        """Test text file detection."""
        assert _is_text_file(b"Hello world") == True
        assert _is_text_file("# Header\nContent".encode('utf-8')) == True
        assert _is_text_file(b"\x00\x01\x02\x03") == False
    
    def test_get_file_extension(self):
        """Test file extension extraction."""
        assert _get_file_extension("test.py") == ".py"
        assert _get_file_extension("README.MD") == ".md"
        assert _get_file_extension("Dockerfile") == ""
        assert _get_file_extension("path/to/file.tsx") == ".tsx"

    @pytest.mark.asyncio
    async def test_fetch_repository_files_success(self):
        """Test successful repository file fetching."""
        # Mock GitHub client and repository
        mock_client = Mock()
        mock_repo = Mock()
        mock_client.get_repo.return_value = mock_repo
        
        # Mock file content
        mock_file = Mock()
        mock_file.name = "test.py"
        mock_file.path = "test.py"
        mock_file.type = "file"
        mock_file.size = 1000
        mock_file.sha = "abc123"
        mock_file.html_url = "https://github.com/owner/repo/blob/main/test.py"
        mock_file.content = base64.b64encode(b"print('Hello World')").decode('utf-8')
        
        mock_repo.get_contents.return_value = [mock_file]
        
        # Test the function
        result = await fetch_repository_files(
            client=mock_client,
            repo_full_name="owner/repo",
            max_files=10
        )
        
        # Assertions
        assert len(result) == 1
        assert result[0]["content"] == "print('Hello World')"
        assert result[0]["metadata"]["type"] == "repository_file"
        assert result[0]["metadata"]["file_path"] == "test.py"
        assert result[0]["metadata"]["file_extension"] == ".py"

    @pytest.mark.asyncio
    async def test_fetch_repository_issues_success(self):
        """Test successful repository issues fetching."""
        # Mock GitHub client and repository
        mock_client = Mock()
        mock_repo = Mock()
        mock_client.get_repo.return_value = mock_repo
        
        # Mock issue
        mock_issue = Mock()
        mock_issue.number = 123
        mock_issue.title = "Test Issue"
        mock_issue.body = "This is a test issue"
        mock_issue.state = "open"
        mock_issue.pull_request = None  # Not a PR
        mock_issue.comments = 0
        mock_issue.created_at.isoformat.return_value = "2023-01-01T00:00:00"
        mock_issue.updated_at.isoformat.return_value = "2023-01-01T00:00:00"
        mock_issue.html_url = "https://github.com/owner/repo/issues/123"
        mock_issue.labels = []
        
        mock_repo.get_issues.return_value = [mock_issue]
        
        # Test the function
        result = await fetch_repository_issues(
            client=mock_client,
            repo_full_name="owner/repo",
            max_issues=10
        )
        
        # Assertions
        assert len(result) == 1
        assert "Issue #123: Test Issue" in result[0]["content"]
        assert result[0]["metadata"]["type"] == "issue"
        assert result[0]["metadata"]["issue_number"] == 123

    @pytest.mark.asyncio
    async def test_fetch_repository_metadata_success(self):
        """Test successful repository metadata fetching."""
        # Mock GitHub client and repository
        mock_client = Mock()
        mock_repo = Mock()
        mock_client.get_repo.return_value = mock_repo
        
        # Mock repository attributes
        mock_repo.description = "Test repository"
        mock_repo.topics = ["python", "github"]
        mock_repo.language = "Python"
        mock_repo.stargazers_count = 100
        mock_repo.forks_count = 50
        mock_repo.created_at.isoformat.return_value = "2023-01-01T00:00:00"
        mock_repo.updated_at.isoformat.return_value = "2023-01-01T00:00:00"
        
        # Mock README
        mock_readme = Mock()
        mock_readme.path = "README.md"
        mock_readme.sha = "def456"
        mock_readme.content = base64.b64encode(b"# Test Repository\nThis is a test.").decode('utf-8')
        mock_repo.get_readme.return_value = mock_readme
        
        # Test the function
        result = await fetch_repository_metadata(
            client=mock_client,
            repo_full_name="owner/repo"
        )
        
        # Assertions
        assert len(result) == 2  # metadata + README
        
        # Check metadata document
        metadata_doc = next(doc for doc in result if doc["metadata"]["type"] == "repository_metadata")
        assert "Description: Test repository" in metadata_doc["content"]
        assert "Topics: python, github" in metadata_doc["content"]
        
        # Check README document
        readme_doc = next(doc for doc in result if doc["metadata"]["type"] == "readme")
        assert "# Test Repository" in readme_doc["content"]

    @pytest.mark.asyncio
    async def test_fetch_all_repository_content_success(self):
        """Test comprehensive repository content fetching."""
        mock_client = Mock()
        
        # Mock all individual fetch functions
        with patch('app.core.github_utils.fetch_repository_metadata') as mock_metadata, \
             patch('app.core.github_utils.fetch_repository_files') as mock_files, \
             patch('app.core.github_utils.fetch_repository_issues') as mock_issues, \
             patch('app.core.github_utils.fetch_repository_pull_requests') as mock_prs:
            
            # Set return values
            mock_metadata.return_value = [{"content": "metadata", "metadata": {"type": "repository_metadata"}}]
            mock_files.return_value = [{"content": "file", "metadata": {"type": "repository_file"}}]
            mock_issues.return_value = [{"content": "issue", "metadata": {"type": "issue"}}]
            mock_prs.return_value = [{"content": "pr", "metadata": {"type": "pull_request"}}]
            
            # Test the function
            result = await fetch_all_repository_content(
                client=mock_client,
                repo_full_name="owner/repo"
            )
            
            # Assertions
            assert len(result) == 4
            content_types = [doc["metadata"]["type"] for doc in result]
            assert "repository_metadata" in content_types
            assert "repository_file" in content_types
            assert "issue" in content_types
            assert "pull_request" in content_types

    @pytest.mark.asyncio
    async def test_get_or_init_repo_knowledge_base_new_repo(self):
        """Test RAG knowledge base initialization for new repository."""
        repo_full_name = "test/repo"
        installation_id = 12345
        
        with patch('app.services.rag_service.get_github_app_installation_client') as mock_client_func, \
             patch('app.services.rag_service.fetch_all_repository_content') as mock_fetch, \
             patch('app.services.rag_service.initialize_rag_system') as mock_init_rag:
            
            # Mock successful client creation
            mock_client = Mock()
            mock_client_func.return_value = mock_client
            
            # Mock successful content fetching
            mock_fetch.return_value = [
                {"content": "test content", "metadata": {"type": "repository_file"}}
            ]
            
            # Mock successful RAG initialization
            mock_rag_system = {"qa_chain": Mock(), "memory": Mock()}
            mock_init_rag.return_value = mock_rag_system
            
            # Test the function
            result = await get_or_init_repo_knowledge_base(
                repo_full_name=repo_full_name,
                installation_id=installation_id
            )
            
            # Assertions
            assert result == mock_rag_system
            mock_fetch.assert_called_once()
            mock_init_rag.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_or_init_repo_knowledge_base_existing_repo(self):
        """Test RAG knowledge base retrieval for existing repository."""
        repo_full_name = "test/repo"
        installation_id = 12345
        
        # Import the module to access the global variable
        from app.services.rag_service import repo_knowledge_base
        
        # Set up existing knowledge base
        existing_rag = {"qa_chain": Mock(), "memory": Mock()}
        repo_knowledge_base[repo_full_name] = existing_rag
        
        try:
            # Test the function
            result = await get_or_init_repo_knowledge_base(
                repo_full_name=repo_full_name,
                installation_id=installation_id
            )
            
            # Assertions
            assert result == existing_rag
            
        finally:
            # Clean up
            if repo_full_name in repo_knowledge_base:
                del repo_knowledge_base[repo_full_name]

    @pytest.mark.asyncio
    async def test_fetch_repository_files_handles_binary_files(self):
        """Test that binary files are properly excluded."""
        mock_client = Mock()
        mock_repo = Mock()
        mock_client.get_repo.return_value = mock_repo
        
        # Mock binary file
        mock_file = Mock()
        mock_file.name = "image.png"
        mock_file.path = "image.png"
        mock_file.type = "file"
        mock_file.size = 1000
        mock_file.content = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00").decode('utf-8')
        
        mock_repo.get_contents.return_value = [mock_file]
        
        # Test the function
        result = await fetch_repository_files(
            client=mock_client,
            repo_full_name="owner/repo"
        )
        
        # Binary files should be excluded
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_fetch_repository_files_handles_large_files(self):
        """Test that large files are properly excluded."""
        mock_client = Mock()
        mock_repo = Mock()
        mock_client.get_repo.return_value = mock_repo
        
        # Mock large file
        mock_file = Mock()
        mock_file.name = "large.py"
        mock_file.path = "large.py"
        mock_file.type = "file"
        mock_file.size = 2 * 1024 * 1024  # 2MB
        
        mock_repo.get_contents.return_value = [mock_file]
        
        # Test the function with 1MB limit
        result = await fetch_repository_files(
            client=mock_client,
            repo_full_name="owner/repo",
            max_file_size=1024 * 1024
        )
        
        # Large files should be excluded
        assert len(result) == 0 

def test_installation_deleted_incomplete_data():
    """Test handling of installation deleted event with incomplete data."""
    payload = {
        "action": "deleted",
        "installation": {
            "id": 123456
        }
        # Note: repositories key is missing
    }
    
    response = client.post("/webhook", json=payload, headers={
        "X-GitHub-Event": "installation",
        "Content-Type": "application/json"
    })
    
    # Should still return success even with incomplete data
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_pr_opened_webhook_success():
    """Test successful PR opened webhook handling."""
    
    # Mock PR payload
    pr_payload = {
        "action": "opened",
        "number": 42,
        "pull_request": {
            "id": 123456789,
            "number": 42,
            "title": "Add new feature",
            "body": "This PR adds a new feature to the codebase.",
            "user": {
                "login": "contributor",
                "type": "User"
            },
            "state": "open",
            "merged": False,
            "head": {"ref": "feature-branch", "sha": "abc123"},
            "base": {"ref": "main", "sha": "def456"},
            "draft": False,
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        },
        "repository": {
            "id": 987654321,
            "name": "test-repo",
            "full_name": "owner/test-repo",
            "owner": {"login": "owner"},
            "private": False
        },
        "installation": {
            "id": 12345678
        },
        "sender": {
            "login": "contributor",
            "type": "User"
        }
    }
    
    # Mock all the dependent functions
    with patch('app.services.rag_service.handle_pr_opened_event') as mock_handler:
        mock_handler.return_value = None  # Async function returning None
        
        response = client.post("/webhook", json=pr_payload, headers={
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json"
        })
        
        # Assertions
        assert response.status_code == 200
        assert "Successfully processed pull_request event" in response.json()["detail"]
        mock_handler.assert_called_once()

@pytest.mark.asyncio 
async def test_pr_opened_webhook_bot_user():
    """Test PR opened webhook handling when sender is a bot."""
    
    # Mock PR payload with bot sender
    pr_payload = {
        "action": "opened",
        "number": 42,
        "pull_request": {
            "id": 123456789,
            "number": 42,
            "title": "Automated update",
            "body": "This is an automated PR.",
            "user": {
                "login": "dependabot[bot]",
                "type": "Bot"
            },
            "state": "open",
            "merged": False,
            "head": {"ref": "dependabot/npm_and_yarn/axios-1.0.0", "sha": "abc123"},
            "base": {"ref": "main", "sha": "def456"},
            "draft": False,
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        },
        "repository": {
            "id": 987654321,
            "name": "test-repo", 
            "full_name": "owner/test-repo",
            "owner": {"login": "owner"},
            "private": False
        },
        "installation": {
            "id": 12345678
        },
        "sender": {
            "login": "dependabot[bot]",
            "type": "Bot"
        }
    }
    
    response = client.post("/webhook", json=pr_payload, headers={
        "X-GitHub-Event": "pull_request",
        "Content-Type": "application/json"
    })
    
    # Assertions - should ignore bot PRs
    assert response.status_code == 200
    assert "Pull request from bot ignored" in response.json()["detail"]

@pytest.mark.asyncio
async def test_pr_updated_webhook_success():
    """Test successful PR updated (synchronize) webhook handling."""
    
    # Mock PR payload
    pr_payload = {
        "action": "synchronize",
        "number": 42,
        "before": "old_sha",
        "after": "new_sha",
        "pull_request": {
            "id": 123456789,
            "number": 42,
            "title": "Add new feature",
            "body": "This PR adds a new feature to the codebase.",
            "user": {
                "login": "contributor",
                "type": "User"
            },
            "state": "open",
            "merged": False,
            "head": {"ref": "feature-branch", "sha": "new_sha"},
            "base": {"ref": "main", "sha": "def456"},
            "draft": False,
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        },
        "repository": {
            "id": 987654321,
            "name": "test-repo",
            "full_name": "owner/test-repo",
            "owner": {"login": "owner"},
            "private": False
        },
        "installation": {
            "id": 12345678
        },
        "sender": {
            "login": "contributor",
            "type": "User"
        }
    }
    
    # Mock the handler function
    with patch('app.services.rag_service.handle_pr_updated_event') as mock_handler:
        mock_handler.return_value = None
        
        response = client.post("/webhook", json=pr_payload, headers={
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json"
        })
        
        # Assertions
        assert response.status_code == 200
        assert "Successfully processed pull_request event" in response.json()["detail"]
        mock_handler.assert_called_once()

@pytest.mark.asyncio
async def test_pr_closed_webhook_success():
    """Test successful PR closed webhook handling."""
    
    # Mock PR payload for merged PR
    pr_payload = {
        "action": "closed",
        "number": 42,
        "pull_request": {
            "id": 123456789,
            "number": 42,
            "title": "Add new feature",
            "body": "This PR adds a new feature to the codebase.",
            "user": {
                "login": "contributor",
                "type": "User"
            },
            "state": "closed",
            "merged": True,
            "merged_at": "2023-01-01T01:00:00Z",
            "head": {"ref": "feature-branch", "sha": "abc123"},
            "base": {"ref": "main", "sha": "def456"},
            "draft": False,
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T01:00:00Z",
            "closed_at": "2023-01-01T01:00:00Z"
        },
        "repository": {
            "id": 987654321,
            "name": "test-repo",
            "full_name": "owner/test-repo",
            "owner": {"login": "owner"},
            "private": False
        },
        "installation": {
            "id": 12345678
        },
        "sender": {
            "login": "contributor",
            "type": "User"
        }
    }
    
    # Mock the handler function
    with patch('app.services.rag_service.handle_pr_closed_event') as mock_handler:
        mock_handler.return_value = None
        
        response = client.post("/webhook", json=pr_payload, headers={
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json"
        })
        
        # Assertions
        assert response.status_code == 200
        assert "Successfully processed pull_request event" in response.json()["detail"]
        mock_handler.assert_called_once() 