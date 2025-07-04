"""
Standalone tests for repository content fetching functions.
These tests don't require environment variables and can run independently.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from unittest.mock import Mock, patch
import base64
from pathlib import Path

# Import the functions directly
from app.core.github_utils import (
    _should_exclude_file,
    _is_text_file,
    _get_file_extension,
    fetch_repository_files,
    fetch_repository_issues,
    fetch_repository_metadata,
    fetch_all_repository_content
)

class TestRepositoryContentFetching:
    
    def test_should_exclude_file(self):
        """Test file exclusion logic."""
        assert _should_exclude_file("node_modules/package.json") == True
        assert _should_exclude_file("src/components/Button.tsx") == False
        assert _should_exclude_file(".git/config") == True
        assert _should_exclude_file("dist/main.js") == True
        assert _should_exclude_file("README.md") == False
        assert _should_exclude_file("__pycache__/module.pyc") == True
        assert _should_exclude_file("src/main.py") == False
    
    def test_is_text_file(self):
        """Test text file detection."""
        assert _is_text_file(b"Hello world") == True
        assert _is_text_file("# Header\nContent".encode('utf-8')) == True
        assert _is_text_file(b"\x00\x01\x02\x03") == False
        assert _is_text_file("console.log('hello');".encode('utf-8')) == True
        assert _is_text_file(b"\x89PNG\r\n\x1a\n") == False
    
    def test_get_file_extension(self):
        """Test file extension extraction."""
        assert _get_file_extension("test.py") == ".py"
        assert _get_file_extension("README.MD") == ".md"
        assert _get_file_extension("Dockerfile") == ""
        assert _get_file_extension("path/to/file.tsx") == ".tsx"
        assert _get_file_extension("config.yaml") == ".yaml"
        assert _get_file_extension("script.sh") == ".sh"

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
    async def test_fetch_repository_files_excludes_binary(self):
        """Test that binary files are properly excluded."""
        mock_client = Mock()
        mock_repo = Mock()
        mock_client.get_repo.return_value = mock_repo
        
        # Mock binary file (PNG header)
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
    async def test_fetch_repository_files_excludes_large_files(self):
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

    @pytest.mark.asyncio
    async def test_fetch_repository_files_handles_directories(self):
        """Test recursive directory processing."""
        mock_client = Mock()
        mock_repo = Mock()
        mock_client.get_repo.return_value = mock_repo
        
        # Mock directory structure
        mock_dir = Mock()
        mock_dir.name = "src"
        mock_dir.path = "src"
        mock_dir.type = "dir"
        
        mock_file = Mock()
        mock_file.name = "main.py"
        mock_file.path = "src/main.py"
        mock_file.type = "file"
        mock_file.size = 500
        mock_file.sha = "def456"
        mock_file.html_url = "https://github.com/owner/repo/blob/main/src/main.py"
        mock_file.content = base64.b64encode(b"def main(): pass").decode('utf-8')
        
        # Mock get_contents calls
        def mock_get_contents(path):
            if path == "":
                return [mock_dir]
            elif path == "src":
                return [mock_file]
            return []
        
        mock_repo.get_contents.side_effect = mock_get_contents
        
        # Test the function
        result = await fetch_repository_files(
            client=mock_client,
            repo_full_name="owner/repo",
            max_files=10
        )
        
        # Should find the file in the subdirectory
        assert len(result) == 1
        assert result[0]["content"] == "def main(): pass"
        assert result[0]["metadata"]["file_path"] == "src/main.py"

    @pytest.mark.asyncio
    async def test_fetch_repository_issues_success(self):
        """Test successful repository issues fetching."""
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
        assert "This is a test issue" in result[0]["content"]
        assert result[0]["metadata"]["type"] == "issue"
        assert result[0]["metadata"]["issue_number"] == 123

    @pytest.mark.asyncio
    async def test_fetch_repository_metadata_success(self):
        """Test successful repository metadata fetching."""
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
    async def test_fetch_all_repository_content_integration(self):
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
            
            # Verify all functions were called
            mock_metadata.assert_called_once()
            mock_files.assert_called_once()
            mock_issues.assert_called_once()
            mock_prs.assert_called_once()

if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"]) 