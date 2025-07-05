import logging
from typing import Optional, List, Dict, Any, Set
from github import Github, Auth, GithubIntegration
from github.GithubException import GithubException
from github.Repository import Repository
from github.ContentFile import ContentFile
import jwt
from app.config import settings
import time
import base64
from pathlib import Path
from app.core.github_rate_limiter import github_rate_limiter
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("github_utils")

GITHUB_APP_ID = settings.github_app_id
GITHUB_PRIVATE_KEY_PATH = settings.github_private_key

# File extensions to include for code analysis
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.r', '.m',
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.sql', '.html', '.css', '.scss',
    '.sass', '.less', '.vue', '.svelte', '.yaml', '.yml', '.json', '.xml',
    '.toml', '.ini', '.cfg', '.conf', '.env', '.dockerfile', '.makefile'
}

# Documentation file extensions
DOC_EXTENSIONS = {
    '.md', '.rst', '.txt', '.adoc', '.org', '.tex', '.rtf'
}

# Files to exclude from processing
EXCLUDED_PATTERNS = {
    'node_modules', '.git', '.github', 'dist', 'build', 'target', 'out',
    '__pycache__', '.pytest_cache', '.coverage', '.nyc_output', 'coverage',
    '.env', '.env.local', '.env.production', '.env.development',
    'package-lock.json', 'yarn.lock', 'poetry.lock', 'Pipfile.lock'
}

def generate_jwt_token(app_id: str, private_key: str) -> Optional[str]:
    """Generate a JWT token for GitHub App authentication using the private key string."""
    try:
        # Use the private key directly from settings
        now = int(time.time())
        payload = {
            "iat": now - 60,
            "exp": now + (10 * 60),
            "iss": app_id
        }
        encoded_jwt = jwt.encode(payload, private_key, algorithm="RS256")
        logger.info("Generated JWT for GitHub App authentication.")
        return encoded_jwt if isinstance(encoded_jwt, str) else encoded_jwt.decode("utf-8")
    except Exception as e:
        logger.exception(f"Failed to generate JWT token: {str(e)}")
        return None

@github_rate_limiter.with_rate_limit(category="core")
async def get_github_app_installation_client(
    app_id: str,
    private_key: str,
    installation_id: int
) -> Optional[Github]:
    """Get an authenticated GitHub client for an installation."""
    try:
        # Create GitHub integration directly with app auth
        git_integration = GithubIntegration(
            auth=Auth.AppAuth(
                app_id=app_id,
                private_key=private_key
            )
        )
        
        try:
            # Get installation access token
            access_token = git_integration.get_access_token(installation_id).token
            
            # Create and return GitHub client
            return Github(auth=Auth.Token(access_token))
        except Exception as e:
            logger.error(f"GitHub integration failed: {str(e)}")
            return None
    
    except Exception as e:
        logger.error(f"Failed to get GitHub client: {str(e)}")
        return None

@github_rate_limiter.with_rate_limit(category="core")
async def post_issue_comment(
    client: Github,
    repo_full_name: str,
    issue_number: int,
    comment: str
) -> bool:
    """Post a comment on a GitHub issue."""
    try:
        repo = client.get_repo(repo_full_name)
        issue = repo.get_issue(issue_number)
        issue.create_comment(comment)
        return True
    except Exception as e:
        logger.error(f"Failed to post comment: {e}")
        return False

def _should_exclude_file(file_path: str) -> bool:
    """Check if a file should be excluded from processing."""
    path_parts = set(Path(file_path).parts)
    return bool(path_parts.intersection(EXCLUDED_PATTERNS))

def _is_text_file(content: bytes) -> bool:
    """Check if content appears to be text (not binary)."""
    try:
        # First check for null bytes which are common in binary files
        if b'\x00' in content:
            return False
        
        # Try to decode as UTF-8
        text = content.decode('utf-8')
        
        # Check for common binary patterns in the decoded text
        # Control characters (except common whitespace)
        import string
        printable_chars = set(string.printable)
        
        # Allow for some non-printable characters but not too many
        non_printable_count = sum(1 for char in text if char not in printable_chars)
        non_printable_ratio = non_printable_count / len(text) if text else 0
        
        # If more than 30% of characters are non-printable, likely binary
        return non_printable_ratio < 0.3
        
    except UnicodeDecodeError:
        return False

def _get_file_extension(file_path: str) -> str:
    """Get the file extension in lowercase."""
    return Path(file_path).suffix.lower()

@github_rate_limiter.with_rate_limit(category="core")
async def fetch_repository_files(
    client: Github,
    repo_full_name: str,
    path: str = "",
    extensions: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Fetch repository files with rate limit handling."""
    try:
        repo = client.get_repo(repo_full_name)
        contents = []
        
        async def process_contents(path: str):
            items = repo.get_contents(path)
            for item in items:
                if item.type == "dir":
                    await process_contents(item.path)
                elif item.type == "file":
                    # Check extensions
                    if extensions and not any(
                        item.path.endswith(ext) for ext in extensions
                    ):
                        continue
                    
                    # Check exclude patterns
                    if exclude_patterns and any(
                        pattern in item.path for pattern in exclude_patterns
                    ):
                        continue
                    
                    try:
                        content = item.decoded_content
                        if _is_text_file(content):
                            # Try UTF-8 first
                            try:
                                decoded_content = content.decode('utf-8')
                            except UnicodeDecodeError:
                                # Fallback to latin-1 which can decode any byte sequence
                                decoded_content = content.decode('latin-1')
                            
                            contents.append({
                                "path": item.path,
                                "content": decoded_content,
                                "sha": item.sha,
                                "size": item.size,
                                "type": "file"
                            })
                    except Exception as e:
                        logger.warning(f"Skipping file {item.path} due to decoding error: {e}")
                        continue
        
        await process_contents(path)
        return contents
    
    except Exception as e:
        logger.error(f"Failed to fetch repository files: {e}")
        return []

async def fetch_repository_issues(
    client: Github,
    repo_full_name: str,
    state: str = "all",
    max_issues: int = 100,
    include_comments: bool = True
) -> List[Dict[str, Any]]:
    """
    Fetch repository issues and optionally their comments.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name (owner/repo)
        state: Issue state ("open", "closed", "all")
        max_issues: Maximum number of issues to fetch
        include_comments: Whether to include issue comments
        
    Returns:
        List of document dictionaries with issue content and metadata
    """
    documents = []
    
    try:
        repo = client.get_repo(repo_full_name)
        logger.info(f"Fetching issues from {repo_full_name} (state: {state})")
        
        issues = repo.get_issues(state=state, sort="updated", direction="desc")
        
        issue_count = 0
        for issue in issues:
            if issue_count >= max_issues:
                break
                
            # Skip pull requests (they appear in issues list)
            if issue.pull_request:
                continue
                
            issue_count += 1
            
            # Process issue body
            issue_body = issue.body or ""
            if issue_body.strip():
                documents.append({
                    "content": f"Issue #{issue.number}: {issue.title}\n\n{issue_body}",
                    "metadata": {
                        "type": "issue",
                        "issue_number": issue.number,
                        "title": issue.title,
                        "state": issue.state,
                        "created_at": issue.created_at.isoformat(),
                        "updated_at": issue.updated_at.isoformat(),
                        "url": issue.html_url,
                        "labels": [label.name for label in issue.labels],
                        "repository": repo_full_name
                    }
                })
            
            # Process issue comments if requested
            if include_comments and issue.comments > 0:
                try:
                    comments = issue.get_comments()
                    for comment in comments:
                        if comment.body and comment.body.strip():
                            documents.append({
                                "content": f"Comment on Issue #{issue.number}:\n\n{comment.body}",
                                "metadata": {
                                    "type": "issue_comment",
                                    "issue_number": issue.number,
                                    "comment_id": comment.id,
                                    "created_at": comment.created_at.isoformat(),
                                    "updated_at": comment.updated_at.isoformat(),
                                    "url": comment.html_url,
                                    "repository": repo_full_name
                                }
                            })
                except Exception as e:
                    logger.warning(f"Failed to fetch comments for issue #{issue.number}: {e}")
        
        logger.info(f"Fetched {len(documents)} issue documents from {repo_full_name}")
        return documents
        
    except GithubException as e:
        logger.error(f"GitHub API error fetching issues: {e}")
        return []
    except Exception as e:
        logger.exception(f"Failed to fetch issues from {repo_full_name}")
        return []

async def fetch_repository_pull_requests(
    client: Github,
    repo_full_name: str,
    state: str = "all",
    max_prs: int = 50,
    include_comments: bool = True,
    include_review_comments: bool = True
) -> List[Dict[str, Any]]:
    """
    Fetch repository pull requests and optionally their comments.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name (owner/repo)
        state: PR state ("open", "closed", "all")
        max_prs: Maximum number of PRs to fetch
        include_comments: Whether to include PR comments
        include_review_comments: Whether to include code review comments
        
    Returns:
        List of document dictionaries with PR content and metadata
    """
    documents = []
    
    try:
        repo = client.get_repo(repo_full_name)
        logger.info(f"Fetching pull requests from {repo_full_name} (state: {state})")
        
        pulls = repo.get_pulls(state=state, sort="updated", direction="desc")
        
        pr_count = 0
        for pr in pulls:
            if pr_count >= max_prs:
                break
                
            pr_count += 1
            
            # Process PR body
            pr_body = pr.body or ""
            if pr_body.strip():
                documents.append({
                    "content": f"Pull Request #{pr.number}: {pr.title}\n\n{pr_body}",
                    "metadata": {
                        "type": "pull_request",
                        "pr_number": pr.number,
                        "title": pr.title,
                        "state": pr.state,
                        "created_at": pr.created_at.isoformat(),
                        "updated_at": pr.updated_at.isoformat(),
                        "merged": pr.merged,
                        "url": pr.html_url,
                        "labels": [label.name for label in pr.labels],
                        "repository": repo_full_name
                    }
                })
            
            # Process PR comments if requested
            if include_comments:
                try:
                    comments = pr.get_issue_comments()
                    for comment in comments:
                        if comment.body and comment.body.strip():
                            documents.append({
                                "content": f"Comment on PR #{pr.number}:\n\n{comment.body}",
                                "metadata": {
                                    "type": "pr_comment",
                                    "pr_number": pr.number,
                                    "comment_id": comment.id,
                                    "created_at": comment.created_at.isoformat(),
                                    "updated_at": comment.updated_at.isoformat(),
                                    "url": comment.html_url,
                                    "repository": repo_full_name
                                }
                            })
                except Exception as e:
                    logger.warning(f"Failed to fetch comments for PR #{pr.number}: {e}")
            
            # Process review comments if requested
            if include_review_comments:
                try:
                    review_comments = pr.get_review_comments()
                    for comment in review_comments:
                        if comment.body and comment.body.strip():
                            documents.append({
                                "content": f"Review comment on PR #{pr.number}:\n\n{comment.body}",
                                "metadata": {
                                    "type": "pr_review_comment",
                                    "pr_number": pr.number,
                                    "comment_id": comment.id,
                                    "file_path": comment.path,
                                    "line": comment.line,
                                    "created_at": comment.created_at.isoformat(),
                                    "updated_at": comment.updated_at.isoformat(),
                                    "url": comment.html_url,
                                    "repository": repo_full_name
                                }
                            })
                except Exception as e:
                    logger.warning(f"Failed to fetch review comments for PR #{pr.number}: {e}")
        
        logger.info(f"Fetched {len(documents)} PR documents from {repo_full_name}")
        return documents
        
    except GithubException as e:
        logger.error(f"GitHub API error fetching pull requests: {e}")
        return []
    except Exception as e:
        logger.exception(f"Failed to fetch pull requests from {repo_full_name}")
        return []

async def fetch_repository_metadata(client: Github, repo_full_name: str) -> List[Dict[str, Any]]:
    """
    Fetch repository metadata including README, description, and topics.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name (owner/repo)
        
    Returns:
        List of document dictionaries with repository metadata
    """
    documents = []
    
    try:
        repo = client.get_repo(repo_full_name)
        logger.info(f"Fetching repository metadata from {repo_full_name}")
        
        # Repository description and topics
        repo_info = []
        if repo.description:
            repo_info.append(f"Description: {repo.description}")
        
        if repo.topics:
            repo_info.append(f"Topics: {', '.join(repo.topics)}")
            
        if repo.language:
            repo_info.append(f"Primary Language: {repo.language}")
            
        if repo_info:
            documents.append({
                "content": f"Repository Information for {repo_full_name}:\n\n" + "\n".join(repo_info),
                "metadata": {
                    "type": "repository_metadata",
                    "repository": repo_full_name,
                    "language": repo.language,
                    "topics": repo.topics,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "created_at": repo.created_at.isoformat(),
                    "updated_at": repo.updated_at.isoformat()
                }
            })
        
        # Try to fetch README content
        try:
            readme = repo.get_readme()
            readme_content = base64.b64decode(readme.content).decode('utf-8')
            if readme_content.strip():
                documents.append({
                    "content": f"README for {repo_full_name}:\n\n{readme_content}",
                    "metadata": {
                        "type": "readme",
                        "file_path": readme.path,
                        "repository": repo_full_name,
                        "sha": readme.sha
                    }
                })
        except Exception as e:
            logger.info(f"No README found or could not fetch README: {e}")
        
        logger.info(f"Fetched {len(documents)} metadata documents from {repo_full_name}")
        return documents
        
    except GithubException as e:
        logger.error(f"GitHub API error fetching repository metadata: {e}")
        return []
    except Exception as e:
        logger.exception(f"Failed to fetch repository metadata from {repo_full_name}")
        return []

@github_rate_limiter.with_rate_limit(category="core")
async def fetch_all_repository_content(
    client: Github,
    repo_full_name: str,
    include_issues: bool = True,
    include_pulls: bool = True,
    max_items: int = 100
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all repository content with rate limit handling."""
    try:
        repo = client.get_repo(repo_full_name)
        content = {
            "files": [],
            "issues": [],
            "pulls": []
        }
        
        # Fetch files
        content["files"] = await fetch_repository_files(client, repo_full_name)
        
        # Fetch issues
        if include_issues:
            issues = repo.get_issues(state="all")
            for i, issue in enumerate(issues):
                if i >= max_items:
                    break
                if not issue.pull_request:  # Skip pull requests
                    comments = issue.get_comments()
                    content["issues"].append({
                        "number": issue.number,
                        "title": issue.title,
                        "body": issue.body,
                        "state": issue.state,
                        "created_at": issue.created_at.isoformat(),
                        "updated_at": issue.updated_at.isoformat(),
                        "comments": [
                            {
                                "body": comment.body,
                                "created_at": comment.created_at.isoformat()
                            }
                            for j, comment in enumerate(comments) if j < max_items
                        ]
                    })
        
        # Fetch pull requests
        if include_pulls:
            pulls = repo.get_pulls(state="all")
            for i, pull in enumerate(pulls):
                if i >= max_items:
                    break
                comments = pull.get_comments()
                content["pulls"].append({
                    "number": pull.number,
                    "title": pull.title,
                    "body": pull.body,
                    "state": pull.state,
                    "created_at": pull.created_at.isoformat(),
                    "updated_at": pull.updated_at.isoformat(),
                    "comments": [
                        {
                            "body": comment.body,
                            "created_at": comment.created_at.isoformat()
                        }
                        for j, comment in enumerate(comments) if j < max_items
                    ]
                })
        
        return content
    
    except Exception as e:
        logger.error(f"Failed to fetch repository content: {e}")
        return {"files": [], "issues": [], "pulls": []} 