from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class User(BaseModel):
    login: str
    id: int
    type: str
    site_admin: bool = False

class Repository(BaseModel):
    id: int
    name: str
    full_name: str
    private: bool
    owner: Optional[User] = None  # Make owner optional for edge cases
    default_branch: str = "main"
    # Allow additional fields
    class Config:
        extra = "allow"

class SimpleRepository(BaseModel):
    """Simplified repository model for installation events where full data might not be available."""
    id: int
    name: str
    full_name: str
    private: bool
    node_id: Optional[str] = None
    # Allow additional fields - don't require owner for installation events
    class Config:
        extra = "allow"

class Installation(BaseModel):
    id: int
    # Allow additional fields
    class Config:
        extra = "allow"

class Issue(BaseModel):
    id: int
    number: int
    title: str
    body: Optional[str] = None
    user: User
    state: str
    locked: bool = False
    comments: int = 0
    created_at: str
    updated_at: str
    closed_at: Optional[str] = None
    labels: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    # Allow additional fields
    class Config:
        extra = "allow"

class PullRequest(BaseModel):
    id: int
    number: int
    title: str
    body: Optional[str] = None
    user: User
    state: str  # "open", "closed"
    merged: bool = False
    merged_at: Optional[str] = None
    merge_commit_sha: Optional[str] = None
    head: Dict[str, Any]  # Contains branch and commit info
    base: Dict[str, Any]  # Contains target branch info
    draft: bool = False
    mergeable: Optional[bool] = None
    mergeable_state: Optional[str] = None
    merged_by: Optional[User] = None
    comments: int = 0
    review_comments: int = 0
    commits: int = 0
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    created_at: str
    updated_at: str
    closed_at: Optional[str] = None
    labels: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    assignees: Optional[List[User]] = Field(default_factory=list)
    requested_reviewers: Optional[List[User]] = Field(default_factory=list)
    # Allow additional fields
    class Config:
        extra = "allow"

class PRFile(BaseModel):
    sha: str
    filename: str
    status: str  # "added", "removed", "modified", "renamed"
    additions: int
    deletions: int
    changes: int
    blob_url: Optional[str] = None
    raw_url: Optional[str] = None
    contents_url: Optional[str] = None
    patch: Optional[str] = None
    previous_filename: Optional[str] = None
    # Allow additional fields
    class Config:
        extra = "allow"

class ReviewComment(BaseModel):
    id: int
    user: User
    body: str
    path: str
    position: Optional[int] = None
    original_position: Optional[int] = None
    commit_id: str
    original_commit_id: str
    created_at: str
    updated_at: str
    url: str
    html_url: str
    pull_request_url: str
    diff_hunk: Optional[str] = None
    line: Optional[int] = None
    original_line: Optional[int] = None
    side: Optional[str] = None  # "LEFT" or "RIGHT"
    start_line: Optional[int] = None
    original_start_line: Optional[int] = None
    start_side: Optional[str] = None
    # Allow additional fields
    class Config:
        extra = "allow"

class Review(BaseModel):
    id: int
    user: User
    body: Optional[str] = None
    state: str  # "PENDING", "APPROVED", "CHANGES_REQUESTED", "COMMENTED"
    html_url: str
    pull_request_url: str
    submitted_at: Optional[str] = None
    commit_id: Optional[str] = None
    # Allow additional fields
    class Config:
        extra = "allow"

class Comment(BaseModel):
    id: int
    user: User
    created_at: str
    updated_at: str
    body: str
    # Allow additional fields
    class Config:
        extra = "allow"

class Commit(BaseModel):
    id: str
    message: str
    added: List[str] = Field(default_factory=list)
    removed: List[str] = Field(default_factory=list)
    modified: List[str] = Field(default_factory=list)
    # Allow additional fields
    class Config:
        extra = "allow"

class PushPayload(BaseModel):
    ref: str
    repository: Repository
    installation: Installation
    commits: List[Commit]
    before: str
    after: str
    # Allow additional fields
    class Config:
        extra = "allow"

class IssueCommentPayload(BaseModel):
    action: str
    comment: Comment
    repository: Repository
    issue: Issue
    installation: Installation
    sender: Optional[User] = None
    # Allow additional fields
    class Config:
        extra = "allow"

class IssuesPayload(BaseModel):
    action: str
    repository: Repository
    issue: Issue
    installation: Installation
    sender: Optional[User] = None
    # Allow additional fields
    class Config:
        extra = "allow"

class PullRequestPayload(BaseModel):
    action: str  # "opened", "closed", "reopened", "edited", "assigned", "unassigned", "review_requested", "review_request_removed", "labeled", "unlabeled", "synchronize"
    number: int
    repository: Repository
    installation: Installation
    pull_request: PullRequest
    sender: Optional[User] = None
    changes: Optional[Dict[str, Any]] = None  # Present for "edited" action
    # Allow additional fields
    class Config:
        extra = "allow"

class PullRequestReviewPayload(BaseModel):
    action: str  # "submitted", "edited", "dismissed"
    repository: Repository
    installation: Installation
    pull_request: PullRequest
    review: Review
    sender: Optional[User] = None
    # Allow additional fields
    class Config:
        extra = "allow"

class PullRequestReviewCommentPayload(BaseModel):
    action: str  # "created", "edited", "deleted"
    repository: Repository
    installation: Installation
    pull_request: PullRequest
    comment: ReviewComment
    sender: Optional[User] = None
    # Allow additional fields
    class Config:
        extra = "allow"

class InstallationPayload(BaseModel):
    action: str  # "created", "deleted", etc.
    installation: Installation
    repositories: Optional[List[SimpleRepository]] = None  # Use SimpleRepository for more flexibility
    sender: Optional[User] = None  # Make sender optional for deletion events
    # Allow additional fields
    class Config:
        extra = "allow"

class InstallationRepositoriesPayload(BaseModel):
    action: str  # "added", "removed"
    installation: Installation
    repositories_added: Optional[List[SimpleRepository]] = None  # Use SimpleRepository
    repositories_removed: Optional[List[SimpleRepository]] = None  # Use SimpleRepository
    repository_selection: str
    sender: Optional[User] = None  # Make sender optional
    # Allow additional fields
    class Config:
        extra = "allow" 