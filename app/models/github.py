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
    owner: User
    description: Optional[str] = None
    language: Optional[str] = None
    default_branch: str = "main"
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
    # Allow additional fields
    class Config:
        extra = "allow"

class Comment(BaseModel):
    id: int
    body: str
    user: User
    created_at: str
    updated_at: str
    # Allow additional fields
    class Config:
        extra = "allow"

class Installation(BaseModel):
    id: int
    node_id: Optional[str] = None
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