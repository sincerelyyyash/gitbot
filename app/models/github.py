from pydantic import BaseModel
from typing import Optional, List

class Repository(BaseModel):
    full_name: str

class Issue(BaseModel):
    number: int
    body: Optional[str] = None
    title: Optional[str] = None

class Comment(BaseModel):
    body: str

class Installation(BaseModel):
    id: int

class Commit(BaseModel):
    id: str
    message: str
    added: List[str]
    removed: List[str]
    modified: List[str]

class PushPayload(BaseModel):
    ref: str
    repository: Repository
    installation: Installation
    commits: List[Commit]
    before: str
    after: str

class IssueCommentPayload(BaseModel):
    action: str
    comment: Comment
    repository: Repository
    issue: Issue
    installation: Installation

class IssuesPayload(BaseModel):
    action: str
    repository: Repository
    issue: Issue
    installation: Installation 