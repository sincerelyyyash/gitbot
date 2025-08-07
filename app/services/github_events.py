"""
GitHub Events Service

Handles GitHub webhook events and coordinates with other services:
- Event routing and processing
- Event validation and filtering
- Service coordination
- Error handling and recovery
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

from .base import BaseService
from .repository import RepositoryService
from .analytics import AnalyticsService
from .comment import CommentService
from .rag import RAGService
from .issue_similarity import IssueSimilarityService
from .pr_analysis import PRAnalysisService
from app.models.github import IssueCommentPayload, IssuesPayload, PushPayload
from app.config import settings

@dataclass
class EventContext:
    """Event context data class."""
    event_type: str
    event_action: str
    repo_full_name: str
    installation_id: int
    payload: Dict[str, Any]
    timestamp: datetime

@dataclass
class EventResult:
    """Event processing result."""
    success: bool
    action_taken: str
    response_posted: bool
    error_message: Optional[str]
    processing_time_ms: float
    metadata: Dict[str, Any]

class GitHubEventService(BaseService[EventResult]):
    """Service for handling GitHub webhook events."""
    
    def __init__(self):
        super().__init__("GitHubEventService")
        self._event_handlers = self._initialize_event_handlers()
        self._recent_events: Dict[str, datetime] = {}
        self._event_rate_limit = 100  # Events per minute
    
    def _initialize_event_handlers(self) -> Dict[str, callable]:
        """Initialize event handler functions."""
        return {
            "issue_comment": self._handle_issue_comment_event,
            "issues": self._handle_issue_event,
            "push": self._handle_push_event,
            "pull_request": self._handle_pull_request_event,
            "pull_request_review": self._handle_pull_request_review_event
        }
    
    async def process_event(
        self,
        event_type: str,
        event_action: str,
        payload: Dict[str, Any],
        repo_full_name: str,
        installation_id: int
    ) -> EventResult:
        """
        Process a GitHub webhook event.
        
        Args:
            event_type: Type of GitHub event
            event_action: Action within the event type
            payload: Event payload data
            repo_full_name: Repository full name
            installation_id: GitHub installation ID
            
        Returns:
            EventResult with processing outcome
        """
        operation = "process_event"
        start_time = self.log_operation_start(
            operation,
            event_type=event_type,
            event_action=event_action,
            repo=repo_full_name
        )
        
        try:
            # Check rate limiting
            if not self._check_event_rate_limit(repo_full_name):
                return EventResult(
                    success=False,
                    action_taken="rate_limited",
                    response_posted=False,
                    error_message="Event rate limit exceeded",
                    processing_time_ms=0,
                    metadata={"rate_limited": True}
                )
            
            # Create event context
            context = EventContext(
                event_type=event_type,
                event_action=event_action,
                repo_full_name=repo_full_name,
                installation_id=installation_id,
                payload=payload,
                timestamp=datetime.utcnow()
            )
            
            # Log action start
            action_log_id = await analytics_service.log_action_start(
                action_type=f"event_{event_type}",
                repo_full_name=repo_full_name,
                action_subtype=event_action,
                github_event_type=event_type,
                github_event_action=event_action,
                target_type=self._get_target_type(event_type, payload),
                target_number=self._get_target_number(event_type, payload),
                metadata={"installation_id": installation_id}
            )
            
            # Process event
            result = await self._route_event(context)
            
            # Log action completion
            await analytics_service.log_action_complete(
                action_log_id=action_log_id,
                success=result.success,
                error_message=result.error_message,
                response_posted=result.response_posted,
                metadata=result.metadata
            )
            
            # Record event for rate limiting
            self._record_event(repo_full_name)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.processing_time_ms = processing_time
            
            self.log_operation_complete(
                operation,
                start_time,
                success=result.success,
                action_taken=result.action_taken
            )
            
            return result
            
        except Exception as e:
            self.log_error(operation, e, event_type=event_type, repo=repo_full_name)
            return EventResult(
                success=False,
                action_taken="error",
                response_posted=False,
                error_message=str(e),
                processing_time_ms=0,
                metadata={"error_type": type(e).__name__}
            )
    
    async def _route_event(self, context: EventContext) -> EventResult:
        """Route event to appropriate handler."""
        handler = self._event_handlers.get(context.event_type)
        
        if not handler:
            return EventResult(
                success=False,
                action_taken="no_handler",
                response_posted=False,
                error_message=f"No handler for event type: {context.event_type}",
                processing_time_ms=0,
                metadata={"event_type": context.event_type}
            )
        
        try:
            return await handler(context)
        except Exception as e:
            self.logger.error(f"Event handler failed: {e}", exc_info=True)
            return EventResult(
                success=False,
                action_taken="handler_error",
                response_posted=False,
                error_message=str(e),
                processing_time_ms=0,
                metadata={"error_type": type(e).__name__}
            )
    
    async def _handle_issue_comment_event(self, context: EventContext) -> EventResult:
        """Handle issue comment events."""
        try:
            # Parse payload
            payload = IssueCommentPayload(**context.payload)
            
            # Skip bot comments
            if payload.comment.user.type == "Bot":
                return EventResult(
                    success=True,
                    action_taken="skipped_bot_comment",
                    response_posted=False,
                    error_message=None,
                    processing_time_ms=0,
                    metadata={"bot_user": True}
                )
            
            # Check if comment is a question
            comment_text = payload.comment.body
            if not self._is_question(comment_text):
                return EventResult(
                    success=True,
                    action_taken="not_a_question",
                    response_posted=False,
                    error_message=None,
                    processing_time_ms=0,
                    metadata={"question_detected": False}
                )
            
            # Get GitHub client
            client = await repository_service.get_github_client(context.repo_full_name)
            if not client:
                return EventResult(
                    success=False,
                    action_taken="no_client",
                    response_posted=False,
                    error_message="Could not get GitHub client",
                    processing_time_ms=0,
                    metadata={}
                )
            
            # Get RAG system
            rag_result = await rag_service.get_or_init_repo_knowledge_base(
                repo_full_name=context.repo_full_name,
                installation_id=context.installation_id
            )
            
            if not rag_result or isinstance(rag_result, dict) and rag_result.get("error"):
                # Post error response
                error_msg = rag_result.get("error_message", "Failed to initialize knowledge base") if isinstance(rag_result, dict) else "Failed to initialize knowledge base"
                
                await comment_service.post_issue_comment(
                    client=client,
                    repo_full_name=context.repo_full_name,
                    issue_number=payload.issue.number,
                    comment_content="",
                    template_name="error_response",
                    template_vars={"error_message": error_msg}
                )
                
                return EventResult(
                    success=False,
                    action_taken="rag_error",
                    response_posted=True,
                    error_message=error_msg,
                    processing_time_ms=0,
                    metadata={"rag_error": True}
                )
            
            # Query RAG system
            response, _ = await rag_service.query_rag_system(
                rag_result, comment_text, chat_history=[]
            )
            
            # Post response
            success = await comment_service.post_issue_comment(
                client=client,
                repo_full_name=context.repo_full_name,
                issue_number=payload.issue.number,
                comment_content=response
            )
            
            return EventResult(
                success=success,
                action_taken="answered_question",
                response_posted=success,
                error_message=None,
                processing_time_ms=0,
                metadata={
                    "question_detected": True,
                    "response_length": len(response) if response else 0
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling issue comment event: {e}")
            return EventResult(
                success=False,
                action_taken="error",
                response_posted=False,
                error_message=str(e),
                processing_time_ms=0,
                metadata={"error_type": type(e).__name__}
            )
    
    async def _handle_issue_event(self, context: EventContext) -> EventResult:
        """Handle issue events."""
        try:
            # Parse payload
            payload = IssuesPayload(**context.payload)
            
            # Only process opened issues
            if payload.action != "opened":
                return EventResult(
                    success=True,
                    action_taken="skipped_non_opened",
                    response_posted=False,
                    error_message=None,
                    processing_time_ms=0,
                    metadata={"action": payload.action}
                )
            
            # Get GitHub client
            client = await repository_service.get_github_client(context.repo_full_name)
            if not client:
                return EventResult(
                    success=False,
                    action_taken="no_client",
                    response_posted=False,
                    error_message="Could not get GitHub client",
                    processing_time_ms=0,
                    metadata={}
                )
            
            # Analyze issue similarity
            similarity_result = await issue_similarity_service.analyze_issue_similarity(
                new_issue=payload.issue.dict(),
                repo_full_name=context.repo_full_name,
                installation_id=context.installation_id
            )
            
            # Log issue analysis
            await analytics_service.log_issue_analysis(
                action_log_id=None,
                repo_full_name=context.repo_full_name,
                issue_number=payload.issue.number,
                issue_title=payload.issue.title,
                issue_body_length=len(payload.issue.body) if payload.issue.body else 0,
                is_question=self._is_question(payload.issue.title),
                is_duplicate=similarity_result.has_duplicates,
                is_invalid=similarity_result.is_likely_invalid,
                similar_issues_found=len(similarity_result.similar_issues),
                highest_similarity_score=max([issue.similarity_score for issue in similarity_result.similar_issues]) if similarity_result.similar_issues else None,
                similar_issue_numbers=[issue.issue_number for issue in similarity_result.similar_issues[:5]],
                response_generated=False
            )
            
            # Post similarity analysis if needed
            response_posted = False
            if similarity_result.has_duplicates or similarity_result.is_likely_invalid:
                response_posted = await issue_similarity_service.auto_comment_on_similar_issues(
                    issue_number=payload.issue.number,
                    repo_full_name=context.repo_full_name,
                    installation_id=context.installation_id,
                    similarity_result=similarity_result
                )
            
            return EventResult(
                success=True,
                action_taken="analyzed_issue",
                response_posted=response_posted,
                error_message=None,
                processing_time_ms=0,
                metadata={
                    "has_duplicates": similarity_result.has_duplicates,
                    "is_invalid": similarity_result.is_likely_invalid,
                    "similar_issues_count": len(similarity_result.similar_issues)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling issue event: {e}")
            return EventResult(
                success=False,
                action_taken="error",
                response_posted=False,
                error_message=str(e),
                processing_time_ms=0,
                metadata={"error_type": type(e).__name__}
            )
    
    async def _handle_push_event(self, context: EventContext) -> EventResult:
        """Handle push events."""
        try:
            # Parse payload
            payload = PushPayload(**context.payload)
            
            # Schedule reindexing
            await rag_service.schedule_reindex(
                repo_full_name=context.repo_full_name,
                installation_id=context.installation_id,
                delay_minutes=60
            )
            
            return EventResult(
                success=True,
                action_taken="scheduled_reindex",
                response_posted=False,
                error_message=None,
                processing_time_ms=0,
                metadata={
                    "ref": payload.ref,
                    "commits_count": len(payload.commits)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling push event: {e}")
            return EventResult(
                success=False,
                action_taken="error",
                response_posted=False,
                error_message=str(e),
                processing_time_ms=0,
                metadata={"error_type": type(e).__name__}
            )
    
    async def _handle_pull_request_event(self, context: EventContext) -> EventResult:
        """Handle pull request events."""
        try:
            # This would handle PR opened, updated, closed events
            # For now, return a basic result
            return EventResult(
                success=True,
                action_taken="processed_pr_event",
                response_posted=False,
                error_message=None,
                processing_time_ms=0,
                metadata={"event_action": context.event_action}
            )
            
        except Exception as e:
            self.logger.error(f"Error handling pull request event: {e}")
            return EventResult(
                success=False,
                action_taken="error",
                response_posted=False,
                error_message=str(e),
                processing_time_ms=0,
                metadata={"error_type": type(e).__name__}
            )
    
    async def _handle_pull_request_review_event(self, context: EventContext) -> EventResult:
        """Handle pull request review events."""
        try:
            # This would handle PR review events
            # For now, return a basic result
            return EventResult(
                success=True,
                action_taken="processed_pr_review",
                response_posted=False,
                error_message=None,
                processing_time_ms=0,
                metadata={"event_action": context.event_action}
            )
            
        except Exception as e:
            self.logger.error(f"Error handling pull request review event: {e}")
            return EventResult(
                success=False,
                action_taken="error",
                response_posted=False,
                error_message=str(e),
                processing_time_ms=0,
                metadata={"error_type": type(e).__name__}
            )
    
    def _is_question(self, text: str) -> bool:
        """Determine if text is a question."""
        if not text:
            return False
        
        text_lower = text.lower().strip()
        
        # Keywords that often start a question
        question_starters = [
            "how", "what", "when", "where", "who", "why", "which",
            "can", "could", "do", "does", "is", "are", "will", "would",
            "should", "shall", "may", "might", "must", "need", "want",
            "help", "explain", "understand", "clarify", "tell me"
        ]
        
        # Check if the text starts with a question word
        starts_with_question = any(text_lower.startswith(word) for word in question_starters)
        
        # Check if the text ends with a question mark
        ends_with_question_mark = text_lower.endswith("?")
        
        return starts_with_question or ends_with_question_mark
    
    def _get_target_type(self, event_type: str, payload: Dict[str, Any]) -> Optional[str]:
        """Get target type from event payload."""
        if event_type == "issue_comment":
            return "issue"
        elif event_type == "issues":
            return "issue"
        elif event_type == "pull_request":
            return "pull_request"
        elif event_type == "push":
            return "repository"
        return None
    
    def _get_target_number(self, event_type: str, payload: Dict[str, Any]) -> Optional[int]:
        """Get target number from event payload."""
        if event_type == "issue_comment" and "issue" in payload:
            return payload["issue"].get("number")
        elif event_type == "issues" and "issue" in payload:
            return payload["issue"].get("number")
        elif event_type == "pull_request" and "pull_request" in payload:
            return payload["pull_request"].get("number")
        return None
    
    def _check_event_rate_limit(self, repo_full_name: str) -> bool:
        """Check if we're within event rate limits."""
        now = datetime.utcnow()
        window_start = now.replace(second=0, microsecond=0)
        
        # Count recent events
        recent_count = sum(
            1 for timestamp in self._recent_events.values()
            if timestamp >= window_start
        )
        
        return recent_count < self._event_rate_limit
    
    def _record_event(self, repo_full_name: str) -> None:
        """Record that an event was processed."""
        self._recent_events[repo_full_name] = datetime.utcnow()
        
        # Clean up old entries (older than 1 minute)
        now = datetime.utcnow()
        window_start = now.replace(second=0, microsecond=0)
        
        expired_keys = [
            key for key, timestamp in self._recent_events.items()
            if timestamp < window_start
        ]
        
        for key in expired_keys:
            del self._recent_events[key]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the GitHub events service."""
        return {
            "status": "healthy",
            "event_handlers_count": len(self._event_handlers),
            "recent_events_count": len(self._recent_events),
            "event_rate_limit": self._event_rate_limit
        }
