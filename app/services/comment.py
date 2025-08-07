"""
Comment Service

Handles GitHub comment operations including:
- Issue comment posting
- PR comment posting
- Comment formatting and templating
- Comment rate limiting and error handling
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base import BaseService
from app.core.github_utils import (
    post_issue_comment,
    post_pr_comment,
    create_pr_review
)
from app.config import settings

@dataclass
class CommentTemplate:
    """Comment template data class."""
    name: str
    template: str
    variables: List[str]
    description: str

class CommentService(BaseService[str]):
    """Service for managing GitHub comments."""
    
    def __init__(self):
        super().__init__("CommentService")
        self._comment_templates = self._initialize_templates()
        self._recent_comments: Dict[str, datetime] = {}
        self._rate_limit_window = timedelta(minutes=1)
        self._max_comments_per_window = 10
    
    def _initialize_templates(self) -> Dict[str, CommentTemplate]:
        """Initialize comment templates."""
        return {
            "issue_analysis": CommentTemplate(
                name="issue_analysis",
                template="""
🤖 **Automated Issue Analysis**

{analysis_content}

---
*This analysis was performed automatically. Please review and take appropriate action.*
                """,
                variables=["analysis_content"],
                description="Template for issue analysis comments"
            ),
            "pr_analysis": CommentTemplate(
                name="pr_analysis",
                template="""
🤖 **Automated PR Analysis**

## Overall Assessment
{overall_assessment}

## Issues Found
{issues_summary}

## Suggestions
{suggestions}

---
*This analysis was performed automatically. Please review and address any issues before merging.*
                """,
                variables=["overall_assessment", "issues_summary", "suggestions"],
                description="Template for PR analysis comments"
            ),
            "fallback_pr": CommentTemplate(
                name="fallback_pr",
                template="""
🤖 **PR Summary**

This PR contains {file_count} changed files.

**Note**: Detailed analysis is currently unavailable. Please review the changes manually.

---
*This is an automated summary. Please review the changes carefully.*
                """,
                variables=["file_count"],
                description="Template for fallback PR comments"
            ),
            "error_response": CommentTemplate(
                name="error_response",
                template="""
❌ **Analysis Error**

We encountered an issue while analyzing your request: {error_message}

**What you can do:**
- Try rephrasing your question
- Check if the repository is properly indexed
- Contact support if the issue persists

---
*This is an automated error response.*
                """,
                variables=["error_message"],
                description="Template for error response comments"
            ),
            "duplicate_issue": CommentTemplate(
                name="duplicate_issue",
                template="""
🔄 **Duplicate Issue Detected**

This issue appears to be similar to existing issues:

{similar_issues}

**Recommendation**: {recommendation}

---
*This analysis was performed automatically.*
                """,
                variables=["similar_issues", "recommendation"],
                description="Template for duplicate issue comments"
            )
        }
    
    async def post_issue_comment(
        self,
        client,
        repo_full_name: str,
        issue_number: int,
        comment_content: str,
        template_name: Optional[str] = None,
        template_vars: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Post a comment on a GitHub issue.
        
        Args:
            client: GitHub client
            repo_full_name: Repository full name
            issue_number: Issue number
            comment_content: Comment content (or template name if template_name is provided)
            template_name: Optional template name to use
            template_vars: Variables for template substitution
            
        Returns:
            True if comment posted successfully, False otherwise
        """
        operation = "post_issue_comment"
        start_time = self.log_operation_start(
            operation, 
            repo=repo_full_name, 
            issue=issue_number,
            template=template_name
        )
        
        try:
            # Check rate limiting
            if not self._check_rate_limit(repo_full_name):
                self.logger.warning(f"Rate limit exceeded for {repo_full_name}")
                return False
            
            # Format comment content
            if template_name:
                content = self._format_template(template_name, template_vars or {})
            else:
                content = comment_content
            
            # Post comment
            success = await post_issue_comment(
                client=client,
                repo_full_name=repo_full_name,
                issue_number=issue_number,
                comment=content
            )
            
            if success:
                self._record_comment(repo_full_name)
                self.log_operation_complete(
                    operation, 
                    start_time, 
                    success=True,
                    comment_length=len(content)
                )
            else:
                self.log_operation_complete(operation, start_time, success=False)
            
            return success
            
        except Exception as e:
            self.log_error(operation, e, repo=repo_full_name, issue=issue_number)
            return False
    
    async def post_pr_comment(
        self,
        client,
        repo_full_name: str,
        pr_number: int,
        comment_content: str,
        template_name: Optional[str] = None,
        template_vars: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Post a comment on a GitHub pull request.
        
        Args:
            client: GitHub client
            repo_full_name: Repository full name
            pr_number: PR number
            comment_content: Comment content (or template name if template_name is provided)
            template_name: Optional template name to use
            template_vars: Variables for template substitution
            
        Returns:
            True if comment posted successfully, False otherwise
        """
        operation = "post_pr_comment"
        start_time = self.log_operation_start(
            operation, 
            repo=repo_full_name, 
            pr=pr_number,
            template=template_name
        )
        
        try:
            # Check rate limiting
            if not self._check_rate_limit(repo_full_name):
                self.logger.warning(f"Rate limit exceeded for {repo_full_name}")
                return False
            
            # Format comment content
            if template_name:
                content = self._format_template(template_name, template_vars or {})
            else:
                content = comment_content
            
            # Post comment
            success = await post_pr_comment(
                client=client,
                repo_full_name=repo_full_name,
                pr_number=pr_number,
                comment=content
            )
            
            if success:
                self._record_comment(repo_full_name)
                self.log_operation_complete(
                    operation, 
                    start_time, 
                    success=True,
                    comment_length=len(content)
                )
            else:
                self.log_operation_complete(operation, start_time, success=False)
            
            return success
            
        except Exception as e:
            self.log_error(operation, e, repo=repo_full_name, pr=pr_number)
            return False
    
    async def create_pr_review(
        self,
        client,
        repo_full_name: str,
        pr_number: int,
        review_body: str,
        event: str = "COMMENT",
        comments: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Create a PR review with comments.
        
        Args:
            client: GitHub client
            repo_full_name: Repository full name
            pr_number: PR number
            review_body: Review body text
            event: Review event (COMMENT, APPROVE, REQUEST_CHANGES)
            comments: List of review comments
            
        Returns:
            True if review created successfully, False otherwise
        """
        operation = "create_pr_review"
        start_time = self.log_operation_start(
            operation, 
            repo=repo_full_name, 
            pr=pr_number,
            event=event
        )
        
        try:
            # Check rate limiting
            if not self._check_rate_limit(repo_full_name):
                self.logger.warning(f"Rate limit exceeded for {repo_full_name}")
                return False
            
            # Create review
            success = await create_pr_review(
                client=client,
                repo_full_name=repo_full_name,
                pr_number=pr_number,
                body=review_body,
                event=event,
                comments=comments or []
            )
            
            if success:
                self._record_comment(repo_full_name)
                self.log_operation_complete(
                    operation, 
                    start_time, 
                    success=True,
                    comments_count=len(comments) if comments else 0
                )
            else:
                self.log_operation_complete(operation, start_time, success=False)
            
            return success
            
        except Exception as e:
            self.log_error(operation, e, repo=repo_full_name, pr=pr_number)
            return False
    
    def _format_template(self, template_name: str, variables: Dict[str, str]) -> str:
        """
        Format a comment template with variables.
        
        Args:
            template_name: Name of the template to use
            variables: Variables to substitute in the template
            
        Returns:
            Formatted comment content
        """
        if template_name not in self._comment_templates:
            self.logger.warning(f"Unknown template: {template_name}")
            return f"Error: Unknown template '{template_name}'"
        
        template = self._comment_templates[template_name]
        content = template.template
        
        # Substitute variables
        for var_name, value in variables.items():
            placeholder = f"{{{var_name}}}"
            content = content.replace(placeholder, str(value))
        
        # Remove any unused placeholders
        import re
        content = re.sub(r'\{[^}]+\}', '', content)
        
        return content.strip()
    
    def _check_rate_limit(self, repo_full_name: str) -> bool:
        """
        Check if we're within rate limits for posting comments.
        
        Args:
            repo_full_name: Repository full name
            
        Returns:
            True if within rate limits, False otherwise
        """
        now = datetime.utcnow()
        window_start = now - self._rate_limit_window
        
        # Count recent comments
        recent_count = sum(
            1 for timestamp in self._recent_comments.values()
            if timestamp > window_start
        )
        
        return recent_count < self._max_comments_per_window
    
    def _record_comment(self, repo_full_name: str) -> None:
        """Record that a comment was posted for rate limiting."""
        self._recent_comments[repo_full_name] = datetime.utcnow()
        
        # Clean up old entries
        now = datetime.utcnow()
        window_start = now - self._rate_limit_window
        
        expired_keys = [
            key for key, timestamp in self._recent_comments.items()
            if timestamp < window_start
        ]
        
        for key in expired_keys:
            del self._recent_comments[key]
    
    def get_template(self, template_name: str) -> Optional[CommentTemplate]:
        """
        Get a comment template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            CommentTemplate object or None if not found
        """
        return self._comment_templates.get(template_name)
    
    def list_templates(self) -> List[str]:
        """
        Get list of available template names.
        
        Returns:
            List of template names
        """
        return list(self._comment_templates.keys())
    
    def add_template(self, template: CommentTemplate) -> None:
        """
        Add a new comment template.
        
        Args:
            template: CommentTemplate object to add
        """
        self._comment_templates[template.name] = template
        self.logger.info(f"Added new template: {template.name}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the comment service."""
        return {
            "status": "healthy",
            "templates_count": len(self._comment_templates),
            "recent_comments_count": len(self._recent_comments),
            "rate_limit_window_minutes": self._rate_limit_window.total_seconds() / 60,
            "max_comments_per_window": self._max_comments_per_window
        }
