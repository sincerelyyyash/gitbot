from app.core.rag_system import (
    initialize_rag_system, 
    query_rag_system,
    get_collection_info,
    delete_collection,
    list_collections,
    _sanitize_collection_name
)
from app.core.github_utils import (
    get_github_app_installation_client, 
    post_issue_comment,
    fetch_all_repository_content,
    fetch_repository_files
)
from app.core.quota_manager import quota_manager
from app.config import settings
from app.models.github import IssueCommentPayload, IssuesPayload, PushPayload
import logging
import os
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Set, Optional, Tuple

logger = logging.getLogger("rag_service")

# Global dictionary to store repository knowledge bases
repo_knowledge_base = {}

# Track last activity for repositories
repo_last_activity: Dict[str, datetime] = {}

# Track scheduled tasks
scheduled_tasks = set()

# Track recent error responses to prevent duplicate fallback messages
recent_error_responses: Dict[str, datetime] = {}

# Circuit breaker: track error counts per repository
repo_error_counts: Dict[str, int] = {}
repo_circuit_breaker: Dict[str, datetime] = {}

def is_issue_a_question(title: str) -> bool:
    """Determine if an issue title is likely a question."""
    title_lower = title.lower().strip()
    # Keywords that often start a question
    question_starters = [
        "how", "what", "when", "where", "who", "why", "which",
        "can", "could", "do", "does", "is", "are", "will", "would"
    ]
    # Check if the title starts with a question word or ends with a question mark
    return any(title_lower.startswith(word) for word in question_starters) or title_lower.endswith("?")

def update_repo_activity(repo_full_name: str):
    """Update last activity timestamp for a repository."""
    repo_last_activity[repo_full_name] = datetime.utcnow()

def should_send_error_response(repo_full_name: str, issue_number: int, error_type: str) -> bool:
    """Check if we should send an error response or if we've already sent one recently."""
    key = f"{repo_full_name}:{issue_number}:{error_type}"
    now = datetime.utcnow()
    
    # Check if we've sent this error type for this issue recently (within last 5 minutes)
    if key in recent_error_responses:
        last_response = recent_error_responses[key]
        if now - last_response < timedelta(minutes=5):
            logger.info(f"Skipping duplicate error response for {key}")
            return False
    
    # Record this error response
    recent_error_responses[key] = now
    
    # Clean up old entries (older than 1 hour)
    cutoff_time = now - timedelta(hours=1)
    keys_to_remove = [k for k, v in recent_error_responses.items() if v < cutoff_time]
    for k in keys_to_remove:
        del recent_error_responses[k]
    
    return True

def is_circuit_breaker_open(repo_full_name: str) -> bool:
    """Check if circuit breaker is open (too many recent errors)."""
    now = datetime.utcnow()
    
    # Check if circuit breaker is currently open
    if repo_full_name in repo_circuit_breaker:
        breaker_time = repo_circuit_breaker[repo_full_name]
        # Circuit breaker is open for 10 minutes after 5 consecutive errors
        if now - breaker_time < timedelta(minutes=10):
            return True
        else:
            # Reset circuit breaker after timeout
            del repo_circuit_breaker[repo_full_name]
            repo_error_counts[repo_full_name] = 0
    
    return False

def increment_error_count(repo_full_name: str):
    """Increment error count and potentially open circuit breaker."""
    repo_error_counts[repo_full_name] = repo_error_counts.get(repo_full_name, 0) + 1
    
    # Open circuit breaker after 5 consecutive errors
    if repo_error_counts[repo_full_name] >= 5:
        repo_circuit_breaker[repo_full_name] = datetime.utcnow()
        logger.warning(f"Circuit breaker opened for {repo_full_name} due to repeated errors")

def reset_error_count(repo_full_name: str):
    """Reset error count on successful operation."""
    if repo_full_name in repo_error_counts:
        repo_error_counts[repo_full_name] = 0

async def get_or_init_repo_knowledge_base(
    repo_full_name: str, 
    installation_id: int,
    include_current_content: bool = True,
    current_documents_data: list = None,
    force_refresh: bool = False
):
    """
    Get or initialize the RAG knowledge base for a repository with persistent storage.
    
    Args:
        repo_full_name: Repository full name (owner/repo)
        installation_id: GitHub installation ID
        include_current_content: Whether to include the current issue/comment content
        current_documents_data: Additional documents to include (e.g., current issue/comment)
        force_refresh: Whether to force refresh of the collection
    """
    collection_name = _sanitize_collection_name(repo_full_name)
    persist_dir = os.path.join(settings.chromadb_persist_dir, "repositories")
    
    # Check if we have this repository in memory and it's still valid
    if (not force_refresh and 
        repo_full_name in repo_knowledge_base and 
        repo_knowledge_base[repo_full_name].get("collection_name") == collection_name):
        
        logger.info(f"Using existing in-memory RAG system for {repo_full_name}")
        
        # Add current content if provided
        if include_current_content and current_documents_data:
            # For real-time content, we might want to add it temporarily without persisting
            logger.info(f"Current content will be included in context for {repo_full_name}")
        
        return repo_knowledge_base[repo_full_name]
    
    logger.info(f"Initializing RAG knowledge base for {repo_full_name}")
    logger.info(f"Collection name: {collection_name}, Persist dir: {persist_dir}")
    
    # Get GitHub client for content fetching
    client = await get_github_app_installation_client(
        settings.github_app_id, 
        settings.github_private_key, 
        installation_id
    )
    if not client:
        logger.error("Could not authenticate GitHub client for content fetching.")
        return None
    
    try:
        # Fetch comprehensive repository content
        logger.info(f"Fetching repository content for {repo_full_name}")
        repo_content = await fetch_all_repository_content(
            client=client,
            repo_full_name=repo_full_name,
            include_issues=True,
            include_pulls=True,
            max_items=50
        )
        
        # Convert the content into documents
        repo_documents = []
        
        # Add files
        for file in repo_content.get("files", []):
            repo_documents.append({
                "content": file["content"],
                "metadata": {
                    "type": "file",
                    "file_path": file["path"],
                    "repository": repo_full_name
                }
            })
        
        # Add issues
        for issue in repo_content.get("issues", []):
            if issue.get("body"):
                repo_documents.append({
                    "content": f"Issue #{issue['number']}: {issue['title']}\n\n{issue['body']}",
                    "metadata": {
                        "type": "issue",
                        "issue_number": issue["number"],
                        "repository": repo_full_name,
                        "created_at": issue["created_at"]
                    }
                })
                # Add issue comments
                for comment in issue.get("comments", []):
                    if comment.get("body"):
                        repo_documents.append({
                            "content": f"Comment on Issue #{issue['number']}:\n\n{comment['body']}",
                            "metadata": {
                                "type": "issue_comment",
                                "issue_number": issue["number"],
                                "repository": repo_full_name,
                                "created_at": comment["created_at"]
                            }
                        })
        
        # Add pull requests
        for pr in repo_content.get("pulls", []):
            if pr.get("body"):
                repo_documents.append({
                    "content": f"Pull Request #{pr['number']}: {pr['title']}\n\n{pr['body']}",
                    "metadata": {
                        "type": "pull_request",
                        "pr_number": pr["number"],
                        "repository": repo_full_name,
                        "created_at": pr["created_at"]
                    }
                })
                # Add PR comments
                for comment in pr.get("comments", []):
                    if comment.get("body"):
                        repo_documents.append({
                            "content": f"Comment on PR #{pr['number']}:\n\n{comment['body']}",
                            "metadata": {
                                "type": "pr_comment",
                                "pr_number": pr["number"],
                                "repository": repo_full_name,
                                "created_at": comment["created_at"]
                            }
                        })
        
        logger.info(f"Fetched {len(repo_documents)} documents from repository {repo_full_name}")
        
        # Add current content if provided
        if include_current_content and current_documents_data:
            repo_documents.extend(current_documents_data)
            logger.info(f"Added {len(current_documents_data)} current context documents")
        
        # Ensure we have some content to work with
        if not repo_documents:
            logger.warning(f"No documents found for repository {repo_full_name}")
            repo_documents = [{
                "content": f"Repository: {repo_full_name}\nNo content available for indexing.",
                "metadata": {"type": "placeholder", "repository": repo_full_name}
            }]
        
        # Initialize RAG system, resetting the collection to ensure a fresh start
        rag_result = await initialize_rag_system(
            documents_data=repo_documents,
            gemini_api_key=settings.gemini_api_key,
            chroma_persist_dir=persist_dir,
            collection_name=collection_name,
            reset_collection=True
        )
        
        if "error" in rag_result:
            error_type = rag_result.get("error_type", "UnknownError")
            error_message = rag_result["error"]
            suggestions = rag_result.get("suggestions", [])
            
            logger.error(f"RAG initialization error for {repo_full_name}: {error_message}")
            
            if suggestions:
                logger.info(f"Suggestions for {repo_full_name}: {'; '.join(suggestions)}")
            
            return {
                "error": True,
                "error_type": error_type,
                "error_message": error_message,
                "suggestions": suggestions,
                "fallback_available": rag_result.get("fallback_available", False)
            }
        
        # Store in memory for quick access
        repo_knowledge_base[repo_full_name] = rag_result
        
        # Log collection information
        collection_info = await get_collection_info(rag_result)
        logger.info(f"Successfully initialized RAG knowledge base for {repo_full_name}")
        logger.info(f"Collection info: {collection_info}")
        
        return rag_result
        
    except Exception as e:
        logger.exception(f"Failed to initialize repository knowledge base for {repo_full_name}")
        return None

async def handle_push_event(payload: PushPayload):
    """Handle repository push events with incremental updates."""
    repo_full_name = payload.repository.full_name
    installation_id = payload.installation.id
    
    logger.info(f"Handling push event for {repo_full_name}")
    update_repo_activity(repo_full_name)
    
    # Skip if not on default branch (usually main/master)
    if not payload.ref.endswith("main") and not payload.ref.endswith("master"):
        logger.info(f"Skipping push event for non-main branch: {payload.ref}")
        return
    
    # Get GitHub client
    client = await get_github_app_installation_client(
        settings.github_app_id,
        settings.github_private_key,
        installation_id
    )
    if not client:
        logger.error("Could not authenticate GitHub client")
        return
    
    # Collect changed files
    changed_files: Set[str] = set()
    for commit in payload.commits:
        changed_files.update(commit.added)
        changed_files.update(commit.modified)
        # We'll handle removed files separately
        
    if not changed_files:
        logger.info("No files to update")
        return
    
    # Fetch content for changed files
    try:
        new_documents = await fetch_repository_files(
            client=client,
            repo_full_name=repo_full_name,
            include_code=True,
            include_docs=True,
            max_file_size=512 * 1024
        )
        
        # Filter to only changed files
        new_documents = [
            doc for doc in new_documents
            if doc.get("metadata", {}).get("file_path") in changed_files
        ]
        
        if not new_documents:
            logger.info("No valid documents to update")
            return
        
        # Get or initialize RAG system
        rag = await get_or_init_repo_knowledge_base(
            repo_full_name,
            installation_id,
            include_current_content=True,
            current_documents_data=new_documents,
            force_refresh=False  # We'll update incrementally
        )
        
        if not rag:
            logger.error("Could not initialize/access RAG system")
            return
        
        # Update the knowledge base
        logger.info(f"Updating {len(new_documents)} documents in knowledge base")
        
        # Handle removed files if any
        removed_files = set()
        for commit in payload.commits:
            removed_files.update(commit.removed)
        
        if removed_files:
            # Note: Implementation depends on your vector store. For ChromaDB:
            # You might need to delete and recreate the collection, or implement
            # document deletion based on metadata matching
            logger.info(f"Files were removed, scheduling full refresh")
            await schedule_reindex(repo_full_name, installation_id, delay_minutes=5)
        
    except Exception as e:
        logger.exception(f"Error handling push event for {repo_full_name}")
        await schedule_reindex(repo_full_name, installation_id, delay_minutes=60)

async def schedule_reindex(repo_full_name: str, installation_id: int, delay_minutes: int = 60):
    """Schedule a repository for reindexing after a delay."""
    task_key = f"reindex_{repo_full_name}"
    
    # Cancel existing task if any
    if task_key in scheduled_tasks:
        logger.info(f"Cancelling existing reindex task for {repo_full_name}")
        return
    
    scheduled_tasks.add(task_key)
    
    try:
        logger.info(f"Scheduled reindex for {repo_full_name} in {delay_minutes} minutes")
        await asyncio.sleep(delay_minutes * 60)
        await refresh_repository_knowledge_base(repo_full_name, installation_id)
    finally:
        scheduled_tasks.remove(task_key)

async def cleanup_inactive_collections():
    """Clean up collections for repositories with no recent activity."""
    while True:
        try:
            current_time = datetime.utcnow()
            inactive_threshold = current_time - timedelta(days=30)  # 30 days inactive
            
            # Find inactive repositories
            inactive_repos = [
                repo for repo, last_activity in repo_last_activity.items()
                if last_activity < inactive_threshold
            ]
            
            for repo in inactive_repos:
                logger.info(f"Cleaning up inactive repository: {repo}")
                await delete_repository_collection(repo)
                del repo_last_activity[repo]
                if repo in repo_knowledge_base:
                    del repo_knowledge_base[repo]
            
            # Sleep for 24 hours
            await asyncio.sleep(24 * 60 * 60)
            
        except Exception as e:
            logger.exception("Error in cleanup task")
            await asyncio.sleep(60 * 60)  # Retry in 1 hour

async def handle_issue_comment_event(payload: IssueCommentPayload):
    """Handle issue comment events with comprehensive repository context."""
    repo_full_name = payload.repository.full_name
    issue_number = payload.issue.number
    installation_id = payload.installation.id
    issue_content = payload.issue.body or ""
    issue_title = getattr(payload.issue, 'title', '') or f"Issue #{issue_number}"
    comment = payload.comment.body
    
    logger.info(f"Handling issue comment event for {repo_full_name} issue #{issue_number}")
    update_repo_activity(repo_full_name)
    
    # Check circuit breaker
    if is_circuit_breaker_open(repo_full_name):
        logger.info(f"Circuit breaker is open for {repo_full_name}, skipping response")
        return
    
    # Check quota before proceeding
    if not await quota_manager.check_quota(repo_full_name):
        client = await get_github_app_installation_client(
            settings.github_app_id,
            settings.github_private_key,
            installation_id
        )
        if client:
            quota_stats = await quota_manager.get_usage_stats(repo_full_name)
            quota_message = (
                "I've reached my API quota limit for today. "
                f"Currently used {quota_stats['tokens_used_today']} tokens. "
                "Please try again later."
            )
            await post_issue_comment(client, repo_full_name, issue_number, quota_message)
        return
    
    # Prepare current context documents
    current_documents = []
    if issue_content.strip():
        current_documents.append({
            "content": f"Current Issue #{issue_number}: {issue_title}\n\n{issue_content}",
            "metadata": {
                "type": "current_issue", 
                "issue_number": issue_number,
                "repository": repo_full_name,
                "timestamp": "current"
            }
        })
    
    if comment.strip():
        current_documents.append({
            "content": f"Current Comment on Issue #{issue_number}:\n\n{comment}",
            "metadata": {
                "type": "current_comment", 
                "issue_number": issue_number,
                "repository": repo_full_name,
                "timestamp": "current"
            }
        })
    
    # Get or initialize repository knowledge base
    rag = await get_or_init_repo_knowledge_base(
        repo_full_name,
        installation_id,
        include_current_content=True,
        current_documents_data=current_documents
    )
    
    # Handle RAG initialization errors with fallback responses
    if not rag or (isinstance(rag, dict) and rag.get("error")):
        client = await get_github_app_installation_client(
            settings.github_app_id,
            settings.github_private_key,
            installation_id
        )
        if client:
            if isinstance(rag, dict) and rag.get("error"):
                error_type = rag.get("error_type", "UnknownError")
                error_message = rag.get("error_message", "Unknown error occurred")
                suggestions = rag.get("suggestions", [])
                
                # Check if we should send error response (prevent duplicates)
                if not should_send_error_response(repo_full_name, issue_number, error_type):
                    return
                
                if error_type == "APIKeyRestrictedError":
                    fallback_message = (
                        "⚠️ **Configuration Issue**\n\n"
                        "I'm currently unable to process your question due to API key restrictions. "
                        "The repository administrator needs to:\n"
                        "• Remove IP address restrictions from the Google API key, or\n"
                        "• Add the current server IP to the allowed list\n\n"
                        "Please contact the repository administrator to resolve this issue."
                    )
                elif error_type == "QuotaExceededError":
                    fallback_message = (
                        "⚠️ **Quota Exceeded**\n\n"
                        "I've reached the daily API quota limit. Please try again tomorrow "
                        "or contact the repository administrator to increase the quota."
                    )
                elif error_type == "InvalidAPIKeyError":
                    fallback_message = (
                        "⚠️ **Configuration Issue**\n\n"
                        "There's an issue with the API key configuration. "
                        "Please contact the repository administrator to resolve this."
                    )
                else:
                    fallback_message = (
                        "⚠️ **Service Temporarily Unavailable**\n\n"
                        "I'm currently unable to process your question due to a technical issue. "
                        "Please try again later or contact the repository administrator if the problem persists."
                    )
            else:
                # Check if we should send error response (prevent duplicates)
                if not should_send_error_response(repo_full_name, issue_number, "InitializationError"):
                    return
                    
                fallback_message = (
                    "I'm currently setting up the knowledge base for this repository. "
                    "Please try asking your question again in a few minutes."
                )
            
            await post_issue_comment(client, repo_full_name, issue_number, fallback_message)
        
        # Increment error count for circuit breaker
        increment_error_count(repo_full_name)
        return
    
    try:
        # Initialize GitHub client first
        client = await get_github_app_installation_client(
            settings.github_app_id,
            settings.github_private_key,
            installation_id
        )
        if not client:
            logger.error("Failed to initialize GitHub client")
            return

        # Build a chat history with the original issue context
        chat_history = []
        if issue_content.strip():
            chat_history.append({
                "role": "user",
                "content": f"The original issue is titled '{issue_title}' and has the following content: {issue_content}"
            })

        # The user's new comment is the question
        question = comment

        # Query RAG system
        result = await query_rag_system(
            rag,
            question,
            repo_full_name,
            chat_history=chat_history,
            github_client=client
        )
        
        # Handle both tuple and string returns
        if isinstance(result, tuple):
            answer, _ = result  # Ignore usage stats
        else:
            # If we got a string, it's an error message
            answer = result
        
        # Post response
        await post_issue_comment(client, repo_full_name, issue_number, answer)
        
        # Reset error count on successful operation
        reset_error_count(repo_full_name)
    
    except Exception as e:
        logger.exception(f"Error processing comment for {repo_full_name}")
        
        # Only send error message if we haven't sent one recently
        if should_send_error_response(repo_full_name, issue_number, "ProcessingError"):
            client = await get_github_app_installation_client(
                settings.github_app_id,
                settings.github_private_key,
                installation_id
            )
            if client:
                error_message = (
                    "I encountered an error while processing your question. "
                    "Please try again later or contact the repository administrators."
                )
                await post_issue_comment(client, repo_full_name, issue_number, error_message)
                
        # Increment error count for circuit breaker
        increment_error_count(repo_full_name)

async def handle_issue_event(payload: IssuesPayload):
    """
    Handle 'issues' webhook events.
    - When an issue is opened, analyze it and provide a summary or suggestions.
    - If the issue is identified as a question, provide a direct answer.
    - When an issue is edited, re-evaluate if needed.
    """
    action = payload.action
    issue = payload.issue
    repo_full_name = payload.repository.full_name
    installation_id = payload.installation.id

    logger.info(f"Handling issue event: {action} for issue #{issue.number} in {repo_full_name}")

    if action not in ["opened", "edited"]:
        logger.info(f"Ignoring issue event action: {action}")
        return

    if is_circuit_breaker_open(repo_full_name):
        logger.warning(f"Circuit breaker is open for {repo_full_name}, skipping issue event.")
        return

    update_repo_activity(repo_full_name)

    # Initialize GitHub client first
    client = await get_github_app_installation_client(
        settings.github_app_id,
        settings.github_private_key,
        installation_id
    )
    if not client:
        logger.error("Failed to initialize GitHub client")
        return

    # Check for rate limiting
    if not await quota_manager.check_quota(repo_full_name):
        logger.warning(f"Rate limit exceeded for {repo_full_name}. Skipping issue event.")
        if should_send_error_response(repo_full_name, issue.number, "rate_limit"):
            await post_issue_comment(
                client,
                repo_full_name,
                issue.number,
                "I've reached my processing limit for now. Please try again later."
            )
        return

    # Prepare current issue content
    current_documents = [{
        "content": f"Current Issue #{issue.number}: {issue.title}\n\n{issue.body}",
        "metadata": {
            "type": "current_issue",
            "issue_number": issue.number,
            "repository": repo_full_name
        }
    }]

    # Get or initialize the knowledge base
    rag_system = await get_or_init_repo_knowledge_base(
        repo_full_name=repo_full_name,
        installation_id=installation_id,
        current_documents_data=current_documents
    )

    if not rag_system:
        logger.error(f"Failed to initialize RAG system for {repo_full_name}")
        increment_error_count(repo_full_name)
        return

    # Define the question for the RAG system
    if is_issue_a_question(issue.title):
        # This is a direct question from a user.
        question = f"""
        A user has asked a question in a GitHub issue. Please provide a clear and direct answer based on the repository's knowledge base.

        The user's question is:
        Title: "{issue.title}"
        Body: "{issue.body}"

        Please answer the question directly. Do not analyze the issue, just provide the answer.
        For example, if the user asks "How do I run the project?", provide the steps to run the project.
        """
        response_footer = f"*Answer based on an analysis of {rag_system.get('collection_count', 'several')} repository documents.*"
        logger.info(f"Identified issue #{issue.number} as a question. Preparing a direct answer.")
    else:
        # This is likely a bug report, feature request, or other type of issue.
        question = f"""
        Analyze the following GitHub issue and provide a comprehensive analysis.
        The issue is from repository {repo_full_name}, issue number #{issue.number}.
        Title: "{issue.title}"
        Body: "{issue.body}"

        Your analysis should include:
        1. A brief summary of what the issue is about (e.g., bug report, feature request).
        2. An initial analysis or suggestions for how to approach this issue.
        3. Identification of relevant code sections or documentation from the knowledge base.
        4. A check for similar past issues in the knowledge base.
        5. Suggested potential solutions or next steps.
        """
        response_footer = f"*Analysis based on an analysis of {rag_system.get('collection_count', 'several')} repository documents.*"
        logger.info(f"Identified issue #{issue.number} as a standard issue. Preparing a detailed analysis.")

    logger.info(f"Querying RAG system for issue #{issue.number} in {repo_full_name}")

    # Query the RAG system
    try:
        result = await query_rag_system(
            qa_chain_dict=rag_system,
            query=question,
            repo_full_name=repo_full_name,
            chat_history=[],
            github_client=client
        )
        answer, usage_stats = result if isinstance(result, tuple) else (result, None)

        # Post the analysis as a comment
        await post_issue_comment(
            client,
            repo_full_name,
            issue.number,
            f"{answer}\n\n{response_footer}"
        )
        
        # Consume quota on success
        if usage_stats:
            await quota_manager.update_usage(repo_full_name, usage_stats.get('total_tokens', 0))
        reset_error_count(repo_full_name)

        logger.info(f"Posted analysis for issue #{issue.number} in {repo_full_name}")

    except Exception as e:
        logger.exception(f"Error querying RAG system for issue #{issue.number}: {e}")
        increment_error_count(repo_full_name)
        if should_send_error_response(repo_full_name, issue.number, "rag_error"):
            await post_issue_comment(
                client,
                repo_full_name,
                issue.number,
                "I encountered an error while analyzing this issue. I will try again later."
            )


async def refresh_repository_knowledge_base(repo_full_name: str, installation_id: int):
    """
    Trigger a full refresh of the repository's knowledge base.
    """
    logger.info(f"Refreshing knowledge base for {repo_full_name}")
    
    # Remove existing in-memory knowledge base
    if repo_full_name in repo_knowledge_base:
        del repo_knowledge_base[repo_full_name]
    
    # Force reinitialize with fresh content
    result = await get_or_init_repo_knowledge_base(
        repo_full_name, 
        installation_id, 
        include_current_content=False,
        force_refresh=True
    )
    
    if result:
        collection_info = await get_collection_info(result)
        logger.info(f"Successfully refreshed knowledge base for {repo_full_name}: {collection_info}")
        return True
    else:
        logger.error(f"Failed to refresh knowledge base for {repo_full_name}")
        return False

async def get_repository_collection_info(repo_full_name: str) -> dict:
    """
    Get information about a repository's ChromaDB collection.
    """
    if repo_full_name in repo_knowledge_base:
        return await get_collection_info(repo_knowledge_base[repo_full_name])
    
    # Try to get info from persistent storage
    collection_name = _sanitize_collection_name(repo_full_name)
    persist_dir = os.path.join(settings.chromadb_persist_dir, "repositories")
    
    existing_collections = await list_collections(persist_dir)
    existing_collection = next(
        (col for col in existing_collections if col["name"] == collection_name), 
        None
    )
    
    if existing_collection:
        return {
            "collection_name": collection_name,
            "document_count": existing_collection["count"],
            "status": "persisted",
            "in_memory": False
        }
    
    return {"error": f"No collection found for repository {repo_full_name}"}

async def delete_repository_collection(repo_full_name: str) -> bool:
    """
    Delete a repository's ChromaDB collection and remove from memory.
    """
    logger.info(f"Deleting collection for repository {repo_full_name}")
    
    # Remove from memory
    if repo_full_name in repo_knowledge_base:
        del repo_knowledge_base[repo_full_name]
    
    # Delete persistent collection
    collection_name = _sanitize_collection_name(repo_full_name)
    persist_dir = os.path.join(settings.chromadb_persist_dir, "repositories")
    
    success = await delete_collection(collection_name, persist_dir)
    
    if success:
        logger.info(f"Successfully deleted collection for {repo_full_name}")
    else:
        logger.error(f"Failed to delete collection for {repo_full_name}")
    
    return success

async def list_all_repository_collections() -> list:
    """
    List all repository collections in the ChromaDB instance.
    """
    persist_dir = os.path.join(settings.chromadb_persist_dir, "repositories")
    return await list_collections(persist_dir) 