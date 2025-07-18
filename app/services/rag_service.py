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
    post_pr_comment,
    create_pr_review,
    get_pr_files,
    get_pr_diff,
    add_pr_labels,
    fetch_all_repository_content,
    fetch_repository_files
)
from app.core.quota_manager import quota_manager
from app.config import settings
from app.models.github import IssueCommentPayload, IssuesPayload, PushPayload
from app.services.analytics_service import analytics_service
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
        "can", "could", "do", "does", "is", "are", "will", "would",
        "should", "shall", "may", "might", "must", "need", "want",
        "help", "explain", "understand", "clarify", "tell me"
    ]
    
    # Phrases that indicate questions
    question_phrases = [
        "how to", "how do", "how can", "how does", "how should",
        "what is", "what are", "what does", "what do", "what should",
        "where is", "where are", "where can", "where do",
        "why does", "why is", "why are", "why do",
        "which is", "which are", "which one", "which way",
        "can i", "can you", "can we", "can it", "can this",
        "could you", "could i", "could we", "could this",
        "is there", "are there", "is it", "are they",
        "does it", "does this", "do i", "do we", "do you",
        "should i", "should we", "should this",
        "need help", "help with", "help me", "explain",
        "run locally", "get started", "set up", "setup",
        "install", "installation", "configure", "configuration"
    ]
    
    # Check if the title starts with a question word
    starts_with_question = any(title_lower.startswith(word) for word in question_starters)
    
    # Check if the title contains question phrases
    contains_question_phrase = any(phrase in title_lower for phrase in question_phrases)
    
    # Check if the title ends with a question mark
    ends_with_question_mark = title_lower.endswith("?")
    
    # Return true if any condition is met
    return starts_with_question or contains_question_phrase or ends_with_question_mark

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
        # Fetch comprehensive repository content with enhanced code analysis
        logger.info(f"Fetching repository content for {repo_full_name} with enhanced code analysis")
        repo_content = await fetch_all_repository_content(
            client=client,
            repo_full_name=repo_full_name,
            include_issues=True,
            include_pulls=True,
            include_code=True,
            include_docs=True,
            max_items=100,
            enhanced_code_analysis=True
        )
        
        # Process enhanced documents - files are already properly structured
        repo_documents = []
        
        # Add enhanced file documents (already contain comprehensive metadata)
        for file_doc in repo_content.get("files", []):
            # Files from enhanced analysis already have proper structure
            if isinstance(file_doc, dict) and "content" in file_doc and "metadata" in file_doc:
                repo_documents.append(file_doc)
            else:
                # Fallback for any legacy structure
                repo_documents.append({
                    "content": file_doc.get("content", str(file_doc)),
                    "metadata": file_doc.get("metadata", {
                        "type": "file",
                        "repository": repo_full_name
                    })
                })
        
        # Add repository metadata
        for metadata_doc in repo_content.get("metadata", []):
            if isinstance(metadata_doc, dict) and "content" in metadata_doc:
                repo_documents.append(metadata_doc)
        
        # Add issues with enhanced structure
        for issue_doc in repo_content.get("issues", []):
            if isinstance(issue_doc, dict) and "content" in issue_doc:
                repo_documents.append(issue_doc)
        
        # Add pull requests with enhanced structure
        for pr_doc in repo_content.get("pulls", []):
            if isinstance(pr_doc, dict) and "content" in pr_doc:
                repo_documents.append(pr_doc)
        
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
    
    # Fetch content for changed files with enhanced analysis
    try:
        new_documents = await fetch_repository_files(
            client=client,
            repo_full_name=repo_full_name,
            include_code=True,
            include_docs=True,
            max_file_size=512 * 1024,
            enhanced_code_analysis=True
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
    
    # Start tracking this action
    action_log_id = await analytics_service.log_action_start(
        action_type="issue_comment",
        repo_full_name=repo_full_name,
        github_event_type="issue_comment",
        github_event_action="created",
        target_type="issue",
        target_number=issue_number,
        target_id=str(payload.comment.id),
        metadata={
            "issue_title": issue_title,
            "comment_length": len(comment),
            "issue_body_length": len(issue_content)
        }
    )
    
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
            answer, usage_stats = result
        else:
            # If we got a string, it's an error message
            answer = result
            usage_stats = None
        
        # Post response
        await post_issue_comment(client, repo_full_name, issue_number, answer)
        
        # Reset error count on successful operation
        reset_error_count(repo_full_name)
        
        # Log successful completion
        await analytics_service.log_action_complete(
            action_log_id=action_log_id,
            success=True,
            response_posted=True,
            tokens_used=usage_stats.get('total_tokens', 0) if usage_stats else None,
            metadata={
                "response_length": len(answer),
                "rag_system_available": rag is not None
            }
        )
    
    except Exception as e:
        logger.exception(f"Error processing comment for {repo_full_name}")
        
        # Log failed completion
        await analytics_service.log_action_complete(
            action_log_id=action_log_id,
            success=False,
            error_message=str(e),
            response_posted=False
        )
        
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
        # Create a comprehensive question that covers all aspects
        question = f"""
        A user has asked a question about this GitHub repository. Please provide a comprehensive and helpful answer based on the repository's content.

        User's question:
        Title: "{issue.title}"
        Body: "{issue.body or 'No additional details provided.'}"

        Please analyze the repository and provide a detailed answer that covers:

        1. If asking "what is this project" or "what does it do":
           - Project purpose and main functionality
           - Key features and capabilities
           - Target audience or use cases
           - Technology stack and architecture

        2. If asking about setup/installation ("how to run locally"):
           - Prerequisites and dependencies
           - Step-by-step installation instructions
           - Configuration requirements
           - Examples of running the project

        3. If asking about frameworks/languages:
           - Primary programming languages used
           - Frameworks and libraries
           - Development tools and build systems
           - Version requirements

        4. If asking about specific functionality:
           - Relevant code sections and files
           - Implementation details
           - Usage examples
           - API documentation if available

        5. For any other questions:
           - Search the codebase for relevant information
           - Reference specific files, functions, or documentation
           - Provide practical examples when possible

        Base your answer strictly on the actual repository content. Be comprehensive but well-organized with clear sections. If certain information isn't available in the repository, clearly state what's missing.
        """
        logger.info(f"Identified issue #{issue.number} as a question. Preparing a comprehensive answer.")
    else:
        # This is likely a bug report, feature request, or other type of issue.
        question = f"""
        Analyze the following GitHub issue and provide a comprehensive analysis based on the repository's codebase and documentation.
        
        Repository: {repo_full_name}
        Issue #{issue.number}: "{issue.title}"
        Body: "{issue.body or 'No details provided.'}"

        Your analysis should include:
        1. **Issue Classification**: Determine if this is a bug report, feature request, question, or other type of issue
        2. **Code Analysis**: Search the repository for related code sections, files, or functionality
        3. **Context**: Find relevant documentation, similar past issues, or related discussions
        4. **Technical Assessment**: 
           - If it's a bug: identify potential causes and affected components
           - If it's a feature request: assess feasibility and suggest implementation approaches
           - If it's a question: provide a comprehensive answer with examples
        5. **Recommendations**: Suggest next steps, solutions, or additional information needed

        Search the entire repository including code files, documentation, configuration files, and past issues to provide the most helpful response.
        """
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

        # For questions, provide just the answer without duplicate analysis
        if is_issue_a_question(issue.title):
            # Post the direct answer to the question
            await post_issue_comment(
                client,
                repo_full_name,
                issue.number,
                answer
            )
        else:
            # For non-questions, we can include similarity analysis
            try:
                # Import here to avoid circular imports
                from app.services.issue_similarity_service import IssueSimilarityService
                
                similarity_service = IssueSimilarityService()
                similarity_result = await similarity_service.analyze_issue_similarity(
                    new_issue={
                        "number": issue.number,
                        "title": issue.title,
                        "body": issue.body or "",
                        "state": "open",
                        "created_at": datetime.utcnow().isoformat(),
                        "labels": []
                    },
                    repo_full_name=repo_full_name,
                    installation_id=installation_id,
                    rag_system=rag_system
                )
                
                # Combine AI analysis with similarity analysis for non-questions
                combined_response = f"{answer}\n\n---\n\n🤖 **Automated Issue Analysis**\n\n"
                
                if similarity_result.similar_issues:
                    combined_response += "🔍 **Similar Issues Found:**\n\n"
                    for similar in similarity_result.similar_issues[:3]:  # Show top 3
                        combined_response += f"- Similar to #{similar.issue_number}: *{similar.title}* (similarity: {similar.similarity_score:.1%})\n"
                    combined_response += "\n"
                
                if similarity_result.suggestions:
                    combined_response += "💡 **Suggestions:**\n\n"
                    for suggestion in similarity_result.suggestions:
                        combined_response += f"- {suggestion}\n"
                
                combined_response += "\nThis analysis was performed automatically. Please review and take appropriate action."
                
                await post_issue_comment(
                    client,
                    repo_full_name,
                    issue.number,
                    combined_response
                )
                
            except Exception as similarity_error:
                logger.warning(f"Similarity analysis failed for issue #{issue.number}: {similarity_error}")
                # Fallback to just posting the AI analysis
                await post_issue_comment(
                    client,
                    repo_full_name,
                    issue.number,
                    answer
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

# New PR event handlers

async def handle_pr_opened_event(payload):
    """Handle pull request opened events with PR summary only (analysis commented out)."""
    pr_data = payload.pull_request
    repo_full_name = payload.repository.full_name
    installation_id = payload.installation.id
    pr_number = pr_data.number
    
    logger.info(f"Handling PR opened event for #{pr_number} in {repo_full_name}")
    
    # Start tracking this action
    action_log_id = await analytics_service.log_action_start(
        action_type="pr_summary",
        action_subtype="opened",
        repo_full_name=repo_full_name,
        github_event_type="pull_request",
        github_event_action="opened",
        target_type="pr",
        target_number=pr_number,
        target_id=str(pr_data.id),
        metadata={
            "pr_title": pr_data.title,
            "pr_body_length": len(pr_data.body or ""),
            "files_changed": pr_data.changed_files,
            "additions": pr_data.additions,
            "deletions": pr_data.deletions
        }
    )
    
    # --- PR Analysis and label logic commented out ---
    # from app.services.pr_analysis_service import pr_analysis_service
    # from app.services.issue_similarity_service import issue_similarity_service
    # ... (all analysis and label logic)
    
    try:
        # Get GitHub client first
        client = await get_github_app_installation_client(
            settings.github_app_id,
            settings.github_private_key,
            installation_id
        )
        
        if not client:
            logger.error(f"Could not authenticate GitHub client for {repo_full_name}")
            return
        
        # Get or initialize repository knowledge base with error handling
        rag_system = None
        try:
            rag_system = await get_or_init_repo_knowledge_base(
                repo_full_name=repo_full_name,
                installation_id=installation_id,
                include_current_content=False
            )
            if rag_system and "error" not in rag_system:
                logger.debug(f"Successfully initialized RAG system for {repo_full_name}")
            else:
                logger.warning(f"RAG system initialization failed for {repo_full_name}, will fallback to basic summary")
                rag_system = None
        except Exception as e:
            logger.warning(f"RAG system initialization error for {repo_full_name}: {e}, proceeding with basic summary")
            rag_system = None

        # Gather PR context
        pr_title = pr_data.title or "(No title)"
        pr_body = pr_data.body or "(No description provided)"
        pr_files = []
        try:
            pr_files = await get_pr_files(client, repo_full_name, pr_number)
        except Exception as e:
            logger.warning(f"Failed to get PR files for summary: {e}")
        changed_files_list = [f.get('filename', f) for f in pr_files] if pr_files else []
        changed_files_str = '\n'.join(f'- {f}' for f in changed_files_list) if changed_files_list else 'No files listed.'

        # Generate PR summary
        summary = None
        if rag_system:
            # Use RAG to generate summary
            summary_prompt = (
                f"Summarize what this pull request does, based on its title, description, and the files it changes, "
                f"in the context of the repository.\n\n"
                f"Title: {pr_title}\n"
                f"Description: {pr_body}\n"
                f"Changed files:\n{changed_files_str}"
            )
            try:
                result = await query_rag_system(
                    rag_system,
                    summary_prompt,
                    repo_full_name,
                    chat_history=[],
                    github_client=client
                )
                summary = result[0] if isinstance(result, tuple) else result
            except Exception as e:
                logger.warning(f"RAG summary generation failed: {e}")
                summary = None
        
        if not summary:
            # Fallback: basic summary
            summary = (
                f"### 📝 PR Summary\n\n"
                f"**Title:** {pr_title}\n\n"
                f"**Description:** {pr_body}\n\n"
                f"**Files changed:**\n{changed_files_str}\n"
            )
        
        # Post the summary as a comment
        await post_pr_comment(client, repo_full_name, pr_number, summary)
        
        # Log successful completion
        await analytics_service.log_action_complete(
            action_log_id=action_log_id,
            success=True,
            response_posted=True,
            metadata={
                "summary_posted": True
            }
        )
        
    except Exception as e:
        logger.exception(f"Unexpected error handling PR opened event for #{pr_number}: {e}")
        await analytics_service.log_action_complete(
            action_log_id=action_log_id,
            success=False,
            error_message=str(e),
            response_posted=False
        )
        
        # Final fallback: try to post a simple acknowledgment comment
        if client:
            try:
                simple_comment = (
                    f"👋 Thank you for your pull request #{pr_number}!\n\n"
                    "I encountered some issues during analysis, but the PR has been received. "
                    "A human reviewer will take a look soon."
                )
                await post_pr_comment(client, repo_full_name, pr_number, simple_comment)
                logger.info(f"Posted fallback comment for PR #{pr_number}")
            except Exception as fallback_error:
                logger.error(f"Even fallback comment failed for PR #{pr_number}: {fallback_error}")

async def handle_pr_updated_event(payload):
    """Handle pull request updated events (synchronize)."""
    try:
        from app.services.pr_analysis_service import pr_analysis_service
        
        pr_data = payload.pull_request
        repo_full_name = payload.repository.full_name
        installation_id = payload.installation.id
        pr_number = pr_data.number
        
        logger.info(f"Handling PR updated event for #{pr_number} in {repo_full_name}")
        
        # Only re-analyze if there are new commits
        if not hasattr(payload, 'before') or not hasattr(payload, 'after'):
            logger.info("No new commits detected, skipping re-analysis")
            return
        
        # Get repository knowledge base
        rag_system = await get_or_init_repo_knowledge_base(
            repo_full_name=repo_full_name,
            installation_id=installation_id,
            include_current_content=False
        )
        
        if not rag_system or "error" in rag_system:
            logger.warning(f"Could not get RAG system for {repo_full_name}")
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
        
        # Get updated PR files
        pr_files = await get_pr_files(client, repo_full_name, pr_number)
        
        # Re-analyze only if significant changes
        if len(pr_files) > 0:
            pr_analysis = await pr_analysis_service.analyze_pull_request(
                pr_data=pr_data.__dict__,
                pr_files=pr_files,
                repo_full_name=repo_full_name,
                installation_id=installation_id,
                rag_system=rag_system
            )
            
            # Post update comment if there are new issues
            if pr_analysis.has_issues:
                await _post_pr_update_comment(
                    client, repo_full_name, pr_number, pr_analysis
                )
        
        logger.info(f"Completed update analysis for PR #{pr_number}")
        
    except Exception as e:
        logger.exception(f"Failed to handle PR updated event: {e}")

async def handle_pr_closed_event(payload):
    """Handle pull request closed events."""
    try:
        pr_data = payload.pull_request
        repo_full_name = payload.repository.full_name
        installation_id = payload.installation.id
        pr_number = pr_data.number
        
        logger.info(f"Handling PR closed event for #{pr_number} in {repo_full_name}")
        
        # If PR was merged to main branch, trigger knowledge base refresh
        if pr_data.merged and pr_data.base.ref in ["main", "master"]:
            logger.info(f"PR #{pr_number} was merged to {pr_data.base.ref}, scheduling knowledge base refresh")
            
            # Schedule refresh after a delay to allow changes to propagate
            await schedule_reindex(repo_full_name, installation_id, delay_minutes=5)
        
    except Exception as e:
        logger.exception(f"Failed to handle PR closed event: {e}")

async def handle_pr_review_submitted_event(payload):
    """Handle PR review submitted events."""
    try:
        review = payload.review
        pr_data = payload.pull_request
        repo_full_name = payload.repository.full_name
        pr_number = pr_data.number
        
        logger.info(f"Handling PR review submitted for #{pr_number} in {repo_full_name}")
        
        # For now, just log the review. Could be extended to:
        # - Analyze review comments for patterns
        # - Update PR analysis based on human feedback
        # - Learn from reviewer suggestions
        
        logger.info(f"Review by {review.user.login}: {review.state}")
        
    except Exception as e:
        logger.exception(f"Failed to handle PR review submitted event: {e}")

async def _post_fallback_pr_comment(client, repo_full_name: str, pr_number: int, file_count: int):
    """Post a basic welcome comment when full analysis isn't available."""
    try:
        comment_parts = [
            f"🤖 **PR Analysis Bot**\n",
            f"👋 Thank you for your pull request #{pr_number}!\n",
            f"📊 **Basic Stats:**",
            f"- Files changed: {file_count}",
            f"- Status: Under review\n",
            "⚠️ *Detailed analysis is temporarily unavailable, but your PR has been registered.*\n",
            "---",
            "*A human reviewer will provide feedback soon.*"
        ]
        
        comment = "\n".join(comment_parts)
        success = await post_pr_comment(client, repo_full_name, pr_number, comment)
        
        if success:
            logger.info(f"Posted fallback comment on PR #{pr_number}")
        return success
        
    except Exception as e:
        logger.error(f"Failed to post fallback PR comment: {e}")
        return False

async def _add_basic_pr_labels(client, repo_full_name: str, pr_number: int):
    """Add basic labels when full analysis isn't available."""
    try:
        basic_labels = ["needs-review"]
        success = await add_pr_labels(client, repo_full_name, pr_number, basic_labels)
        if success:
            logger.info(f"Added basic labels {basic_labels} to PR #{pr_number}")
        return success
        
    except Exception as e:
        logger.error(f"Failed to add basic labels to PR #{pr_number}: {e}")
        return False

async def _post_pr_analysis_comment(
    client,
    repo_full_name: str,
    pr_number: int,
    pr_analysis,
    duplicate_check
):
    """Post comprehensive analysis comment on PR."""
    try:
        comment_parts = ["🤖 **Automated PR Analysis**\n"]
        
        # Overall assessment
        if pr_analysis.overall_score >= 80:
            comment_parts.append(f"✅ **Overall Quality: Excellent** (Score: {pr_analysis.overall_score}/100)")
        elif pr_analysis.overall_score >= 60:
            comment_parts.append(f"⚠️ **Overall Quality: Good** (Score: {pr_analysis.overall_score}/100)")
        else:
            comment_parts.append(f"❌ **Overall Quality: Needs Improvement** (Score: {pr_analysis.overall_score}/100)")
        
        comment_parts.append(f"**Review Priority:** {pr_analysis.review_priority.upper()}\n")
        
        # Security issues
        if pr_analysis.security_issues:
            comment_parts.append("🔒 **Security Issues Found:**")
            for issue in pr_analysis.security_issues[:5]:  # Show top 5
                severity_emoji = {"critical": "🚨", "high": "⚠️", "medium": "📝", "low": "ℹ️"}.get(issue["severity"], "📝")
                comment_parts.append(f"- {severity_emoji} {issue['description']} (Line {issue.get('line', '?')})")
            comment_parts.append("")
        
        # Potential bugs
        if pr_analysis.potential_bugs:
            comment_parts.append("🐛 **Potential Bugs:**")
            for bug in pr_analysis.potential_bugs[:5]:  # Show top 5
                severity_emoji = {"critical": "🚨", "high": "⚠️", "medium": "📝", "low": "ℹ️"}.get(bug["severity"], "📝")
                comment_parts.append(f"- {severity_emoji} {bug['description']} (Line {bug.get('line', '?')})")
            comment_parts.append("")
        
        # Complexity issues
        if pr_analysis.complexity_issues:
            comment_parts.append("🔄 **Complexity Issues:**")
            for issue in pr_analysis.complexity_issues[:3]:  # Show top 3
                comment_parts.append(f"- {issue['description']} (Score: {issue.get('complexity_score', '?')})")
            comment_parts.append("")
        
        # Duplicate functionality
        if duplicate_check.has_duplicates:
            comment_parts.append("♻️ **Duplicate Functionality Detected:**")
            for similar in duplicate_check.similar_issues[:3]:
                comment_parts.append(f"- Similar to #{similar.issue_number}: {similar.title} (Similarity: {similar.similarity_score:.1%})")
            comment_parts.append("")
        
        # Suggestions
        if pr_analysis.suggestions:
            comment_parts.append("💡 **Recommendations:**")
            for suggestion in pr_analysis.suggestions:
                comment_parts.append(f"- {suggestion}")
            comment_parts.append("")
        
        # Additional suggestions from duplicate check
        if duplicate_check.suggestions:
            for suggestion in duplicate_check.suggestions:
                comment_parts.append(f"- {suggestion}")
        
        comment_parts.append("\n---")
        comment_parts.append("*This analysis was performed automatically using AI. Please review and address the findings before merging.*")
        
        comment = "\n".join(comment_parts)
        
        # Post the comment
        success = await post_pr_comment(client, repo_full_name, pr_number, comment)
        
        if success:
            logger.info(f"Posted analysis comment on PR #{pr_number}")
        else:
            logger.error(f"Failed to post analysis comment on PR #{pr_number}")
            
    except Exception as e:
        logger.exception(f"Failed to post PR analysis comment: {e}")

async def _post_pr_update_comment(
    client,
    repo_full_name: str,
    pr_number: int,
    pr_analysis
):
    """Post update comment on PR after changes."""
    try:
        comment_parts = ["🔄 **Updated PR Analysis**\n"]
        
        # Focus on new/critical issues
        critical_issues = [
            issue for issue in pr_analysis.security_issues 
            if issue.get("severity") in ["critical", "high"]
        ]
        
        critical_bugs = [
            bug for bug in pr_analysis.potential_bugs
            if bug.get("severity") in ["critical", "high"]
        ]
        
        if critical_issues or critical_bugs:
            comment_parts.append("⚠️ **Critical Issues Found in Latest Changes:**")
            
            for issue in critical_issues[:3]:
                comment_parts.append(f"- 🔒 {issue['description']}")
                
            for bug in critical_bugs[:3]:
                comment_parts.append(f"- 🐛 {bug['description']}")
                
            comment_parts.append("\nPlease address these issues before proceeding.")
        else:
            comment_parts.append("✅ **No critical issues found in latest changes.**")
            
        comment_parts.append(f"\n**Updated Quality Score:** {pr_analysis.overall_score}/100")
        comment_parts.append("\n---")
        comment_parts.append("*Updated analysis based on recent changes.*")
        
        comment = "\n".join(comment_parts)
        
        success = await post_pr_comment(client, repo_full_name, pr_number, comment)
        
        if success:
            logger.info(f"Posted update comment on PR #{pr_number}")
            
    except Exception as e:
        logger.exception(f"Failed to post PR update comment: {e}")

async def _add_analysis_based_labels(
    client,
    repo_full_name: str,
    pr_number: int,
    pr_analysis
):
    """Add labels to PR based on analysis results."""
    try:
        labels_to_add = []
        
        # Priority labels
        if pr_analysis.review_priority == "critical":
            labels_to_add.append("priority: critical")
        elif pr_analysis.review_priority == "high":
            labels_to_add.append("priority: high")
        
        # Issue type labels
        if pr_analysis.security_issues:
            labels_to_add.append("security")
            
        if pr_analysis.potential_bugs:
            labels_to_add.append("bug-risk")
            
        if pr_analysis.complexity_issues:
            labels_to_add.append("complexity")
            
        if pr_analysis.duplicate_functionality:
            labels_to_add.append("duplicate")
            
        # Quality labels
        if pr_analysis.overall_score >= 80:
            labels_to_add.append("quality: good")
        elif pr_analysis.overall_score < 50:
            labels_to_add.append("quality: needs-work")
        
        # Add labels if any
        if labels_to_add:
            success = await add_pr_labels(client, repo_full_name, pr_number, labels_to_add)
            if success:
                logger.info(f"Added labels {labels_to_add} to PR #{pr_number}")
                
    except Exception as e:
        logger.exception(f"Failed to add labels to PR #{pr_number}: {e}")

# Enhanced issue event handlers

async def handle_issue_event_enhanced(payload):
    """Enhanced issue event handler with similarity detection."""
    try:
        from app.services.issue_similarity_service import issue_similarity_service
        
        issue = payload.issue
        action = payload.action
        repo_full_name = payload.repository.full_name
        installation_id = payload.installation.id
        
        logger.info(f"Handling enhanced issue event: {action} for #{issue.number} in {repo_full_name}")
        
        if action == "opened":
            # Get repository knowledge base
            rag_system = await get_or_init_repo_knowledge_base(
                repo_full_name=repo_full_name,
                installation_id=installation_id,
                include_current_content=False
            )
            
            # Analyze issue similarity
            similarity_result = await issue_similarity_service.analyze_issue_similarity(
                new_issue={
                    "title": issue.title,
                    "body": issue.body or "",
                    "number": issue.number
                },
                repo_full_name=repo_full_name,
                installation_id=installation_id,
                rag_system=rag_system
            )
            
            # Auto-comment if significant findings
            if similarity_result.has_duplicates or similarity_result.is_likely_invalid:
                await issue_similarity_service.auto_comment_on_similar_issues(
                    issue_number=issue.number,
                    repo_full_name=repo_full_name,
                    installation_id=installation_id,
                    similarity_result=similarity_result
                )
        
        # Fall back to original issue handler for other functionality
        await handle_issue_event(payload)
        
    except Exception as e:
        logger.exception(f"Failed to handle enhanced issue event: {e}")
        # Fall back to original handler
        await handle_issue_event(payload) 