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

def update_repo_activity(repo_full_name: str):
    """Update last activity timestamp for a repository."""
    repo_last_activity[repo_full_name] = datetime.utcnow()

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
    client = get_github_app_installation_client(
        settings.github_app_id, 
        settings.github_private_key, 
        installation_id
    )
    if not client:
        logger.error("Could not authenticate GitHub client for content fetching.")
        return None
    
    try:
        # Check if we have existing collection data
        existing_collections = await list_collections(persist_dir)
        existing_collection = next(
            (col for col in existing_collections if col["name"] == collection_name), 
            None
        )
        
        reset_collection = force_refresh
        repo_documents = []
        
        if existing_collection and not force_refresh:
            logger.info(f"Found existing collection '{collection_name}' with {existing_collection['count']} documents")
            # Use existing collection, only fetch current content
            if include_current_content and current_documents_data:
                repo_documents = current_documents_data
            else:
                # Create minimal placeholder for existing collection
                repo_documents = [{
                    "content": f"Repository: {repo_full_name}\nUsing existing knowledge base.",
                    "metadata": {"type": "placeholder", "repository": repo_full_name}
                }]
        else:
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
            for file in repo_content["files"]:
                repo_documents.append({
                    "content": file["content"],
                    "metadata": {
                        "type": "file",
                        "file_path": file["path"],
                        "repository": repo_full_name
                    }
                })
            
            # Add issues
            for issue in repo_content["issues"]:
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
            for pr in repo_content["pulls"]:
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
            
            reset_collection = True  # We're adding new content
        
        # Ensure we have some content to work with
        if not repo_documents:
            logger.warning(f"No documents found for repository {repo_full_name}")
            repo_documents = [{
                "content": f"Repository: {repo_full_name}\nNo content available for indexing.",
                "metadata": {"type": "placeholder", "repository": repo_full_name}
            }]
        
        # Initialize RAG system with persistent storage
        rag_result = await initialize_rag_system(
            documents_data=repo_documents,
            gemini_api_key=settings.gemini_api_key,
            chroma_persist_dir=persist_dir,
            collection_name=collection_name,
            reset_collection=reset_collection
        )
        
        if "error" in rag_result:
            logger.error(f"RAG initialization error for {repo_full_name}: {rag_result['error']}")
            return None
        
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
    client = get_github_app_installation_client(
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
    
    # Check quota before proceeding
    if not await quota_manager.check_quota(repo_full_name):
        client = get_github_app_installation_client(
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
    
    if not rag:
        logger.error(f"Could not initialize RAG system for {repo_full_name}")
        return
    
    try:
        # Query RAG system
        answer, usage_stats = await query_rag_system(rag, comment, repo_full_name)
        
        # Post response
        client = get_github_app_installation_client(
            settings.github_app_id,
            settings.github_private_key,
            installation_id
        )
        if client:
            # Add quota usage information to response
            tokens_remaining = usage_stats['tokens_remaining']
            usage_info = f"\n\n---\n*API Usage: {tokens_remaining:,} tokens remaining today.*"
            await post_issue_comment(client, repo_full_name, issue_number, answer + usage_info)
    
    except Exception as e:
        logger.exception(f"Error processing comment for {repo_full_name}")
        # Post error message
        client = get_github_app_installation_client(
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

async def handle_issue_event(payload: IssuesPayload):
    """Handle new issue events with comprehensive repository context."""
    repo_full_name = payload.repository.full_name
    issue_number = payload.issue.number
    installation_id = payload.installation.id
    issue_body = payload.issue.body or ""
    issue_title = getattr(payload.issue, 'title', '') or f"Issue #{issue_number}"
    
    logger.info(f"Handling issue event for {repo_full_name} issue #{issue_number} (action: {payload.action})")
    update_repo_activity(repo_full_name)
    
    # Only respond to newly opened issues
    if payload.action != "opened":
        logger.info(f"Ignoring issue action '{payload.action}' for issue #{issue_number}")
        return
    
    # Prepare current context documents
    current_documents = []
    if issue_body.strip():
        current_documents.append({
            "content": f"New Issue #{issue_number}: {issue_title}\n\n{issue_body}",
            "metadata": {
                "type": "new_issue", 
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
    
    if not rag:
        logger.error(f"Could not initialize RAG system for {repo_full_name}")
        # Post a fallback message
        client = await get_github_app_installation_client(
            settings.github_app_id, 
            settings.github_private_key, 
            installation_id
        )
        if client:
            fallback_message = "Welcome! I'm currently setting up the knowledge base for this repository. Please try asking questions in a few minutes."
            await post_issue_comment(client, repo_full_name, issue_number, fallback_message)
        return
    
    # Create a contextual query for the new issue
    contextual_query = f"""
    New issue opened: "{issue_title}"
    
    Issue description: {issue_body}
    
    Based on the repository's codebase, documentation, and previous issues, please provide:
    1. Initial analysis or suggestions for this issue
    2. Relevant code sections or documentation that might be related
    3. Similar past issues if any exist
    4. Potential solutions or next steps to investigate
    
    If this appears to be a bug report, question, or feature request, tailor your response accordingly.
    """
    
    # Query the RAG system
    try:
        answer = await query_rag_system(rag, contextual_query)
        
        # Add a note about the knowledge base
        collection_info = await get_collection_info(rag)
        if collection_info and "document_count" in collection_info:
            answer += f"\n\n---\n*Analysis based on {collection_info['document_count']} repository documents*"
        
    except Exception as e:
        logger.exception(f"Error querying RAG system for {repo_full_name}")
        answer = "Thank you for opening this issue! I'm currently processing the repository content and will be able to provide more helpful responses soon."
    
    # Get GitHub client for posting response
    client = await get_github_app_installation_client(
        settings.github_app_id, 
        settings.github_private_key, 
        installation_id
    )
    if not client:
        logger.error("Could not authenticate GitHub client for posting comment.")
        return
    
    # Post the response
    await post_issue_comment(client, repo_full_name, issue_number, answer)
    logger.info(f"Posted RAG-generated response to new issue {repo_full_name} #{issue_number}")

async def refresh_repository_knowledge_base(repo_full_name: str, installation_id: int):
    """
    Refresh the knowledge base for a repository by forcing a complete reindex.
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