"""
Issue Similarity Service

Provides comprehensive issue similarity detection and management including:
- Semantic similarity detection using embeddings
- Duplicate issue identification
- Issue clustering and grouping
- Automatic issue consolidation suggestions
- Invalid issue detection based on code analysis
"""

import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.core.github_utils import (
    get_github_app_installation_client,
    post_issue_comment,
    fetch_repository_issues
)
from app.core.rag_system import query_rag_system
from app.config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger("issue_similarity_service")

@dataclass
class SimilarIssue:
    """Represents a similar issue."""
    issue_number: int
    title: str
    body: str
    state: str
    similarity_score: float
    similarity_type: str  # "semantic", "keyword", "pattern"
    created_at: str
    labels: List[str]

@dataclass
class IssueCluster:
    """Represents a cluster of similar issues."""
    primary_issue: int
    similar_issues: List[SimilarIssue]
    cluster_type: str  # "duplicate", "related", "follow_up"
    confidence_score: float
    suggested_action: str
    merge_suggestion: Optional[str]

@dataclass
class IssueSimilarityResult:
    """Result of issue similarity analysis."""
    has_duplicates: bool
    similar_issues: List[SimilarIssue]
    clusters: List[IssueCluster]
    is_likely_invalid: bool
    invalid_reasons: List[str]
    suggestions: List[str]

class IssueSimilarityService:
    """Service for detecting and managing similar issues."""
    
    def __init__(self):
        self.logger = logging.getLogger("issue_similarity_service")
        self.embeddings = None
        self.tfidf_vectorizer = None
        
    async def initialize_embeddings(self):
        """Initialize embeddings model for semantic similarity."""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=settings.gemini_api_key,
                model="models/embedding-001",
                task_type="retrieval_query"
            )
            self.logger.info("Embeddings model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings: {e}")
            self.embeddings = None
    
    async def analyze_issue_similarity(
        self,
        new_issue: Dict[str, Any],
        repo_full_name: str,
        installation_id: int,
        rag_system: Optional[Dict] = None
    ) -> IssueSimilarityResult:
        """
        Analyze a new issue for similarity to existing issues.
        
        Args:
            new_issue: New issue data from GitHub API
            repo_full_name: Repository full name
            installation_id: GitHub installation ID
            rag_system: Optional RAG system for context analysis
            
        Returns:
            IssueSimilarityResult with comprehensive similarity analysis
        """
        current_issue_number = new_issue.get('number')
        self.logger.info(f"Analyzing similarity for issue #{current_issue_number} in {repo_full_name}")
        
        # Initialize embeddings if needed
        if not self.embeddings:
            await self.initialize_embeddings()
        
        # Fetch existing issues
        client = await get_github_app_installation_client(
            settings.github_app_id,
            settings.github_private_key,
            installation_id
        )
        
        if not client:
            self.logger.error("Could not authenticate GitHub client")
            return IssueSimilarityResult(
                has_duplicates=False,
                similar_issues=[],
                clusters=[],
                is_likely_invalid=False,
                invalid_reasons=[],
                suggestions=["Unable to analyze - authentication failed"]
            )
        
        # Get recent issues for comparison, excluding the current issue
        existing_issues = await self._fetch_recent_issues(
            client, 
            repo_full_name, 
            exclude_issue_number=current_issue_number
        )
        
        # If no other issues exist, return empty result
        if not existing_issues:
            self.logger.info(f"No existing issues found for comparison (excluding #{current_issue_number})")
            return IssueSimilarityResult(
                has_duplicates=False,
                similar_issues=[],
                clusters=[],
                is_likely_invalid=False,
                invalid_reasons=[],
                suggestions=["✅ **First Issue**: No existing issues found for comparison. This appears to be a new issue."]
            )
        
        # Perform different types of similarity analysis
        semantic_similar = await self._find_semantic_similarity(new_issue, existing_issues)
        keyword_similar = await self._find_keyword_similarity(new_issue, existing_issues)
        pattern_similar = await self._find_pattern_similarity(new_issue, existing_issues)
        
        # Combine results
        all_similar = self._combine_similarity_results(
            semantic_similar, keyword_similar, pattern_similar
        )
        
        # Additional safety check: filter out any self-references that might have slipped through
        all_similar = [
            issue for issue in all_similar 
            if issue.issue_number != current_issue_number
        ]
        
        # Create clusters
        clusters = self._create_issue_clusters(new_issue, all_similar)
        
        # Check if issue is invalid
        is_invalid, invalid_reasons = await self._check_if_invalid_issue(
            new_issue, repo_full_name, rag_system
        )
        
        # Generate suggestions
        suggestions = self._generate_similarity_suggestions(
            all_similar, clusters, is_invalid, invalid_reasons
        )
        
        has_duplicates = any(
            issue.similarity_score > 0.8 for issue in all_similar
        )
        
        result = IssueSimilarityResult(
            has_duplicates=has_duplicates,
            similar_issues=all_similar,
            clusters=clusters,
            is_likely_invalid=is_invalid,
            invalid_reasons=invalid_reasons,
            suggestions=suggestions
        )
        
        self.logger.info(f"Similarity analysis completed. Found {len(all_similar)} similar issues")
        return result
    
    async def _fetch_recent_issues(
        self,
        client,
        repo_full_name: str,
        max_issues: int = 200,
        exclude_issue_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fetch recent issues for comparison, excluding the specified issue number."""
        try:
            issues_data = await fetch_repository_issues(
                client=client,
                repo_full_name=repo_full_name,
                state="all",
                max_issues=max_issues,
                include_comments=False  # Don't need comments for similarity
            )
            
            # Convert to simple dict format and exclude the current issue
            issues = []
            for issue_doc in issues_data:
                metadata = issue_doc.get("metadata", {})
                if metadata.get("type") == "issue":
                    issue_number = metadata.get("issue_number")
                    
                    # Skip the current issue to prevent self-comparison
                    if exclude_issue_number and issue_number == exclude_issue_number:
                        continue
                        
                    issues.append({
                        "number": issue_number,
                        "title": metadata.get("title", ""),
                        "body": issue_doc.get("content", "").split("\n\n", 1)[-1] if "\n\n" in issue_doc.get("content", "") else "",
                        "state": metadata.get("state", ""),
                        "created_at": metadata.get("created_at", ""),
                        "labels": metadata.get("labels", [])
                    })
            
            self.logger.info(f"Fetched {len(issues)} recent issues for comparison (excluded issue #{exclude_issue_number})")
            return issues
            
        except Exception as e:
            self.logger.error(f"Failed to fetch recent issues: {e}")
            return []
    
    async def _find_semantic_similarity(
        self,
        new_issue: Dict[str, Any],
        existing_issues: List[Dict[str, Any]]
    ) -> List[SimilarIssue]:
        """Find semantically similar issues using embeddings."""
        similar_issues = []
        
        if not self.embeddings or not existing_issues:
            return similar_issues
        
        current_issue_number = new_issue.get("number")
        
        try:
            # Prepare text for new issue
            new_text = self._prepare_issue_text(new_issue)
            
            # Prepare texts for existing issues, excluding self
            existing_texts = [
                self._prepare_issue_text(issue) 
                for issue in existing_issues 
                if issue.get("number") != current_issue_number
            ]
            
            # Filter existing issues to exclude self
            filtered_existing_issues = [
                issue for issue in existing_issues 
                if issue.get("number") != current_issue_number
            ]
            
            if not filtered_existing_issues:
                return similar_issues
            
            # Generate embeddings
            new_embedding = await self.embeddings.aembed_query(new_text)
            existing_embeddings = await self.embeddings.aembed_documents(existing_texts)
            
            # Calculate similarities
            new_embedding_array = np.array(new_embedding).reshape(1, -1)
            existing_embeddings_array = np.array(existing_embeddings)
            
            similarities = cosine_similarity(new_embedding_array, existing_embeddings_array)[0]
            
            # Find similar issues (threshold > 0.7 for semantic similarity)
            for i, similarity in enumerate(similarities):
                if similarity > 0.7:
                    issue = filtered_existing_issues[i]
                    # Double check to ensure we're not comparing against self
                    if issue.get("number") != current_issue_number:
                        similar_issues.append(SimilarIssue(
                            issue_number=issue["number"],
                            title=issue["title"],
                            body=issue["body"],
                            state=issue["state"],
                            similarity_score=float(similarity),
                            similarity_type="semantic",
                            created_at=issue["created_at"],
                            labels=issue.get("labels", [])
                        ))
            
            # Sort by similarity score
            similar_issues.sort(key=lambda x: x.similarity_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Semantic similarity analysis failed: {e}")
        
        return similar_issues
    
    async def _find_keyword_similarity(
        self,
        new_issue: Dict[str, Any],
        existing_issues: List[Dict[str, Any]]
    ) -> List[SimilarIssue]:
        """Find similar issues using keyword/TF-IDF similarity."""
        similar_issues = []
        
        if not existing_issues:
            return similar_issues
        
        current_issue_number = new_issue.get("number")
        
        try:
            # Filter existing issues to exclude self
            filtered_existing_issues = [
                issue for issue in existing_issues 
                if issue.get("number") != current_issue_number
            ]
            
            if not filtered_existing_issues:
                return similar_issues
            
            # Prepare texts
            new_text = self._prepare_issue_text(new_issue)
            existing_texts = [self._prepare_issue_text(issue) for issue in filtered_existing_issues]
            
            # Create TF-IDF vectors
            all_texts = [new_text] + existing_texts
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Calculate similarities
            new_vector = tfidf_matrix[0]
            existing_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(new_vector, existing_vectors)[0]
            
            # Find similar issues (threshold > 0.5 for keyword similarity)
            for i, similarity in enumerate(similarities):
                if similarity > 0.5:
                    issue = filtered_existing_issues[i]
                    # Double check to ensure we're not comparing against self
                    if issue.get("number") != current_issue_number:
                        similar_issues.append(SimilarIssue(
                            issue_number=issue["number"],
                            title=issue["title"],
                            body=issue["body"],
                            state=issue["state"],
                            similarity_score=float(similarity),
                            similarity_type="keyword",
                            created_at=issue["created_at"],
                            labels=issue.get("labels", [])
                        ))
            
            # Sort by similarity score
            similar_issues.sort(key=lambda x: x.similarity_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Keyword similarity analysis failed: {e}")
        
        return similar_issues
    
    async def _find_pattern_similarity(
        self,
        new_issue: Dict[str, Any],
        existing_issues: List[Dict[str, Any]]
    ) -> List[SimilarIssue]:
        """Find similar issues using pattern matching."""
        similar_issues = []
        
        current_issue_number = new_issue.get("number")
        new_title = new_issue.get("title", "").lower()
        new_body = new_issue.get("body", "").lower()
        
        # Extract key patterns from new issue
        new_patterns = self._extract_issue_patterns(new_title, new_body)
        
        for issue in existing_issues:
            # Skip self-comparison
            if issue.get("number") == current_issue_number:
                continue
                
            title = issue.get("title", "").lower()
            body = issue.get("body", "").lower()
            
            # Extract patterns from existing issue
            existing_patterns = self._extract_issue_patterns(title, body)
            
            # Calculate pattern similarity
            pattern_score = self._calculate_pattern_similarity(new_patterns, existing_patterns)
            
            if pattern_score > 0.6:  # Threshold for pattern similarity
                similar_issues.append(SimilarIssue(
                    issue_number=issue["number"],
                    title=issue["title"],
                    body=issue["body"],
                    state=issue["state"],
                    similarity_score=pattern_score,
                    similarity_type="pattern",
                    created_at=issue["created_at"],
                    labels=issue.get("labels", [])
                ))
        
        # Sort by similarity score
        similar_issues.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return similar_issues
    
    def _prepare_issue_text(self, issue: Dict[str, Any]) -> str:
        """Prepare issue text for similarity analysis."""
        title = issue.get("title", "")
        body = issue.get("body", "")
        
        # Clean and combine title and body
        text = f"{title}\n\n{body}"
        
        # Remove code blocks and formatting
        text = re.sub(r'```[\s\S]*?```', '', text)  # Remove code blocks
        text = re.sub(r'`[^`]*`', '', text)  # Remove inline code
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Remove images
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove links
        text = re.sub(r'#+\s*', '', text)  # Remove headers
        text = re.sub(r'\*+', '', text)  # Remove emphasis
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        return text.strip()
    
    def _extract_issue_patterns(self, title: str, body: str) -> Dict[str, Any]:
        """Extract key patterns from issue text."""
        patterns = {
            "error_messages": [],
            "stack_traces": [],
            "file_names": [],
            "function_names": [],
            "keywords": [],
            "numbers": [],
            "urls": []
        }
        
        text = f"{title} {body}"
        
        # Extract error messages
        error_patterns = [
            r'error[:\s]+([^\n.!?]+)',
            r'exception[:\s]+([^\n.!?]+)',
            r'failed[:\s]+([^\n.!?]+)'
        ]
        for pattern in error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            patterns["error_messages"].extend(matches)
        
        # Extract file names
        file_patterns = [
            r'([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z]{2,4})',  # file.ext
            r'([a-zA-Z0-9_/\\-]+\.[a-zA-Z]{2,4})'  # path/to/file.ext
        ]
        for pattern in file_patterns:
            matches = re.findall(pattern, text)
            patterns["file_names"].extend(matches)
        
        # Extract function names
        function_patterns = [
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # function()
            r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # def function
            r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)'  # function name
        ]
        for pattern in function_patterns:
            matches = re.findall(pattern, text)
            patterns["function_names"].extend(matches)
        
        # Extract version numbers
        version_pattern = r'v?(\d+\.\d+(?:\.\d+)?(?:\.\d+)?)'
        patterns["numbers"] = re.findall(version_pattern, text)
        
        # Extract URLs
        url_pattern = r'https?://[^\s]+'
        patterns["urls"] = re.findall(url_pattern, text)
        
        # Extract technical keywords
        tech_keywords = [
            'bug', 'error', 'crash', 'fail', 'broken', 'issue', 'problem',
            'feature', 'enhancement', 'improvement', 'request',
            'install', 'setup', 'config', 'configuration',
            'performance', 'slow', 'fast', 'optimize',
            'security', 'vulnerability', 'exploit',
            'api', 'database', 'server', 'client', 'frontend', 'backend'
        ]
        
        for keyword in tech_keywords:
            if keyword in text.lower():
                patterns["keywords"].append(keyword)
        
        return patterns
    
    def _calculate_pattern_similarity(
        self,
        patterns1: Dict[str, Any],
        patterns2: Dict[str, Any]
    ) -> float:
        """Calculate similarity score based on pattern matching."""
        total_score = 0.0
        total_weight = 0.0
        
        # Weight different pattern types
        weights = {
            "error_messages": 0.4,
            "file_names": 0.3,
            "function_names": 0.2,
            "keywords": 0.1
        }
        
        for pattern_type, weight in weights.items():
            list1 = patterns1.get(pattern_type, [])
            list2 = patterns2.get(pattern_type, [])
            
            if not list1 and not list2:
                continue
                
            # Calculate Jaccard similarity for this pattern type
            set1 = set(str(item).lower() for item in list1)
            set2 = set(str(item).lower() for item in list2)
            
            if set1 or set2:
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                jaccard_score = intersection / union if union > 0 else 0.0
                
                total_score += jaccard_score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _combine_similarity_results(
        self,
        semantic_similar: List[SimilarIssue],
        keyword_similar: List[SimilarIssue],
        pattern_similar: List[SimilarIssue]
    ) -> List[SimilarIssue]:
        """Combine results from different similarity methods."""
        # Create a map to track all similar issues
        issue_map = {}
        
        # Add all similar issues, keeping the highest score for each
        for similar_list in [semantic_similar, keyword_similar, pattern_similar]:
            for issue in similar_list:
                issue_num = issue.issue_number
                if issue_num not in issue_map or issue.similarity_score > issue_map[issue_num].similarity_score:
                    issue_map[issue_num] = issue
        
        # Convert back to list and sort by score
        combined_similar = list(issue_map.values())
        combined_similar.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Limit to top 10 most similar
        return combined_similar[:10]
    
    def _create_issue_clusters(
        self,
        new_issue: Dict[str, Any],
        similar_issues: List[SimilarIssue]
    ) -> List[IssueCluster]:
        """Create clusters of related issues."""
        clusters = []
        
        if not similar_issues:
            return clusters
        
        # Group by similarity score ranges
        duplicates = [issue for issue in similar_issues if issue.similarity_score > 0.85]
        related = [issue for issue in similar_issues if 0.7 <= issue.similarity_score <= 0.85]
        follow_ups = [issue for issue in similar_issues if 0.5 <= issue.similarity_score < 0.7]
        
        # Create duplicate cluster
        if duplicates:
            clusters.append(IssueCluster(
                primary_issue=new_issue.get("number"),
                similar_issues=duplicates,
                cluster_type="duplicate",
                confidence_score=max(issue.similarity_score for issue in duplicates),
                suggested_action="close_as_duplicate",
                merge_suggestion=f"This issue appears to be a duplicate of #{duplicates[0].issue_number}"
            ))
        
        # Create related cluster
        if related:
            clusters.append(IssueCluster(
                primary_issue=new_issue.get("number"),
                similar_issues=related,
                cluster_type="related",
                confidence_score=max(issue.similarity_score for issue in related),
                suggested_action="link_issues",
                merge_suggestion=f"This issue is related to #{related[0].issue_number}"
            ))
        
        # Create follow-up cluster
        if follow_ups:
            clusters.append(IssueCluster(
                primary_issue=new_issue.get("number"),
                similar_issues=follow_ups,
                cluster_type="follow_up",
                confidence_score=max(issue.similarity_score for issue in follow_ups),
                suggested_action="reference_issues",
                merge_suggestion=f"This might be a follow-up to #{follow_ups[0].issue_number}"
            ))
        
        return clusters
    
    async def _check_if_invalid_issue(
        self,
        issue: Dict[str, Any],
        repo_full_name: str,
        rag_system: Optional[Dict] = None
    ) -> Tuple[bool, List[str]]:
        """Check if an issue is likely invalid by analyzing against code."""
        invalid_reasons = []
        
        if not rag_system:
            return False, invalid_reasons
        
        try:
            title = issue.get("title", "")
            body = issue.get("body", "")
            
            # Quick checks for obviously invalid issues
            if len(title) < 10:
                invalid_reasons.append("Title too short and non-descriptive")
            
            if not body or len(body) < 20:
                invalid_reasons.append("Issue body is empty or too short")
            
            # Check if issue describes functionality that doesn't exist
            functionality_query = f"""
            Based on the repository code, does the following issue describe functionality or behavior 
            that actually exists in the codebase?
            
            Issue Title: {title}
            Issue Description: {body[:500]}
            
            Look for:
            1. Referenced files, functions, or features that don't exist
            2. Described behavior that contradicts actual implementation
            3. Claims about functionality that isn't present in the code
            4. References to non-existent configuration or settings
            
            Respond with specific details if the issue describes non-existent functionality.
            """
            
            result = await query_rag_system(rag_system, functionality_query, chat_history=[])
            
            if isinstance(result, tuple):
                answer, _ = result
            else:
                answer = result
            
            # Analyze AI response for invalid issue indicators
            invalid_indicators = [
                "does not exist", "not found", "no such", "doesn't exist",
                "not implemented", "not present", "no evidence of",
                "contradicts", "incorrect assumption"
            ]
            
            if any(indicator in answer.lower() for indicator in invalid_indicators):
                invalid_reasons.append("References functionality that doesn't exist in the codebase")
            
            # Check if issue is about expected behavior
            behavior_query = f"""
            Is the behavior described in this issue actually the intended/correct behavior 
            based on the code implementation?
            
            Issue: {title}
            Description: {body[:300]}
            
            Check if this is reporting a bug that is actually intended behavior.
            """
            
            result = await query_rag_system(rag_system, behavior_query, chat_history=[])
            
            if isinstance(result, tuple):
                answer, _ = result
            else:
                answer = result
            
            if any(phrase in answer.lower() for phrase in [
                "intended behavior", "working as expected", "by design",
                "correct behavior", "not a bug"
            ]):
                invalid_reasons.append("Describes intended behavior, not a bug")
        
        except Exception as e:
            self.logger.debug(f"Invalid issue check failed: {e}")
        
        is_invalid = len(invalid_reasons) > 0
        return is_invalid, invalid_reasons
    
    def _generate_similarity_suggestions(
        self,
        similar_issues: List[SimilarIssue],
        clusters: List[IssueCluster],
        is_invalid: bool,
        invalid_reasons: List[str]
    ) -> List[str]:
        """Generate actionable suggestions based on similarity analysis."""
        suggestions = []
        
        if is_invalid:
            suggestions.append("❌ **Possible Invalid Issue**: " + "; ".join(invalid_reasons))
            suggestions.append("🔍 **Recommendation**: Verify the issue against actual codebase behavior")
        
        # Handle duplicates
        duplicate_clusters = [c for c in clusters if c.cluster_type == "duplicate"]
        if duplicate_clusters:
            for cluster in duplicate_clusters:
                primary_similar = cluster.similar_issues[0]
                if primary_similar.state == "open":
                    suggestions.append(
                        f"🔄 **Potential Duplicate**: This issue is very similar to #{primary_similar.issue_number} "
                        f"(similarity: {primary_similar.similarity_score:.2f}). Consider closing as duplicate."
                    )
                else:
                    suggestions.append(
                        f"🔄 **Similar Closed Issue**: Issue #{primary_similar.issue_number} was similar "
                        f"and is now {primary_similar.state}. Check if this is resolved."
                    )
        
        # Handle related issues
        related_clusters = [c for c in clusters if c.cluster_type == "related"]
        if related_clusters:
            for cluster in related_clusters:
                related_issues = ", ".join([f"#{issue.issue_number}" for issue in cluster.similar_issues[:3]])
                suggestions.append(
                    f"🔗 **Related Issues**: This issue is related to {related_issues}. "
                    f"Consider linking or referencing these issues."
                )
        
        # Handle high similarity without duplicates
        high_similarity = [issue for issue in similar_issues if issue.similarity_score > 0.75]
        if high_similarity and not duplicate_clusters:
            suggestions.append(
                f"⚠️ **High Similarity**: This issue is very similar to existing issues. "
                f"Please review #{high_similarity[0].issue_number} before proceeding."
            )
        
        # General suggestions
        if not suggestions:
            if similar_issues:
                suggestions.append(
                    f"ℹ️ **Similar Issues Found**: {len(similar_issues)} related issues exist. "
                    f"Consider reviewing them for context."
                )
            else:
                suggestions.append("✅ **Unique Issue**: No similar issues found. This appears to be a new issue.")
        
        return suggestions
    
    async def auto_comment_on_similar_issues(
        self,
        issue_number: int,
        repo_full_name: str,
        installation_id: int,
        similarity_result: IssueSimilarityResult
    ) -> bool:
        """Automatically comment on issues with similarity information."""
        if not similarity_result.has_duplicates and not similarity_result.is_likely_invalid:
            return False
        
        client = await get_github_app_installation_client(
            settings.github_app_id,
            settings.github_private_key,
            installation_id
        )
        
        if not client:
            return False
        
        # Construct comment
        comment_parts = ["🤖 **Automated Issue Analysis**\n"]
        
        if similarity_result.is_likely_invalid:
            comment_parts.append("❌ **Potential Issue with this Report:**")
            for reason in similarity_result.invalid_reasons:
                comment_parts.append(f"- {reason}")
            comment_parts.append("")
        
        if similarity_result.has_duplicates:
            comment_parts.append("🔄 **Duplicate Detection:**")
            duplicates = [
                issue for issue in similarity_result.similar_issues
                if issue.similarity_score > 0.85
            ]
            
            for dup in duplicates[:3]:  # Show top 3 duplicates
                comment_parts.append(
                    f"- Very similar to #{dup.issue_number}: *{dup.title}* "
                    f"(similarity: {dup.similarity_score:.1%})"
                )
            comment_parts.append("")
        
        if similarity_result.suggestions:
            comment_parts.append("💡 **Suggestions:**")
            for suggestion in similarity_result.suggestions:
                comment_parts.append(f"- {suggestion}")
        
        comment_parts.append("\n---")
        comment_parts.append("*This analysis was performed automatically. Please review and take appropriate action.*")
        
        comment = "\n".join(comment_parts)
        
        try:
            success = await post_issue_comment(
                client=client,
                repo_full_name=repo_full_name,
                issue_number=issue_number,
                comment=comment
            )
            
            if success:
                self.logger.info(f"Posted similarity analysis comment on issue #{issue_number}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to post similarity comment: {e}")
            return False

# Global instance
issue_similarity_service = IssueSimilarityService() 