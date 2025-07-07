"""
PR Analysis Service

Provides comprehensive pull request analysis including:
- Code quality assessment
- Security vulnerability detection  
- Bug pattern recognition
- Duplicate functionality detection
- Code complexity analysis
- AI-powered code review suggestions
"""

import logging
import re
import difflib
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass

from app.core.github_utils import (
    get_github_app_installation_client,
    post_issue_comment,
    SECURITY_PATTERNS,
    QUALITY_PATTERNS,
    _analyze_security_patterns,
    _analyze_quality_patterns,
    _calculate_complexity_score,
    _detect_language_from_extension
)
from app.core.rag_system import query_rag_system
from app.config import settings

logger = logging.getLogger("pr_analysis_service")

@dataclass
class PRAnalysisResult:
    """Result of PR analysis."""
    has_issues: bool
    security_issues: List[Dict[str, Any]]
    quality_issues: List[Dict[str, Any]]
    complexity_issues: List[Dict[str, Any]]
    potential_bugs: List[Dict[str, Any]]
    duplicate_functionality: List[Dict[str, Any]]
    suggestions: List[str]
    overall_score: int  # 0-100
    review_priority: str  # "low", "medium", "high", "critical"

@dataclass
class CodeChange:
    """Represents a code change in a PR."""
    file_path: str
    change_type: str  # "added", "modified", "deleted", "renamed"
    additions: int
    deletions: int
    old_content: Optional[str]
    new_content: Optional[str]
    diff: Optional[str]
    language: str

class PRAnalysisService:
    """Service for analyzing pull requests."""
    
    def __init__(self):
        self.logger = logging.getLogger("pr_analysis_service")
        
    async def analyze_pull_request(
        self,
        pr_data: Dict[str, Any],
        pr_files: List[Dict[str, Any]],
        repo_full_name: str,
        installation_id: int,
        rag_system: Optional[Dict] = None
    ) -> PRAnalysisResult:
        """
        Perform comprehensive analysis of a pull request.
        
        Args:
            pr_data: Pull request data from GitHub API
            pr_files: List of changed files in the PR
            repo_full_name: Repository full name
            installation_id: GitHub installation ID
            rag_system: Optional RAG system for context-aware analysis
            
        Returns:
            PRAnalysisResult with comprehensive analysis
        """
        self.logger.info(f"Starting comprehensive analysis of PR #{pr_data.get('number')} in {repo_full_name}")
        
        # Parse code changes
        code_changes = await self._parse_code_changes(pr_files, repo_full_name, installation_id)
        
        # Perform various analyses
        security_issues = await self._analyze_security_vulnerabilities(code_changes)
        quality_issues = await self._analyze_code_quality(code_changes)
        complexity_issues = await self._analyze_complexity(code_changes)
        potential_bugs = await self._detect_potential_bugs(code_changes, rag_system)
        duplicate_functionality = await self._detect_duplicate_functionality(code_changes, rag_system)
        
        # Generate overall assessment
        overall_score = self._calculate_overall_score(
            security_issues, quality_issues, complexity_issues, potential_bugs
        )
        
        review_priority = self._determine_review_priority(
            security_issues, quality_issues, complexity_issues, potential_bugs
        )
        
        suggestions = self._generate_suggestions(
            security_issues, quality_issues, complexity_issues, potential_bugs, duplicate_functionality
        )
        
        has_issues = bool(security_issues or quality_issues or complexity_issues or potential_bugs)
        
        result = PRAnalysisResult(
            has_issues=has_issues,
            security_issues=security_issues,
            quality_issues=quality_issues,
            complexity_issues=complexity_issues,
            potential_bugs=potential_bugs,
            duplicate_functionality=duplicate_functionality,
            suggestions=suggestions,
            overall_score=overall_score,
            review_priority=review_priority
        )
        
        self.logger.info(f"PR analysis completed. Score: {overall_score}, Priority: {review_priority}")
        return result
    
    async def _parse_code_changes(
        self,
        pr_files: List[Dict[str, Any]],
        repo_full_name: str,
        installation_id: int
    ) -> List[CodeChange]:
        """Parse and analyze code changes from PR files."""
        changes = []
        
        # Get GitHub client for fetching file contents
        client = await get_github_app_installation_client(
            settings.github_app_id,
            settings.github_private_key,
            installation_id
        )
        
        for file_data in pr_files:
            file_path = file_data.get("filename", "")
            change_type = file_data.get("status", "modified")
            additions = file_data.get("additions", 0)
            deletions = file_data.get("deletions", 0)
            patch = file_data.get("patch", "")
            
            language = _detect_language_from_extension(file_path.split('.')[-1] if '.' in file_path else "")
            
            # Try to fetch old and new content for more detailed analysis
            old_content = None
            new_content = None
            
            if client and change_type in ["modified", "deleted"]:
                try:
                    # This is a simplified approach - in reality you'd need the specific commit SHAs
                    repo = client.get_repo(repo_full_name)
                    if change_type == "modified":
                        try:
                            file_content = repo.get_contents(file_path)
                            new_content = file_content.decoded_content.decode('utf-8')
                        except:
                            pass
                except Exception as e:
                    self.logger.debug(f"Could not fetch content for {file_path}: {e}")
            
            changes.append(CodeChange(
                file_path=file_path,
                change_type=change_type,
                additions=additions,
                deletions=deletions,
                old_content=old_content,
                new_content=new_content,
                diff=patch,
                language=language
            ))
        
        return changes
    
    async def _analyze_security_vulnerabilities(self, code_changes: List[CodeChange]) -> List[Dict[str, Any]]:
        """Analyze code changes for security vulnerabilities."""
        vulnerabilities = []
        
        for change in code_changes:
            if change.change_type == "deleted":
                continue
                
            content_to_analyze = change.new_content or change.diff or ""
            if not content_to_analyze:
                continue
                
            # Check for security patterns
            security_flags = _analyze_security_patterns(content_to_analyze)
            
            for flag_type in security_flags:
                patterns = SECURITY_PATTERNS.get(flag_type, [])
                for pattern in patterns:
                    matches = re.finditer(pattern, content_to_analyze, re.IGNORECASE)
                    for match in matches:
                        line_num = content_to_analyze[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            "type": flag_type,
                            "severity": self._get_security_severity(flag_type),
                            "file": change.file_path,
                            "line": line_num,
                            "pattern": pattern,
                            "match": match.group(),
                            "description": self._get_security_description(flag_type),
                            "recommendation": self._get_security_recommendation(flag_type)
                        })
        
        return vulnerabilities
    
    async def _analyze_code_quality(self, code_changes: List[CodeChange]) -> List[Dict[str, Any]]:
        """Analyze code changes for quality issues."""
        quality_issues = []
        
        for change in code_changes:
            if change.change_type == "deleted":
                continue
                
            content_to_analyze = change.new_content or change.diff or ""
            if not content_to_analyze:
                continue
                
            # Check for quality patterns
            quality_flags = _analyze_quality_patterns(content_to_analyze)
            
            for flag_type in quality_flags:
                patterns = QUALITY_PATTERNS.get(flag_type, [])
                for pattern in patterns:
                    matches = re.finditer(pattern, content_to_analyze, re.IGNORECASE)
                    for match in matches:
                        line_num = content_to_analyze[:match.start()].count('\n') + 1
                        quality_issues.append({
                            "type": flag_type,
                            "severity": self._get_quality_severity(flag_type),
                            "file": change.file_path,
                            "line": line_num,
                            "pattern": pattern,
                            "match": match.group(),
                            "description": self._get_quality_description(flag_type),
                            "recommendation": self._get_quality_recommendation(flag_type)
                        })
            
            # Additional quality checks
            lines = content_to_analyze.split('\n')
            
            # Check for long lines
            for i, line in enumerate(lines):
                if len(line) > 120:  # Configurable line length limit
                    quality_issues.append({
                        "type": "long_line",
                        "severity": "low",
                        "file": change.file_path,
                        "line": i + 1,
                        "description": f"Line too long ({len(line)} characters)",
                        "recommendation": "Consider breaking long lines for better readability"
                    })
            
            # Check for excessive blank lines
            blank_line_count = 0
            for i, line in enumerate(lines):
                if not line.strip():
                    blank_line_count += 1
                    if blank_line_count > 3:  # More than 3 consecutive blank lines
                        quality_issues.append({
                            "type": "excessive_blank_lines",
                            "severity": "low",
                            "file": change.file_path,
                            "line": i + 1,
                            "description": "Too many consecutive blank lines",
                            "recommendation": "Remove excessive blank lines"
                        })
                else:
                    blank_line_count = 0
        
        return quality_issues
    
    async def _analyze_complexity(self, code_changes: List[CodeChange]) -> List[Dict[str, Any]]:
        """Analyze code changes for complexity issues."""
        complexity_issues = []
        
        for change in code_changes:
            if change.change_type == "deleted":
                continue
                
            content_to_analyze = change.new_content or change.diff or ""
            if not content_to_analyze:
                continue
                
            # Calculate overall file complexity
            complexity_score = _calculate_complexity_score(content_to_analyze)
            
            if complexity_score > 100:  # High complexity threshold
                complexity_issues.append({
                    "type": "high_complexity",
                    "severity": "medium" if complexity_score < 150 else "high",
                    "file": change.file_path,
                    "complexity_score": complexity_score,
                    "description": f"High complexity score: {complexity_score}",
                    "recommendation": "Consider refactoring to reduce complexity"
                })
            
            # Analyze function-level complexity
            if change.language == "python":
                complexity_issues.extend(self._analyze_python_function_complexity(change, content_to_analyze))
            elif change.language in ["javascript", "typescript"]:
                complexity_issues.extend(self._analyze_js_function_complexity(change, content_to_analyze))
        
        return complexity_issues
    
    def _analyze_python_function_complexity(self, change: CodeChange, content: str) -> List[Dict[str, Any]]:
        """Analyze Python function complexity."""
        issues = []
        lines = content.split('\n')
        
        current_function = None
        function_start = None
        indent_level = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
                
            current_indent = len(line) - len(line.lstrip())
            
            # Function definition
            func_match = re.match(r'def\s+(\w+)\s*\(', stripped)
            if func_match:
                # Analyze previous function if exists
                if current_function and function_start is not None:
                    func_complexity = self._calculate_function_complexity_python(
                        lines[function_start:i], current_function
                    )
                    if func_complexity > 10:  # McCabe complexity threshold
                        issues.append({
                            "type": "complex_function",
                            "severity": "medium" if func_complexity < 20 else "high",
                            "file": change.file_path,
                            "line": function_start + 1,
                            "function": current_function,
                            "complexity": func_complexity,
                            "description": f"Function '{current_function}' has high complexity ({func_complexity})",
                            "recommendation": "Consider breaking this function into smaller functions"
                        })
                
                current_function = func_match.group(1)
                function_start = i
                indent_level = current_indent
        
        # Analyze last function
        if current_function and function_start is not None:
            func_complexity = self._calculate_function_complexity_python(
                lines[function_start:], current_function
            )
            if func_complexity > 10:
                issues.append({
                    "type": "complex_function",
                    "severity": "medium" if func_complexity < 20 else "high",
                    "file": change.file_path,
                    "line": function_start + 1,
                    "function": current_function,
                    "complexity": func_complexity,
                    "description": f"Function '{current_function}' has high complexity ({func_complexity})",
                    "recommendation": "Consider breaking this function into smaller functions"
                })
        
        return issues
    
    def _analyze_js_function_complexity(self, change: CodeChange, content: str) -> List[Dict[str, Any]]:
        """Analyze JavaScript/TypeScript function complexity."""
        issues = []
        lines = content.split('\n')
        
        func_patterns = [
            r'function\s+(\w+)\s*\(',
            r'(\w+)\s*[:=]\s*function\s*\(',
            r'(\w+)\s*[:=]\s*\([^)]*\)\s*=>'
        ]
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            for pattern in func_patterns:
                match = re.search(pattern, stripped)
                if match:
                    func_name = match.group(1)
                    func_lines = self._extract_js_function_body(lines, i)
                    func_complexity = self._calculate_function_complexity_js(func_lines, func_name)
                    
                    if func_complexity > 10:
                        issues.append({
                            "type": "complex_function",
                            "severity": "medium" if func_complexity < 20 else "high",
                            "file": change.file_path,
                            "line": i + 1,
                            "function": func_name,
                            "complexity": func_complexity,
                            "description": f"Function '{func_name}' has high complexity ({func_complexity})",
                            "recommendation": "Consider breaking this function into smaller functions"
                        })
                    break
        
        return issues
    
    async def _detect_potential_bugs(
        self,
        code_changes: List[CodeChange],
        rag_system: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Detect potential bugs in code changes."""
        potential_bugs = []
        
        for change in code_changes:
            if change.change_type == "deleted":
                continue
                
            content_to_analyze = change.new_content or change.diff or ""
            if not content_to_analyze:
                continue
                
            # Common bug patterns
            bugs = self._detect_common_bug_patterns(change, content_to_analyze)
            potential_bugs.extend(bugs)
            
            # AI-assisted bug detection using RAG
            if rag_system:
                ai_bugs = await self._detect_bugs_with_ai(change, content_to_analyze, rag_system)
                potential_bugs.extend(ai_bugs)
        
        return potential_bugs
    
    def _detect_common_bug_patterns(self, change: CodeChange, content: str) -> List[Dict[str, Any]]:
        """Detect common bug patterns in code."""
        bugs = []
        lines = content.split('\n')
        
        bug_patterns = {
            'null_pointer': [
                r'\.(\w+)\s*\(',  # Method call without null check
                r'\[(\w+)\]',     # Array access without bounds check
            ],
            'resource_leak': [
                r'open\s*\([^)]+\)',  # File operations without proper closing
                r'connect\s*\([^)]+\)',  # Network connections
            ],
            'infinite_loop': [
                r'while\s*\(\s*true\s*\)',  # while(true) without break
                r'for\s*\(\s*;\s*;\s*\)',   # for(;;) without break
            ],
            'race_condition': [
                r'threading\.',  # Threading operations
                r'async\s+def',  # Async functions
            ],
            'sql_injection': [
                r'execute\s*\([^)]*\+[^)]*\)',  # String concatenation in SQL
                r'query\s*\([^)]*\+[^)]*\)',
            ]
        }
        
        for i, line in enumerate(lines):
            for bug_type, patterns in bug_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        bugs.append({
                            "type": bug_type,
                            "severity": self._get_bug_severity(bug_type),
                            "file": change.file_path,
                            "line": i + 1,
                            "pattern": pattern,
                            "description": self._get_bug_description(bug_type),
                            "recommendation": self._get_bug_recommendation(bug_type)
                        })
        
        return bugs
    
    async def _detect_bugs_with_ai(
        self,
        change: CodeChange,
        content: str,
        rag_system: Dict
    ) -> List[Dict[str, Any]]:
        """Use AI to detect potential bugs based on repository context."""
        bugs = []
        
        try:
            # Create a focused query for bug detection
            query = f"""
            Analyze the following code change for potential bugs, considering the repository context:
            
            File: {change.file_path}
            Language: {change.language}
            Change Type: {change.change_type}
            
            Code:
            {content[:2000]}  # Limit content size
            
            Look for:
            1. Logic errors
            2. Edge cases not handled
            3. Inconsistencies with existing code patterns
            4. Missing error handling
            5. Potential runtime errors
            
            Provide specific line-by-line feedback if issues are found.
            """
            
            result = await query_rag_system(rag_system, query, chat_history=[])
            
            if isinstance(result, tuple):
                answer, _ = result
            else:
                answer = result
            
            # Parse AI response for specific bug mentions
            # This is a simplified parsing - could be enhanced with structured output
            if "bug" in answer.lower() or "error" in answer.lower() or "issue" in answer.lower():
                bugs.append({
                    "type": "ai_detected",
                    "severity": "medium",
                    "file": change.file_path,
                    "description": "AI detected potential issue",
                    "ai_analysis": answer[:500],  # Truncate for storage
                    "recommendation": "Review the AI analysis and code carefully"
                })
                
        except Exception as e:
            self.logger.debug(f"AI bug detection failed for {change.file_path}: {e}")
        
        return bugs
    
    async def _detect_duplicate_functionality(
        self,
        code_changes: List[CodeChange],
        rag_system: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Detect if changes implement functionality that already exists."""
        duplicates = []
        
        if not rag_system:
            return duplicates
            
        for change in code_changes:
            if change.change_type in ["deleted", "renamed"]:
                continue
                
            content_to_analyze = change.new_content or change.diff or ""
            if not content_to_analyze:
                continue
                
            try:
                # Extract function names and purposes
                functions = self._extract_function_signatures(content_to_analyze, change.language)
                
                for func_info in functions:
                    # Query RAG system for similar functionality
                    query = f"""
                    Does the repository already contain similar functionality to this:
                    
                    Function: {func_info['name']}
                    File: {change.file_path}
                    Purpose: {func_info.get('purpose', 'Not specified')}
                    
                    Look for existing functions, classes, or modules that perform similar operations.
                    If similar functionality exists, specify the location and suggest whether this is:
                    1. A legitimate alternative implementation
                    2. Potential duplicate code that should be refactored
                    3. An enhancement to existing functionality
                    """
                    
                    result = await query_rag_system(rag_system, query, chat_history=[])
                    
                    if isinstance(result, tuple):
                        answer, _ = result
                    else:
                        answer = result
                    
                    # Simple heuristic to detect duplicates
                    if any(keyword in answer.lower() for keyword in [
                        "similar", "duplicate", "already exists", "same functionality"
                    ]):
                        duplicates.append({
                            "type": "duplicate_functionality",
                            "severity": "medium",
                            "file": change.file_path,
                            "function": func_info['name'],
                            "description": f"Potentially duplicate functionality: {func_info['name']}",
                            "ai_analysis": answer[:300],
                            "recommendation": "Review existing code to avoid duplication"
                        })
                        
            except Exception as e:
                self.logger.debug(f"Duplicate detection failed for {change.file_path}: {e}")
        
        return duplicates
    
    def _extract_function_signatures(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Extract function signatures and their purposes."""
        functions = []
        lines = content.split('\n')
        
        if language == "python":
            for i, line in enumerate(lines):
                match = re.match(r'def\s+(\w+)\s*\(([^)]*)\):', line.strip())
                if match:
                    func_name = match.group(1)
                    params = match.group(2)
                    
                    # Try to extract docstring as purpose
                    purpose = self._extract_function_purpose(lines, i + 1)
                    
                    functions.append({
                        "name": func_name,
                        "parameters": params,
                        "purpose": purpose,
                        "line": i + 1
                    })
        
        elif language in ["javascript", "typescript"]:
            func_patterns = [
                r'function\s+(\w+)\s*\(([^)]*)\)',
                r'(\w+)\s*[:=]\s*function\s*\(([^)]*)\)',
                r'(\w+)\s*[:=]\s*\(([^)]*)\)\s*=>'
            ]
            
            for i, line in enumerate(lines):
                for pattern in func_patterns:
                    match = re.search(pattern, line.strip())
                    if match:
                        func_name = match.group(1)
                        params = match.group(2) if len(match.groups()) > 1 else ""
                        
                        functions.append({
                            "name": func_name,
                            "parameters": params,
                            "purpose": "JavaScript function",
                            "line": i + 1
                        })
                        break
        
        return functions
    
    def _extract_function_purpose(self, lines: List[str], start_line: int) -> str:
        """Extract function purpose from docstring or comments."""
        if start_line >= len(lines):
            return "No description"
            
        # Check for Python docstring
        next_line = lines[start_line].strip()
        if next_line.startswith('"""') or next_line.startswith("'''"):
            quote_type = '"""' if next_line.startswith('"""') else "'''"
            
            if next_line.count(quote_type) >= 2:
                return next_line.strip(quote_type).strip()
            
            # Multi-line docstring
            purpose_lines = []
            for i in range(start_line, min(start_line + 5, len(lines))):
                line = lines[i]
                if quote_type in line and i > start_line:
                    purpose_lines.append(line.split(quote_type)[0])
                    break
                purpose_lines.append(line.strip(quote_type))
            
            return ' '.join(purpose_lines).strip()
        
        # Check for comment
        if next_line.startswith('#'):
            return next_line.lstrip('#').strip()
        
        return "No description"
    
    def _calculate_function_complexity_python(self, func_lines: List[str], func_name: str) -> int:
        """Calculate McCabe complexity for Python function."""
        complexity = 1  # Base complexity
        
        for line in func_lines:
            stripped = line.strip()
            # Count decision points
            complexity += len(re.findall(r'\b(if|elif|for|while|except|and|or)\b', stripped))
            complexity += stripped.count('?')  # Ternary operators
        
        return complexity
    
    def _calculate_function_complexity_js(self, func_lines: List[str], func_name: str) -> int:
        """Calculate McCabe complexity for JavaScript/TypeScript function."""
        complexity = 1  # Base complexity
        
        for line in func_lines:
            stripped = line.strip()
            # Count decision points
            complexity += len(re.findall(r'\b(if|for|while|switch|case|catch|&&|\|\|)\b', stripped))
            complexity += stripped.count('?')  # Ternary operators
        
        return complexity
    
    def _extract_js_function_body(self, lines: List[str], start_line: int) -> List[str]:
        """Extract JavaScript function body by tracking braces."""
        func_lines = []
        brace_count = 0
        found_opening = False
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            func_lines.append(line)
            
            for char in line:
                if char == '{':
                    brace_count += 1
                    found_opening = True
                elif char == '}':
                    brace_count -= 1
                    
            if found_opening and brace_count == 0:
                break
                
        return func_lines
    
    def _calculate_overall_score(
        self,
        security_issues: List[Dict],
        quality_issues: List[Dict],
        complexity_issues: List[Dict],
        potential_bugs: List[Dict]
    ) -> int:
        """Calculate overall PR quality score (0-100)."""
        score = 100
        
        # Deduct points for issues
        for issue in security_issues:
            severity = issue.get("severity", "medium")
            if severity == "critical":
                score -= 25
            elif severity == "high":
                score -= 15
            elif severity == "medium":
                score -= 10
            else:
                score -= 5
                
        for issue in quality_issues:
            severity = issue.get("severity", "medium")
            if severity == "high":
                score -= 10
            elif severity == "medium":
                score -= 5
            else:
                score -= 2
                
        for issue in complexity_issues:
            severity = issue.get("severity", "medium")
            if severity == "high":
                score -= 15
            elif severity == "medium":
                score -= 8
            else:
                score -= 3
                
        for issue in potential_bugs:
            severity = issue.get("severity", "medium")
            if severity == "critical":
                score -= 20
            elif severity == "high":
                score -= 12
            elif severity == "medium":
                score -= 8
            else:
                score -= 4
        
        return max(0, score)
    
    def _determine_review_priority(
        self,
        security_issues: List[Dict],
        quality_issues: List[Dict],
        complexity_issues: List[Dict],
        potential_bugs: List[Dict]
    ) -> str:
        """Determine review priority based on issues found."""
        
        # Critical priority for security issues
        if any(issue.get("severity") == "critical" for issue in security_issues):
            return "critical"
            
        # Critical priority for critical bugs
        if any(issue.get("severity") == "critical" for issue in potential_bugs):
            return "critical"
            
        # High priority for high severity issues
        high_severity_count = sum([
            len([i for i in security_issues if i.get("severity") == "high"]),
            len([i for i in potential_bugs if i.get("severity") == "high"]),
            len([i for i in complexity_issues if i.get("severity") == "high"])
        ])
        
        if high_severity_count > 0:
            return "high"
            
        # Medium priority for multiple medium issues
        medium_severity_count = sum([
            len([i for i in security_issues if i.get("severity") == "medium"]),
            len([i for i in quality_issues if i.get("severity") == "medium"]),
            len([i for i in complexity_issues if i.get("severity") == "medium"]),
            len([i for i in potential_bugs if i.get("severity") == "medium"])
        ])
        
        if medium_severity_count > 3:
            return "high"
        elif medium_severity_count > 1:
            return "medium"
            
        # Low priority for minor issues only
        total_issues = len(security_issues) + len(quality_issues) + len(complexity_issues) + len(potential_bugs)
        if total_issues > 0:
            return "low"
            
        return "low"
    
    def _generate_suggestions(
        self,
        security_issues: List[Dict],
        quality_issues: List[Dict],
        complexity_issues: List[Dict],
        potential_bugs: List[Dict],
        duplicate_functionality: List[Dict]
    ) -> List[str]:
        """Generate actionable suggestions for improvement."""
        suggestions = []
        
        if security_issues:
            suggestions.append("🔒 **Security Issues Found**: Please review and address security vulnerabilities before merging.")
            
        if potential_bugs:
            suggestions.append("🐛 **Potential Bugs Detected**: Consider adding error handling and edge case validation.")
            
        if complexity_issues:
            suggestions.append("🔄 **High Complexity**: Consider refactoring complex functions into smaller, more maintainable pieces.")
            
        if duplicate_functionality:
            suggestions.append("♻️ **Duplicate Code**: Review existing codebase to avoid reimplementing existing functionality.")
            
        if quality_issues:
            quality_types = set(issue.get("type") for issue in quality_issues)
            if "code_smells" in quality_types:
                suggestions.append("✨ **Code Quality**: Address TODOs, FIXMEs, and improve code structure.")
            if "long_line" in quality_types:
                suggestions.append("📏 **Line Length**: Consider breaking long lines for better readability.")
                
        if not any([security_issues, quality_issues, complexity_issues, potential_bugs]):
            suggestions.append("✅ **Good Job!**: No major issues detected. Code looks clean and well-structured.")
        
        return suggestions
    
    # Helper methods for getting descriptions and recommendations
    def _get_security_severity(self, flag_type: str) -> str:
        severity_map = {
            "sql_injection": "critical",
            "command_injection": "critical",
            "hardcoded_secrets": "high",
            "path_traversal": "high",
            "insecure_random": "medium"
        }
        return severity_map.get(flag_type, "medium")
    
    def _get_security_description(self, flag_type: str) -> str:
        descriptions = {
            "sql_injection": "Potential SQL injection vulnerability",
            "command_injection": "Potential command injection vulnerability",
            "hardcoded_secrets": "Hardcoded credentials or secrets detected",
            "path_traversal": "Potential path traversal vulnerability",
            "insecure_random": "Use of insecure random number generation"
        }
        return descriptions.get(flag_type, "Security vulnerability detected")
    
    def _get_security_recommendation(self, flag_type: str) -> str:
        recommendations = {
            "sql_injection": "Use parameterized queries or prepared statements",
            "command_injection": "Validate and sanitize all user inputs",
            "hardcoded_secrets": "Use environment variables or secure credential storage",
            "path_traversal": "Validate file paths and use safe path joining methods",
            "insecure_random": "Use cryptographically secure random number generators"
        }
        return recommendations.get(flag_type, "Review and fix security issue")
    
    def _get_quality_severity(self, flag_type: str) -> str:
        severity_map = {
            "code_smells": "low",
            "complexity_indicators": "medium",
            "deprecated_patterns": "medium"
        }
        return severity_map.get(flag_type, "low")
    
    def _get_quality_description(self, flag_type: str) -> str:
        descriptions = {
            "code_smells": "Code contains TODOs, FIXMEs, or other quality issues",
            "complexity_indicators": "High complexity patterns detected",
            "deprecated_patterns": "Use of deprecated or discouraged patterns"
        }
        return descriptions.get(flag_type, "Code quality issue detected")
    
    def _get_quality_recommendation(self, flag_type: str) -> str:
        recommendations = {
            "code_smells": "Address TODOs and FIXMEs, improve code structure",
            "complexity_indicators": "Simplify complex logic and reduce nesting",
            "deprecated_patterns": "Update to use modern, recommended patterns"
        }
        return recommendations.get(flag_type, "Improve code quality")
    
    def _get_bug_severity(self, bug_type: str) -> str:
        severity_map = {
            "null_pointer": "high",
            "resource_leak": "medium",
            "infinite_loop": "high",
            "race_condition": "high",
            "sql_injection": "critical"
        }
        return severity_map.get(bug_type, "medium")
    
    def _get_bug_description(self, bug_type: str) -> str:
        descriptions = {
            "null_pointer": "Potential null pointer dereference",
            "resource_leak": "Potential resource leak",
            "infinite_loop": "Potential infinite loop",
            "race_condition": "Potential race condition",
            "sql_injection": "Potential SQL injection"
        }
        return descriptions.get(bug_type, "Potential bug detected")
    
    def _get_bug_recommendation(self, bug_type: str) -> str:
        recommendations = {
            "null_pointer": "Add null checks before method calls or array access",
            "resource_leak": "Ensure proper resource cleanup with try-finally or context managers",
            "infinite_loop": "Add proper break conditions or loop counters",
            "race_condition": "Use proper synchronization mechanisms",
            "sql_injection": "Use parameterized queries"
        }
        return recommendations.get(bug_type, "Review and fix potential bug")

# Global instance
pr_analysis_service = PRAnalysisService() 