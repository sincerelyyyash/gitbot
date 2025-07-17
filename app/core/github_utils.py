import logging
from typing import Optional, List, Dict, Any, Set, Tuple
from github import Github, Auth, GithubIntegration
from github.GithubException import GithubException
from github.Repository import Repository
from github.ContentFile import ContentFile
from github.PullRequest import PullRequest as GithubPR
from github.File import File as GithubFile
import jwt
from app.config import settings
import time
import base64
from pathlib import Path
from app.core.github_rate_limiter import github_rate_limiter
from datetime import datetime, timedelta
import re
import difflib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("github_utils")

GITHUB_APP_ID = settings.github_app_id
GITHUB_PRIVATE_KEY_PATH = settings.github_private_key

# File extensions to include for code analysis
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.r', '.m',
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.sql', '.html', '.css', '.scss',
    '.sass', '.less', '.vue', '.svelte', '.yaml', '.yml', '.json', '.xml',
    '.toml', '.ini', '.cfg', '.conf', '.env', '.dockerfile', '.makefile'
}

# Documentation file extensions
DOC_EXTENSIONS = {
    '.md', '.rst', '.txt', '.adoc', '.org', '.tex', '.rtf'
}

# Security vulnerability patterns
SECURITY_PATTERNS = {
    "sql_injection": [
        r"execute\s*\([^)]*\+[^)]*\)",
        r"query\s*\([^)]*\+[^)]*\)",
        r"SELECT\s+.*\+.*FROM",
        r"INSERT\s+.*\+.*VALUES",
        r"UPDATE\s+.*SET\s+.*\+",
        r"DELETE\s+.*WHERE\s+.*\+"
    ],
    "command_injection": [
        r"os\.system\s*\([^)]*\+[^)]*\)",
        r"subprocess\.[^(]*\([^)]*\+[^)]*\)",
        r"exec\s*\([^)]*\+[^)]*\)",
        r"eval\s*\([^)]*\+[^)]*\)"
    ],
    "hardcoded_secrets": [
        r"password\s*=\s*['\"][^'\"]{8,}['\"]",
        r"api_key\s*=\s*['\"][^'\"]{20,}['\"]",
        r"secret\s*=\s*['\"][^'\"]{16,}['\"]",
        r"token\s*=\s*['\"][^'\"]{20,}['\"]"
    ],
    "path_traversal": [
        r"open\s*\([^)]*\.\.[^)]*\)",
        r"file\s*\([^)]*\.\.[^)]*\)",
        r"\.\.\/|\.\.\\\\",
        r"path.*\+.*\.\."
    ],
    "insecure_random": [
        r"random\.random\(\)",
        r"Math\.random\(\)",
        r"rand\(\)"
    ]
}

# Code quality patterns
QUALITY_PATTERNS = {
    "code_smells": [
        r"TODO|FIXME|HACK|XXX",
        r"print\s*\(",  # Debug prints
        r"console\.log\s*\(",  # Debug logs
        r"debugger;?"
    ],
    "complexity_indicators": [
        r"if.*if.*if.*if",  # Nested ifs
        r"for.*for.*for",   # Nested loops
        r"while.*while.*while"
    ],
    "deprecated_patterns": [
        r"import\s+.*\*",  # Wildcard imports
        r"\.format\s*\(",  # Old string formatting in Python
        r"var\s+\w+",      # var in JavaScript
    ]
}

# Files to exclude from processing
EXCLUDED_PATTERNS = {
    'node_modules', '.git', '.github', 'dist', 'build', 'target', 'out',
    '__pycache__', '.pytest_cache', '.coverage', '.nyc_output', 'coverage',
    '.env', '.env.local', '.env.production', '.env.development',
    'package-lock.json', 'yarn.lock', 'poetry.lock', 'Pipfile.lock'
}

# Security and quality patterns to check in code
SECURITY_PATTERNS = {
    'sql_injection': [
        r'execute\s*\(\s*["\'].*\+.*["\']',  # Basic SQL injection pattern
        r'query\s*=.*\+.*',                  # String concatenation in queries
        r'SELECT.*\+.*FROM',                 # Direct SQL string building
        r'INSERT.*\+.*VALUES',               # Direct SQL string building
        r'UPDATE.*SET.*\+',                  # Direct SQL string building
        r'DELETE.*WHERE.*\+',                # Direct SQL string building
    ],
    'hardcoded_secrets': [
        r'password\s*=\s*["\'][^"\']+["\']',      # Hardcoded passwords
        r'api_key\s*=\s*["\'][^"\']+["\']',       # Hardcoded API keys
        r'secret\s*=\s*["\'][^"\']+["\']',        # Hardcoded secrets
        r'token\s*=\s*["\'][^"\']+["\']',         # Hardcoded tokens
        r'["\'][A-Za-z0-9]{32,}["\']',            # Long alphanumeric strings
    ],
    'path_traversal': [
        r'\.\./',                                  # Directory traversal
        r'\.\.\\\\',                               # Windows directory traversal
        r'os\.path\.join\([^)]*\.\.[^)]*\)',      # Unsafe path join
    ],
    'command_injection': [
        r'os\.system\s*\(',                      # os.system calls
        r'subprocess\.(call|run|Popen)\s*\(',    # subprocess calls
        r'exec\s*\(',                            # exec calls
        r'eval\s*\(',                            # eval calls
    ],
    'insecure_random': [
        r'random\.random\(',                     # Insecure random
        r'Math\.random\(',                       # JavaScript insecure random
    ]
}

QUALITY_PATTERNS = {
    'code_smells': [
        r'TODO:.*',                              # TODOs
        r'FIXME:.*',                             # FIXMEs
        r'HACK:.*',                              # HACKs
        r'XXX:.*',                               # XXX markers
    ],
    'complexity_indicators': [
        r'if.*if.*if.*if',                       # Nested conditions
        r'for.*for.*for',                        # Nested loops
        r'while.*while.*while',                  # Nested loops
    ],
    'deprecated_patterns': [
        r'\.innerHTML\s*=',                      # innerHTML usage (XSS risk)
        r'document\.write\s*\(',                 # document.write
        r'eval\s*\(',                            # eval usage
    ]
}

def generate_jwt_token(app_id: str, private_key: str) -> Optional[str]:
    """Generate a JWT token for GitHub App authentication using the private key string."""
    try:
        # Use the private key directly from settings
        now = int(time.time())
        payload = {
            "iat": now - 60,
            "exp": now + (10 * 60),
            "iss": app_id
        }
        encoded_jwt = jwt.encode(payload, private_key, algorithm="RS256")
        logger.info("Generated JWT for GitHub App authentication.")
        return encoded_jwt if isinstance(encoded_jwt, str) else encoded_jwt.decode("utf-8")
    except Exception as e:
        logger.exception(f"Failed to generate JWT token: {str(e)}")
        return None

@github_rate_limiter.with_rate_limit(category="core")
async def get_github_app_installation_client(
    app_id: str,
    private_key: str,
    installation_id: int
) -> Optional[Github]:
    """Get an authenticated GitHub client for an installation."""
    try:
        # Create GitHub integration directly with app auth
        git_integration = GithubIntegration(
            auth=Auth.AppAuth(
                app_id=app_id,
                private_key=private_key
            )
        )
        
        try:
            # Get installation access token
            access_token = git_integration.get_access_token(installation_id).token
            
            # Create and return GitHub client
            return Github(auth=Auth.Token(access_token))
        except Exception as e:
            logger.error(f"GitHub integration failed: {str(e)}")
            return None
    
    except Exception as e:
        logger.error(f"Failed to get GitHub client: {str(e)}")
        return None

@github_rate_limiter.with_rate_limit(category="core")
async def post_issue_comment(
    client: Github,
    repo_full_name: str,
    issue_number: int,
    comment: str
) -> bool:
    """Post a comment on a GitHub issue, avoiding duplicates and internal errors."""
    # Internal error messages that should not be posted
    INTERNAL_ERROR_MESSAGES = [
        "RAG system not initialized",
        "API quota exceeded",
        "I encountered an unexpected error",
        "Failed to initialize RAG system",
        "Query must be a non-empty string."
    ]
    # If the comment contains any internal error message, do not post
    for err in INTERNAL_ERROR_MESSAGES:
        if err.lower() in comment.lower():
            logger.info(f"Not posting internal error message to GitHub: {err}")
            return False
    try:
        repo = client.get_repo(repo_full_name)
        issue = repo.get_issue(issue_number)
        # Fetch all comments for this issue
        existing_comments = [c.body.strip() for c in issue.get_comments() if c.body]
        # Avoid posting duplicate comments (ignoring whitespace)
        if any(comment.strip() == existing for existing in existing_comments):
            logger.info("Duplicate comment detected, not posting to GitHub.")
            return False
        issue.create_comment(comment)
        return True
    except Exception as e:
        logger.error(f"Failed to post comment: {e}")
        return False

def _should_exclude_file(file_path: str) -> bool:
    """Check if a file should be excluded from processing."""
    path_parts = set(Path(file_path).parts)
    return bool(path_parts.intersection(EXCLUDED_PATTERNS))

def _is_text_file(content: bytes) -> bool:
    """Check if content appears to be text (not binary)."""
    try:
        # First check for null bytes which are common in binary files
        if b'\x00' in content:
            return False
        
        # Try to decode as UTF-8
        text = content.decode('utf-8')
        
        # Check for common binary patterns in the decoded text
        # Control characters (except common whitespace)
        import string
        printable_chars = set(string.printable)
        
        # Allow for some non-printable characters but not too many
        non_printable_count = sum(1 for char in text if char not in printable_chars)
        non_printable_ratio = non_printable_count / len(text) if text else 0
        
        # If more than 30% of characters are non-printable, likely binary
        return non_printable_ratio < 0.3
        
    except UnicodeDecodeError:
        return False

def _get_file_extension(file_path: str) -> str:
    """Get the file extension in lowercase."""
    return Path(file_path).suffix.lower()

@github_rate_limiter.with_rate_limit(category="core")
async def fetch_repository_files(
    client: Github,
    repo_full_name: str,
    include_code: bool = True,
    include_docs: bool = True,
    max_file_size: int = 1024 * 1024,  # 1MB max file size
    path: str = "",
    enhanced_code_analysis: bool = True
) -> List[Dict[str, Any]]:
    """
    Enhanced repository file fetching with comprehensive code analysis.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name (owner/repo)
        include_code: Whether to include code files
        include_docs: Whether to include documentation files  
        max_file_size: Maximum file size to process (bytes)
        path: Starting path for recursive search
        enhanced_code_analysis: Whether to perform enhanced code analysis
        
    Returns:
        List of document dictionaries with enhanced metadata for code analysis
    """
    documents = []
    
    try:
        repo = client.get_repo(repo_full_name)
        logger.info(f"Fetching repository files from {repo_full_name} with enhanced analysis")
        
        async def process_contents(current_path: str, depth: int = 0):
            # Prevent infinite recursion and overly deep directory traversal
            if depth > 10:
                return
                
            try:
                items = repo.get_contents(current_path)
                if not hasattr(items, '__iter__'):
                    items = [items]
                    
                for item in items:
                    if item.type == "dir":
                        # Skip excluded directories
                        if _should_exclude_file(item.path):
                            continue
                        await process_contents(item.path, depth + 1)
                        
                    elif item.type == "file":
                        # Skip files that are too large
                        if item.size > max_file_size:
                            logger.info(f"Skipping large file: {item.path} ({item.size} bytes)")
                            continue
                            
                        # Skip excluded files
                        if _should_exclude_file(item.path):
                            continue
                            
                        file_ext = _get_file_extension(item.path)
                        
                        # Determine if we should process this file
                        should_process = False
                        if include_code and file_ext in CODE_EXTENSIONS:
                            should_process = True
                        elif include_docs and file_ext in DOC_EXTENSIONS:
                            should_process = True
                        elif file_ext in {'.gitignore', '.env.example', 'dockerfile'}:
                            should_process = True  # Include important config files
                            
                        if not should_process:
                            continue
                            
                        try:
                            content = item.decoded_content
                            if not _is_text_file(content):
                                continue
                                
                            # Decode content
                            try:
                                decoded_content = content.decode('utf-8')
                            except UnicodeDecodeError:
                                try:
                                    decoded_content = content.decode('latin-1')
                                except:
                                    logger.warning(f"Could not decode file: {item.path}")
                                    continue
                            
                            # Enhanced code analysis
                            if enhanced_code_analysis and file_ext in CODE_EXTENSIONS:
                                file_documents = _analyze_code_file(
                                    decoded_content, 
                                    item.path, 
                                    file_ext, 
                                    repo_full_name,
                                    item.sha,
                                    item.size
                                )
                                documents.extend(file_documents)
                            else:
                                # Standard file processing for non-code files
                                documents.append({
                                    "content": _create_file_content_summary(decoded_content, item.path),
                                    "metadata": {
                                        "type": "file",
                                        "file_path": item.path,
                                        "file_extension": file_ext,
                                        "file_size": item.size,
                                        "repository": repo_full_name,
                                        "sha": item.sha,
                                        "is_code": file_ext in CODE_EXTENSIONS,
                                        "is_documentation": file_ext in DOC_EXTENSIONS,
                                        "language": _detect_language_from_extension(file_ext)
                                    }
                                })
                                
                        except Exception as e:
                            logger.warning(f"Error processing file {item.path}: {e}")
                            continue
                            
            except Exception as e:
                logger.warning(f"Error processing directory {current_path}: {e}")
                return
        
        await process_contents(path)
        logger.info(f"Enhanced analysis fetched {len(documents)} document chunks from {repo_full_name}")
        return documents
        
    except Exception as e:
        logger.error(f"Failed to fetch repository files: {e}")
        return []

def _analyze_code_file(
    content: str, 
    file_path: str, 
    file_ext: str, 
    repo_full_name: str,
    sha: str,
    size: int
) -> List[Dict[str, Any]]:
    """
    Perform enhanced analysis of code files including function/class extraction.
    
    Args:
        content: File content
        file_path: Path to the file
        file_ext: File extension
        repo_full_name: Repository name
        sha: File SHA
        size: File size
        
    Returns:
        List of document chunks with enhanced metadata
    """
    documents = []
    language = _detect_language_from_extension(file_ext)
    
    # Base metadata for all chunks from this file
    base_metadata = {
        "type": "code_file",
        "file_path": file_path,
        "file_extension": file_ext,
        "file_size": size,
        "repository": repo_full_name,
        "sha": sha,
        "language": language,
        "is_code": True,
        "is_documentation": False
    }
    
    # Extract code structure based on language
    code_elements = _extract_code_elements(content, language, file_path)
    
    # Create file overview document
    file_overview = _create_file_overview(content, file_path, code_elements)
    documents.append({
        "content": file_overview,
        "metadata": {
            **base_metadata,
            "chunk_type": "file_overview",
            "functions_count": len(code_elements.get("functions", [])),
            "classes_count": len(code_elements.get("classes", [])),
            "imports_count": len(code_elements.get("imports", [])),
            "complexity_score": _calculate_complexity_score(content),
            "security_flags": _analyze_security_patterns(content),
            "quality_flags": _analyze_quality_patterns(content)
        }
    })
    
    # Create individual documents for functions and classes
    for func in code_elements.get("functions", []):
        documents.append({
            "content": func["content"],
            "metadata": {
                **base_metadata,
                "chunk_type": "function",
                "function_name": func["name"],
                "start_line": func.get("start_line"),
                "end_line": func.get("end_line"),
                "parameters": func.get("parameters", []),
                "docstring": func.get("docstring"),
                "complexity": func.get("complexity", 0)
            }
        })
    
    for cls in code_elements.get("classes", []):
        documents.append({
            "content": cls["content"], 
            "metadata": {
                **base_metadata,
                "chunk_type": "class",
                "class_name": cls["name"],
                "start_line": cls.get("start_line"),
                "end_line": cls.get("end_line"),
                "methods": cls.get("methods", []),
                "docstring": cls.get("docstring"),
                "inheritance": cls.get("inheritance", [])
            }
        })
    
    # If file is small enough, also include the full file content
    if len(content) < 5000:  # Less than 5KB
        documents.append({
            "content": f"Complete file content for {file_path}:\n\n{content}",
            "metadata": {
                **base_metadata,
                "chunk_type": "full_file"
            }
        })
    else:
        # For larger files, create intelligent chunks
        intelligent_chunks = _create_intelligent_code_chunks(content, language, file_path)
        for i, chunk in enumerate(intelligent_chunks):
            documents.append({
                "content": chunk["content"],
                "metadata": {
                    **base_metadata,
                    "chunk_type": "code_chunk",
                    "chunk_index": i,
                    "total_chunks": len(intelligent_chunks),
                    "start_line": chunk.get("start_line"),
                    "end_line": chunk.get("end_line")
                }
            })
    
    return documents

def _extract_code_elements(content: str, language: str, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Extract functions, classes, and other code elements based on language."""
    elements = {
        "functions": [],
        "classes": [],
        "imports": [],
        "constants": []
    }
    
    lines = content.split('\n')
    
    if language == "python":
        elements.update(_extract_python_elements(lines))
    elif language in ["javascript", "typescript"]:
        elements.update(_extract_js_ts_elements(lines))
    elif language == "java":
        elements.update(_extract_java_elements(lines))
    elif language in ["cpp", "c"]:
        elements.update(_extract_c_cpp_elements(lines))
    elif language == "go":
        elements.update(_extract_go_elements(lines))
    else:
        # Generic extraction for other languages
        elements.update(_extract_generic_elements(lines))
    
    return elements

def _extract_python_elements(lines: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract Python-specific code elements."""
    elements = {"functions": [], "classes": [], "imports": [], "constants": []}
    
    current_function = None
    current_class = None
    indentation_stack = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
            
        indent_level = len(line) - len(line.lstrip())
        
        # Handle imports
        if stripped.startswith(('import ', 'from ')):
            elements["imports"].append({
                "name": stripped,
                "line": i + 1,
                "content": line
            })
            
        # Handle constants (ALL_CAPS variables at module level)
        elif '=' in stripped and indent_level == 0 and stripped.split('=')[0].strip().isupper():
            const_name = stripped.split('=')[0].strip()
            elements["constants"].append({
                "name": const_name,
                "line": i + 1,
                "content": line
            })
            
        # Handle class definitions
        elif stripped.startswith('class '):
            if current_class:
                current_class["end_line"] = i
                current_class["content"] = '\n'.join(lines[current_class["start_line"]-1:i])
                elements["classes"].append(current_class)
            
            class_match = re.match(r'class\s+(\w+)(?:\([^)]*\))?:', stripped)
            if class_match:
                current_class = {
                    "name": class_match.group(1),
                    "start_line": i + 1,
                    "methods": [],
                    "docstring": _extract_docstring(lines, i + 1)
                }
                
        # Handle function definitions
        elif stripped.startswith('def '):
            if current_function:
                current_function["end_line"] = i
                current_function["content"] = '\n'.join(lines[current_function["start_line"]-1:i])
                if current_class and indent_level > 0:
                    current_class["methods"].append(current_function["name"])
                else:
                    elements["functions"].append(current_function)
            
            func_match = re.match(r'def\s+(\w+)\s*\(([^)]*)\):', stripped)
            if func_match:
                current_function = {
                    "name": func_match.group(1),
                    "start_line": i + 1,
                    "parameters": [p.strip() for p in func_match.group(2).split(',') if p.strip()],
                    "docstring": _extract_docstring(lines, i + 1),
                    "complexity": _calculate_function_complexity(lines, i)
                }
    
    # Handle last function/class
    if current_function:
        current_function["end_line"] = len(lines)
        current_function["content"] = '\n'.join(lines[current_function["start_line"]-1:])
        if current_class:
            current_class["methods"].append(current_function["name"])
        else:
            elements["functions"].append(current_function)
            
    if current_class:
        current_class["end_line"] = len(lines)
        current_class["content"] = '\n'.join(lines[current_class["start_line"]-1:])
        elements["classes"].append(current_class)
    
    return elements

def _extract_js_ts_elements(lines: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract JavaScript/TypeScript code elements."""
    elements = {"functions": [], "classes": [], "imports": [], "constants": []}
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            continue
            
        # Handle imports
        if stripped.startswith(('import ', 'const ', 'require(')):
            elements["imports"].append({
                "name": stripped,
                "line": i + 1,
                "content": line
            })
            
        # Handle function definitions
        func_patterns = [
            r'function\s+(\w+)\s*\(',
            r'(\w+)\s*[:=]\s*function\s*\(',
            r'(\w+)\s*[:=]\s*\([^)]*\)\s*=>'
        ]
        
        for pattern in func_patterns:
            match = re.search(pattern, stripped)
            if match:
                elements["functions"].append({
                    "name": match.group(1),
                    "line": i + 1,
                    "content": line,
                    "complexity": _calculate_function_complexity(lines, i)
                })
                break
                
        # Handle class definitions
        class_match = re.search(r'class\s+(\w+)', stripped)
        if class_match:
            elements["classes"].append({
                "name": class_match.group(1),
                "line": i + 1,
                "content": line
            })
    
    return elements

def _extract_java_elements(lines: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract Java code elements."""
    elements = {"functions": [], "classes": [], "imports": [], "constants": []}
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            continue
            
        # Handle imports
        if stripped.startswith('import '):
            elements["imports"].append({
                "name": stripped,
                "line": i + 1,
                "content": line
            })
            
        # Handle class definitions
        class_match = re.search(r'(?:public|private|protected)?\s*class\s+(\w+)', stripped)
        if class_match:
            elements["classes"].append({
                "name": class_match.group(1),
                "line": i + 1,
                "content": line
            })
            
        # Handle method definitions
        method_match = re.search(r'(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\(', stripped)
        if method_match and not class_match:
            elements["functions"].append({
                "name": method_match.group(1),
                "line": i + 1,
                "content": line,
                "complexity": _calculate_function_complexity(lines, i)
            })
    
    return elements

def _extract_c_cpp_elements(lines: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract C/C++ code elements."""
    elements = {"functions": [], "classes": [], "imports": [], "constants": []}
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            continue
            
        # Handle includes
        if stripped.startswith('#include'):
            elements["imports"].append({
                "name": stripped,
                "line": i + 1,
                "content": line
            })
            
        # Handle class definitions (C++)
        class_match = re.search(r'class\s+(\w+)', stripped)
        if class_match:
            elements["classes"].append({
                "name": class_match.group(1),
                "line": i + 1,
                "content": line
            })
            
        # Handle function definitions
        func_match = re.search(r'^\s*[\w\*]+\s+(\w+)\s*\([^)]*\)\s*{?', line)
        if func_match and not any(keyword in stripped for keyword in ['if', 'while', 'for', 'switch']):
            elements["functions"].append({
                "name": func_match.group(1),
                "line": i + 1,
                "content": line,
                "complexity": _calculate_function_complexity(lines, i)
            })
    
    return elements

def _extract_go_elements(lines: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract Go code elements."""
    elements = {"functions": [], "classes": [], "imports": [], "constants": []}
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            continue
            
        # Handle imports
        if stripped.startswith('import'):
            elements["imports"].append({
                "name": stripped,
                "line": i + 1,
                "content": line
            })
            
        # Handle function definitions
        func_match = re.search(r'func\s+(\w+)\s*\(', stripped)
        if func_match:
            elements["functions"].append({
                "name": func_match.group(1),
                "line": i + 1,
                "content": line,
                "complexity": _calculate_function_complexity(lines, i)
            })
            
        # Handle struct definitions (Go's equivalent to classes)
        struct_match = re.search(r'type\s+(\w+)\s+struct', stripped)
        if struct_match:
            elements["classes"].append({
                "name": struct_match.group(1),
                "line": i + 1,
                "content": line
            })
    
    return elements

def _extract_generic_elements(lines: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Generic extraction for unsupported languages."""
    elements = {"functions": [], "classes": [], "imports": [], "constants": []}
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
            
        # Look for function-like patterns
        if any(pattern in stripped.lower() for pattern in ['function', 'def', 'func', 'method']):
            # Try to extract a name
            name_match = re.search(r'(\w+)\s*[({]', stripped)
            if name_match:
                elements["functions"].append({
                    "name": name_match.group(1),
                    "line": i + 1,
                    "content": line
                })
    
    return elements

def _extract_docstring(lines: List[str], start_line: int) -> Optional[str]:
    """Extract docstring for Python functions/classes."""
    if start_line >= len(lines):
        return None
        
    next_line = lines[start_line].strip()
    if next_line.startswith('"""') or next_line.startswith("'''"):
        quote_type = '"""' if next_line.startswith('"""') else "'''"
        
        # Single line docstring
        if next_line.count(quote_type) >= 2:
            return next_line.strip(quote_type).strip()
            
        # Multi-line docstring
        docstring_lines = [next_line.strip(quote_type)]
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if quote_type in line:
                docstring_lines.append(line.split(quote_type)[0])
                break
            docstring_lines.append(line)
            
        return '\n'.join(docstring_lines).strip()
    
    return None

def _calculate_function_complexity(lines: List[str], start_line: int) -> int:
    """Calculate cyclomatic complexity of a function."""
    complexity = 1  # Base complexity
    brace_count = 0
    
    for i in range(start_line, min(start_line + 100, len(lines))):  # Look ahead max 100 lines
        line = lines[i].strip()
        
        # Count decision points
        complexity += len(re.findall(r'\b(if|elif|else|for|while|switch|case|catch|except)\b', line))
        complexity += len(re.findall(r'[?&|]{2}', line))  # && || operators
        
        # Track braces to know when function ends (approximate)
        brace_count += line.count('{') - line.count('}')
        if brace_count < 0:  # Function likely ended
            break
            
    return min(complexity, 50)  # Cap complexity at 50

def _calculate_complexity_score(content: str) -> int:
    """Calculate overall file complexity score."""
    lines = content.split('\n')
    complexity = 0
    
    for line in lines:
        # Count nested structures
        complexity += len(re.findall(r'\b(if|for|while|switch|try|catch)\b', line))
        complexity += line.count('&&') + line.count('||')
        complexity += max(0, (len(line) - len(line.lstrip())) // 4)  # Indentation complexity
        
    return min(complexity, 200)  # Cap at 200

def _analyze_security_patterns(content: str) -> List[str]:
    """Analyze content for security vulnerability patterns."""
    flags = []
    content_lower = content.lower()
    
    for category, patterns in SECURITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                flags.append(category)
                break
                
    return flags

def _analyze_quality_patterns(content: str) -> List[str]:
    """Analyze content for code quality patterns."""
    flags = []
    
    for category, patterns in QUALITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                flags.append(category)
                break
                
    return flags

def _create_file_overview(content: str, file_path: str, code_elements: Dict[str, List]) -> str:
    """Create a comprehensive overview of the file."""
    overview_parts = [f"File Overview: {file_path}"]
    
    # File statistics
    lines = content.split('\n')
    overview_parts.append(f"Lines: {len(lines)}")
    overview_parts.append(f"Size: {len(content)} characters")
    
    # Code structure summary
    if code_elements.get("functions"):
        func_names = [f["name"] for f in code_elements["functions"]]
        overview_parts.append(f"Functions ({len(func_names)}): {', '.join(func_names[:10])}")
        if len(func_names) > 10:
            overview_parts.append(f"... and {len(func_names) - 10} more functions")
            
    if code_elements.get("classes"):
        class_names = [c["name"] for c in code_elements["classes"]]
        overview_parts.append(f"Classes ({len(class_names)}): {', '.join(class_names)}")
        
    if code_elements.get("imports"):
        overview_parts.append(f"Imports: {len(code_elements['imports'])}")
        
    # Add first few lines as context
    overview_parts.append("\nFile beginning:")
    overview_parts.append('\n'.join(lines[:10]))
    
    if len(lines) > 10:
        overview_parts.append("...")
        
    return '\n'.join(overview_parts)

def _create_file_content_summary(content: str, file_path: str) -> str:
    """Create a summary for non-code files."""
    lines = content.split('\n')
    
    summary_parts = [f"File: {file_path}"]
    
    # For documentation files, include more content
    if _get_file_extension(file_path) in DOC_EXTENSIONS:
        # Include up to first 50 lines for docs
        preview_lines = min(50, len(lines))
        summary_parts.extend(lines[:preview_lines])
        if len(lines) > preview_lines:
            summary_parts.append(f"... ({len(lines) - preview_lines} more lines)")
    else:
        # For config files, include first 20 lines
        preview_lines = min(20, len(lines))
        summary_parts.extend(lines[:preview_lines])
        if len(lines) > preview_lines:
            summary_parts.append(f"... ({len(lines) - preview_lines} more lines)")
            
    return '\n'.join(summary_parts)

def _create_intelligent_code_chunks(content: str, language: str, file_path: str) -> List[Dict[str, Any]]:
    """Create intelligent chunks for large code files."""
    lines = content.split('\n')
    chunks = []
    current_chunk = []
    current_start_line = 1
    chunk_size_limit = 100  # lines per chunk
    
    function_start = None
    class_start = None
    brace_count = 0
    
    for i, line in enumerate(lines):
        current_chunk.append(line)
        
        # Track code structure to avoid breaking in the middle of functions/classes
        if language == "python":
            if re.match(r'^\s*(def|class)\s+', line):
                if len(current_chunk) > chunk_size_limit and current_chunk[:-1]:
                    # Save current chunk (excluding this line)
                    chunks.append({
                        "content": '\n'.join(current_chunk[:-1]),
                        "start_line": current_start_line,
                        "end_line": current_start_line + len(current_chunk) - 2
                    })
                    current_chunk = [line]
                    current_start_line = i + 1
        else:
            # For other languages, use brace counting
            brace_count += line.count('{') - line.count('}')
            
            if len(current_chunk) > chunk_size_limit and brace_count == 0:
                chunks.append({
                    "content": '\n'.join(current_chunk),
                    "start_line": current_start_line,
                    "end_line": current_start_line + len(current_chunk) - 1
                })
                current_chunk = []
                current_start_line = i + 2
    
    # Add remaining chunk
    if current_chunk:
        chunks.append({
            "content": '\n'.join(current_chunk),
            "start_line": current_start_line,
            "end_line": current_start_line + len(current_chunk) - 1
        })
    
    return chunks

def _detect_language_from_extension(file_ext: str) -> str:
    """Detect programming language from file extension."""
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.m': 'objective-c',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'zsh',
        '.fish': 'fish',
        '.ps1': 'powershell',
        '.sql': 'sql',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.less': 'less',
        '.vue': 'vue',
        '.svelte': 'svelte',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'config',
        '.conf': 'config',
        '.env': 'env',
        '.dockerfile': 'dockerfile',
        '.makefile': 'makefile',
        '.md': 'markdown',
        '.rst': 'rst',
        '.txt': 'text'
    }
    
    return language_map.get(file_ext.lower(), 'unknown')

# PR-specific utility functions

@github_rate_limiter.with_rate_limit(category="pr")
async def get_pr_files(
    client: Github,
    repo_full_name: str,
    pr_number: int
) -> List[Dict[str, Any]]:
    """
    Get detailed information about files changed in a PR.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name
        pr_number: Pull request number
        
    Returns:
        List of file change information
    """
    try:
        repo = client.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        
        files_data = []
        for file in pr.get_files():
            files_data.append({
                "filename": file.filename,
                "status": file.status,  # added, modified, deleted, renamed
                "additions": file.additions,
                "deletions": file.deletions,
                "changes": file.changes,
                "patch": file.patch,
                "previous_filename": getattr(file, 'previous_filename', None),
                "blob_url": file.blob_url,
                "raw_url": file.raw_url,
                "sha": file.sha
            })
        
        logger.info(f"Retrieved {len(files_data)} files for PR #{pr_number}")
        return files_data
        
    except Exception as e:
        logger.error(f"Failed to get PR files for #{pr_number}: {e}")
        return []

@github_rate_limiter.with_rate_limit(category="pr")
async def get_pr_diff(
    client: Github,
    repo_full_name: str,
    pr_number: int,
    file_path: Optional[str] = None
) -> Optional[str]:
    """
    Get the diff for a PR or specific file in a PR.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name
        pr_number: Pull request number
        file_path: Optional specific file path
        
    Returns:
        Diff string or None if error
    """
    try:
        repo = client.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        
        if file_path:
            # Get diff for specific file
            for file in pr.get_files():
                if file.filename == file_path:
                    return file.patch
            return None
        else:
            # Get full PR diff
            # Note: GitHub API doesn't provide full diff directly
            # This would require combining individual file patches
            files = pr.get_files()
            diff_parts = []
            
            for file in files:
                if file.patch:
                    diff_parts.append(f"--- a/{file.filename}")
                    diff_parts.append(f"+++ b/{file.filename}")
                    diff_parts.append(file.patch)
                    diff_parts.append("")
            
            return "\n".join(diff_parts)
        
    except Exception as e:
        logger.error(f"Failed to get PR diff for #{pr_number}: {e}")
        return None

@github_rate_limiter.with_rate_limit(category="pr")
async def post_pr_comment(
    client: Github,
    repo_full_name: str,
    pr_number: int,
    comment: str
) -> bool:
    """
    Post a comment on a pull request.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name
        pr_number: Pull request number
        comment: Comment text
        
    Returns:
        True if successful, False otherwise
    """
    try:
        repo = client.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        pr.create_issue_comment(comment)
        logger.info(f"Posted comment on PR #{pr_number}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to post PR comment on #{pr_number}: {e}")
        return False

@github_rate_limiter.with_rate_limit(category="pr")
async def post_pr_review_comment(
    client: Github,
    repo_full_name: str,
    pr_number: int,
    file_path: str,
    line: int,
    comment: str,
    commit_sha: Optional[str] = None
) -> bool:
    """
    Post a review comment on a specific line of a PR.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name
        pr_number: Pull request number
        file_path: Path to the file
        line: Line number (1-indexed)
        comment: Comment text
        commit_sha: Optional commit SHA
        
    Returns:
        True if successful, False otherwise
    """
    try:
        repo = client.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        
        # Get the latest commit if not provided
        if not commit_sha:
            commit_sha = pr.head.sha
        
        pr.create_review_comment(
            body=comment,
            commit=repo.get_commit(commit_sha),
            path=file_path,
            line=line
        )
        
        logger.info(f"Posted review comment on PR #{pr_number} at {file_path}:{line}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to post PR review comment: {e}")
        return False

@github_rate_limiter.with_rate_limit(category="pr")
async def create_pr_review(
    client: Github,
    repo_full_name: str,
    pr_number: int,
    review_body: str,
    event: str = "COMMENT",  # APPROVE, REQUEST_CHANGES, COMMENT
    review_comments: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """
    Create a comprehensive review for a PR.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name
        pr_number: Pull request number
        review_body: Overall review comment
        event: Review event type
        review_comments: List of line-specific comments
        
    Returns:
        True if successful, False otherwise
    """
    try:
        repo = client.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        
        # Prepare review comments
        comments = []
        if review_comments:
            for comment_data in review_comments:
                comments.append({
                    "path": comment_data["file_path"],
                    "line": comment_data["line"],
                    "body": comment_data["comment"]
                })
        
        pr.create_review(
            body=review_body,
            event=event,
            comments=comments
        )
        
        logger.info(f"Created review for PR #{pr_number} with event: {event}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create PR review: {e}")
        return False

@github_rate_limiter.with_rate_limit(category="pr")
async def get_pr_reviews(
    client: Github,
    repo_full_name: str,
    pr_number: int
) -> List[Dict[str, Any]]:
    """
    Get all reviews for a PR.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name
        pr_number: Pull request number
        
    Returns:
        List of review information
    """
    try:
        repo = client.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        
        reviews_data = []
        for review in pr.get_reviews():
            reviews_data.append({
                "id": review.id,
                "user": review.user.login,
                "state": review.state,
                "body": review.body,
                "submitted_at": review.submitted_at.isoformat() if review.submitted_at else None,
                "commit_id": review.commit_id
            })
        
        logger.info(f"Retrieved {len(reviews_data)} reviews for PR #{pr_number}")
        return reviews_data
        
    except Exception as e:
        logger.error(f"Failed to get PR reviews: {e}")
        return []

@github_rate_limiter.with_rate_limit(category="pr")
async def get_pr_commits(
    client: Github,
    repo_full_name: str,
    pr_number: int
) -> List[Dict[str, Any]]:
    """
    Get all commits in a PR.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name
        pr_number: Pull request number
        
    Returns:
        List of commit information
    """
    try:
        repo = client.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        
        commits_data = []
        for commit in pr.get_commits():
            commits_data.append({
                "sha": commit.sha,
                "message": commit.commit.message,
                "author": commit.commit.author.name,
                "date": commit.commit.author.date.isoformat(),
                "url": commit.html_url,
                "stats": {
                    "additions": commit.stats.additions,
                    "deletions": commit.stats.deletions,
                    "total": commit.stats.total
                }
            })
        
        logger.info(f"Retrieved {len(commits_data)} commits for PR #{pr_number}")
        return commits_data
        
    except Exception as e:
        logger.error(f"Failed to get PR commits: {e}")
        return []

@github_rate_limiter.with_rate_limit(category="pr")
async def check_pr_mergeable(
    client: Github,
    repo_full_name: str,
    pr_number: int
) -> Dict[str, Any]:
    """
    Check if a PR is mergeable and get merge status.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name
        pr_number: Pull request number
        
    Returns:
        Merge status information
    """
    try:
        repo = client.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        
        # Refresh PR data to get latest merge status
        pr.update()
        
        status_info = {
            "mergeable": pr.mergeable,
            "mergeable_state": pr.mergeable_state,
            "merged": pr.merged,
            "merge_commit_sha": pr.merge_commit_sha,
            "rebaseable": pr.rebaseable,
            "can_merge": pr.mergeable and pr.mergeable_state == "clean"
        }
        
        # Get status checks
        if hasattr(pr, 'head') and hasattr(pr.head, 'sha'):
            try:
                status = repo.get_commit(pr.head.sha).get_combined_status()
                status_info["status_checks"] = {
                    "state": status.state,
                    "total_count": status.total_count,
                    "statuses": [
                        {
                            "context": s.context,
                            "state": s.state,
                            "description": s.description
                        }
                        for s in status.statuses
                    ]
                }
            except Exception as e:
                logger.debug(f"Could not get status checks: {e}")
                status_info["status_checks"] = None
        
        return status_info
        
    except Exception as e:
        logger.error(f"Failed to check PR mergeable status: {e}")
        return {
            "mergeable": None,
            "mergeable_state": "unknown",
            "merged": False,
            "can_merge": False,
            "error": str(e)
        }

@github_rate_limiter.with_rate_limit(category="pr")
async def add_pr_labels(
    client: Github,
    repo_full_name: str,
    pr_number: int,
    labels: List[str]
) -> bool:
    """
    Add labels to a PR.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name
        pr_number: Pull request number
        labels: List of label names
        
    Returns:
        True if successful, False otherwise
    """
    try:
        repo = client.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        
        # Get the issue associated with the PR to add labels
        issue = repo.get_issue(pr_number)
        issue.add_to_labels(*labels)
        
        logger.info(f"Added labels {labels} to PR #{pr_number}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add labels to PR #{pr_number}: {e}")
        return False

@github_rate_limiter.with_rate_limit(category="pr") 
async def request_pr_reviewers(
    client: Github,
    repo_full_name: str,
    pr_number: int,
    reviewers: List[str],
    team_reviewers: Optional[List[str]] = None
) -> bool:
    """
    Request reviewers for a PR.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name
        pr_number: Pull request number
        reviewers: List of usernames
        team_reviewers: Optional list of team names
        
    Returns:
        True if successful, False otherwise
    """
    try:
        repo = client.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        
        pr.create_review_request(
            reviewers=reviewers,
            team_reviewers=team_reviewers or []
        )
        
        logger.info(f"Requested reviewers {reviewers} for PR #{pr_number}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to request reviewers for PR #{pr_number}: {e}")
        return False

async def fetch_repository_issues(
    client: Github,
    repo_full_name: str,
    state: str = "all",
    max_issues: int = 100,
    include_comments: bool = True
) -> List[Dict[str, Any]]:
    """
    Fetch repository issues and optionally their comments.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name (owner/repo)
        state: Issue state ("open", "closed", "all")
        max_issues: Maximum number of issues to fetch
        include_comments: Whether to include issue comments
        
    Returns:
        List of document dictionaries with issue content and metadata
    """
    documents = []
    
    try:
        repo = client.get_repo(repo_full_name)
        logger.info(f"Fetching issues from {repo_full_name} (state: {state})")
        
        issues = repo.get_issues(state=state, sort="updated", direction="desc")
        
        issue_count = 0
        for issue in issues:
            if issue_count >= max_issues:
                break
                
            # Skip pull requests (they appear in issues list)
            if issue.pull_request:
                continue
                
            issue_count += 1
            
            # Process issue body
            issue_body = issue.body or ""
            if issue_body.strip():
                documents.append({
                    "content": f"Issue #{issue.number}: {issue.title}\n\n{issue_body}",
                    "metadata": {
                        "type": "issue",
                        "issue_number": issue.number,
                        "title": issue.title,
                        "state": issue.state,
                        "created_at": issue.created_at.isoformat(),
                        "updated_at": issue.updated_at.isoformat(),
                        "url": issue.html_url,
                        "labels": [label.name for label in issue.labels],
                        "repository": repo_full_name
                    }
                })
            
            # Process issue comments if requested
            if include_comments and issue.comments > 0:
                try:
                    comments = issue.get_comments()
                    for comment in comments:
                        if comment.body and comment.body.strip():
                            documents.append({
                                "content": f"Comment on Issue #{issue.number}:\n\n{comment.body}",
                                "metadata": {
                                    "type": "issue_comment",
                                    "issue_number": issue.number,
                                    "comment_id": comment.id,
                                    "created_at": comment.created_at.isoformat(),
                                    "updated_at": comment.updated_at.isoformat(),
                                    "url": comment.html_url,
                                    "repository": repo_full_name
                                }
                            })
                except Exception as e:
                    logger.warning(f"Failed to fetch comments for issue #{issue.number}: {e}")
        
        logger.info(f"Fetched {len(documents)} issue documents from {repo_full_name}")
        return documents
        
    except GithubException as e:
        logger.error(f"GitHub API error fetching issues: {e}")
        return []
    except Exception as e:
        logger.exception(f"Failed to fetch issues from {repo_full_name}")
        return []

async def fetch_repository_pull_requests(
    client: Github,
    repo_full_name: str,
    state: str = "all",
    max_prs: int = 50,
    include_comments: bool = True,
    include_review_comments: bool = True
) -> List[Dict[str, Any]]:
    """
    Fetch repository pull requests and optionally their comments.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name (owner/repo)
        state: PR state ("open", "closed", "all")
        max_prs: Maximum number of PRs to fetch
        include_comments: Whether to include PR comments
        include_review_comments: Whether to include code review comments
        
    Returns:
        List of document dictionaries with PR content and metadata
    """
    documents = []
    
    try:
        repo = client.get_repo(repo_full_name)
        logger.info(f"Fetching pull requests from {repo_full_name} (state: {state})")
        
        pulls = repo.get_pulls(state=state, sort="updated", direction="desc")
        
        pr_count = 0
        for pr in pulls:
            if pr_count >= max_prs:
                break
                
            pr_count += 1
            
            # Process PR body
            pr_body = pr.body or ""
            if pr_body.strip():
                documents.append({
                    "content": f"Pull Request #{pr.number}: {pr.title}\n\n{pr_body}",
                    "metadata": {
                        "type": "pull_request",
                        "pr_number": pr.number,
                        "title": pr.title,
                        "state": pr.state,
                        "created_at": pr.created_at.isoformat(),
                        "updated_at": pr.updated_at.isoformat(),
                        "merged": pr.merged,
                        "url": pr.html_url,
                        "labels": [label.name for label in pr.labels],
                        "repository": repo_full_name
                    }
                })
            
            # Process PR comments if requested
            if include_comments:
                try:
                    comments = pr.get_issue_comments()
                    for comment in comments:
                        if comment.body and comment.body.strip():
                            documents.append({
                                "content": f"Comment on PR #{pr.number}:\n\n{comment.body}",
                                "metadata": {
                                    "type": "pr_comment",
                                    "pr_number": pr.number,
                                    "comment_id": comment.id,
                                    "created_at": comment.created_at.isoformat(),
                                    "updated_at": comment.updated_at.isoformat(),
                                    "url": comment.html_url,
                                    "repository": repo_full_name
                                }
                            })
                except Exception as e:
                    logger.warning(f"Failed to fetch comments for PR #{pr.number}: {e}")
            
            # Process review comments if requested
            if include_review_comments:
                try:
                    review_comments = pr.get_review_comments()
                    for comment in review_comments:
                        if comment.body and comment.body.strip():
                            documents.append({
                                "content": f"Review comment on PR #{pr.number}:\n\n{comment.body}",
                                "metadata": {
                                    "type": "pr_review_comment",
                                    "pr_number": pr.number,
                                    "comment_id": comment.id,
                                    "file_path": comment.path,
                                    "line": comment.line,
                                    "created_at": comment.created_at.isoformat(),
                                    "updated_at": comment.updated_at.isoformat(),
                                    "url": comment.html_url,
                                    "repository": repo_full_name
                                }
                            })
                except Exception as e:
                    logger.warning(f"Failed to fetch review comments for PR #{pr.number}: {e}")
        
        logger.info(f"Fetched {len(documents)} PR documents from {repo_full_name}")
        return documents
        
    except GithubException as e:
        logger.error(f"GitHub API error fetching pull requests: {e}")
        return []
    except Exception as e:
        logger.exception(f"Failed to fetch pull requests from {repo_full_name}")
        return []

async def fetch_repository_metadata(client: Github, repo_full_name: str) -> List[Dict[str, Any]]:
    """
    Fetch repository metadata including README, description, and topics.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name (owner/repo)
        
    Returns:
        List of document dictionaries with repository metadata
    """
    documents = []
    
    try:
        repo = client.get_repo(repo_full_name)
        logger.info(f"Fetching repository metadata from {repo_full_name}")
        
        # Repository description and topics
        repo_info = []
        if repo.description:
            repo_info.append(f"Description: {repo.description}")
        
        if repo.topics:
            repo_info.append(f"Topics: {', '.join(repo.topics)}")
            
        if repo.language:
            repo_info.append(f"Primary Language: {repo.language}")
            
        if repo_info:
            documents.append({
                "content": f"Repository Information for {repo_full_name}:\n\n" + "\n".join(repo_info),
                "metadata": {
                    "type": "repository_metadata",
                    "repository": repo_full_name,
                    "language": repo.language,
                    "topics": repo.topics,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "created_at": repo.created_at.isoformat(),
                    "updated_at": repo.updated_at.isoformat()
                }
            })
        
        # Try to fetch README content
        try:
            readme = repo.get_readme()
            readme_content = base64.b64decode(readme.content).decode('utf-8')
            if readme_content.strip():
                documents.append({
                    "content": f"README for {repo_full_name}:\n\n{readme_content}",
                    "metadata": {
                        "type": "readme",
                        "file_path": readme.path,
                        "repository": repo_full_name,
                        "sha": readme.sha
                    }
                })
        except Exception as e:
            logger.info(f"No README found or could not fetch README: {e}")
        
        logger.info(f"Fetched {len(documents)} metadata documents from {repo_full_name}")
        return documents
        
    except GithubException as e:
        logger.error(f"GitHub API error fetching repository metadata: {e}")
        return []
    except Exception as e:
        logger.exception(f"Failed to fetch repository metadata from {repo_full_name}")
        return []

@github_rate_limiter.with_rate_limit(category="core")
async def fetch_all_repository_content(
    client: Github,
    repo_full_name: str,
    include_issues: bool = True,
    include_pulls: bool = True,
    include_code: bool = True,
    include_docs: bool = True,
    max_items: int = 100,
    enhanced_code_analysis: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch all repository content with enhanced code analysis and rate limit handling.
    
    Args:
        client: Authenticated GitHub client
        repo_full_name: Repository full name (owner/repo)
        include_issues: Whether to include issues
        include_pulls: Whether to include pull requests
        include_code: Whether to include code files with enhanced analysis
        include_docs: Whether to include documentation files
        max_items: Maximum number of issues/PRs to fetch
        enhanced_code_analysis: Whether to perform detailed code analysis
        
    Returns:
        Dictionary containing files, issues, and pulls with enhanced metadata
    """
    try:
        repo = client.get_repo(repo_full_name)
        logger.info(f"Fetching comprehensive repository content from {repo_full_name}")
        
        content = {
            "files": [],
            "issues": [],
            "pulls": [],
            "metadata": []
        }
        
        # Fetch files with enhanced analysis
        content["files"] = await fetch_repository_files(
            client=client,
            repo_full_name=repo_full_name,
            include_code=include_code,
            include_docs=include_docs,
            enhanced_code_analysis=enhanced_code_analysis
        )
        
        # Fetch repository metadata
        content["metadata"] = await fetch_repository_metadata(client, repo_full_name)
        
        # Fetch issues with detailed analysis
        if include_issues:
            content["issues"] = await fetch_repository_issues(
                client=client,
                repo_full_name=repo_full_name,
                max_issues=max_items,
                include_comments=True
            )
        
        # Fetch pull requests with detailed analysis
        if include_pulls:
            content["pulls"] = await fetch_repository_pull_requests(
                client=client,
                repo_full_name=repo_full_name,
                max_prs=max_items,
                include_comments=True,
                include_review_comments=True
            )
        
        total_documents = len(content["files"]) + len(content["issues"]) + len(content["pulls"]) + len(content["metadata"])
        logger.info(f"Fetched {total_documents} total documents from {repo_full_name}")
        
        return content
    
    except Exception as e:
        logger.error(f"Failed to fetch repository content: {e}")
        return {"files": [], "issues": [], "pulls": [], "metadata": []} 