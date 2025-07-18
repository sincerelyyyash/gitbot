import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from app.config import settings
import os
import chromadb
from chromadb.config import Settings as ChromaSettings
import hashlib
import shutil
import logging
import asyncio
import json
from app.core.quota_manager import quota_manager
from chromadb.db.base import UniqueConstraintError
from google.api_core.exceptions import PermissionDenied, ResourceExhausted, InvalidArgument
from langchain_google_genai._common import GoogleGenerativeAIError

# Error types for better error handling
class RAGError(Exception):
    """Base class for RAG system errors."""
    pass

class APIKeyRestrictedError(RAGError):
    """Raised when API key has IP address restrictions."""
    pass

class QuotaExceededError(RAGError):
    """Raised when API quota is exceeded."""
    pass

class InvalidAPIKeyError(RAGError):
    """Raised when API key is invalid or unauthorized."""
    pass

class NetworkError(RAGError):
    """Raised when there are network connectivity issues."""
    pass

def _categorize_google_api_error(error: Exception) -> RAGError:
    """Categorize Google API errors into specific error types."""
    error_str = str(error).lower()
    
    if "ip address restriction" in error_str or "api_key_ip_address_blocked" in error_str:
        return APIKeyRestrictedError(
            "Your Google API key has IP address restrictions. "
            "Please either remove the IP restrictions in Google Cloud Console "
            "or add your current IP address to the allowed list."
        )
    elif "quota exceeded" in error_str or "resource_exhausted" in error_str:
        return QuotaExceededError(
            "Google API quota exceeded. Please wait before trying again "
            "or check your quota limits in Google Cloud Console."
        )
    elif "invalid" in error_str and "api" in error_str:
        return InvalidAPIKeyError(
            "Invalid Google API key. Please check your GEMINI_API_KEY "
            "in the environment configuration."
        )
    elif "network" in error_str or "connection" in error_str or "timeout" in error_str:
        return NetworkError(
            "Network connectivity issue. Please check your internet connection "
            "and try again."
        )
    else:
        return RAGError(f"Google API error: {str(error)}")

def _get_error_suggestions(error: RAGError) -> List[str]:
    """Get actionable suggestions based on error type."""
    if isinstance(error, APIKeyRestrictedError):
        return [
            "Remove IP address restrictions from your Google Cloud Console API key",
            "Add your current IP address to the allowed list in Google Cloud Console",
            "Consider using a different API key without IP restrictions",
            "If using a VPN or proxy, try disabling it temporarily"
        ]
    elif isinstance(error, QuotaExceededError):
        return [
            "Wait for quota to reset (usually daily)",
            "Check your quota usage in Google Cloud Console",
            "Consider upgrading your API plan",
            "Implement request throttling in your application"
        ]
    elif isinstance(error, InvalidAPIKeyError):
        return [
            "Verify your GEMINI_API_KEY in environment configuration",
            "Ensure the API key is active in Google Cloud Console",
            "Check if the API key has the necessary permissions",
            "Generate a new API key if the current one is corrupted"
        ]
    elif isinstance(error, NetworkError):
        return [
            "Check your internet connection",
            "Try again in a few minutes",
            "Verify firewall settings aren't blocking the request",
            "Consider using a different network if available"
        ]
    else:
        return [
            "Check the application logs for more details",
            "Verify all configuration parameters are correct",
            "Try restarting the application"
        ]

def _mask_api_key(api_key: str) -> str:
    if not api_key or len(api_key) < 8:
        return "***"
    return api_key[:4] + "..." + api_key[-4:]

def _validate_documents_data(documents_data: List[Dict[str, Any]]) -> bool:
    if not isinstance(documents_data, list):
        return False
    for doc in documents_data:
        if not isinstance(doc, dict) or "content" not in doc:
            return False
    return True

def _sanitize_collection_name(repo_full_name: str) -> str:
    """
    Create a valid ChromaDB collection name from repository full name.
    ChromaDB collection names must be 3-63 characters, alphanumeric + hyphens + underscores.
    """
    # Handle empty or None input
    if not repo_full_name:
        return "default_repo"
    
    # Replace special characters with underscores
    sanitized = repo_full_name.replace("/", "_").replace("-", "_")
    # Remove any other special characters
    sanitized = "".join(c if c.isalnum() or c in "_" else "_" for c in sanitized)
    
    # Remove multiple consecutive underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    
    # Handle empty result after sanitization
    if not sanitized:
        sanitized = "repo"
    
    # Ensure it starts with alphanumeric
    if not sanitized[0].isalnum():
        sanitized = "repo_" + sanitized
    
    # Truncate if too long, but keep it meaningful
    if len(sanitized) > 60:
        # Use first part + hash of full name to ensure uniqueness
        hash_suffix = hashlib.md5(repo_full_name.encode()).hexdigest()[:8]
        sanitized = sanitized[:50] + "_" + hash_suffix
    
    # Ensure minimum length
    if len(sanitized) < 3:
        sanitized = sanitized + "_repo"
    
    return sanitized.lower()

def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize metadata for ChromaDB compatibility.
    ChromaDB only accepts str, int, float, bool values in metadata.
    Complex types like lists, dicts, etc. need to be converted or removed.
    """
    sanitized = {}
    
    for key, value in metadata.items():
        if value is None:
            # Skip None values
            continue
        elif isinstance(value, (str, int, float, bool)):
            # These are directly supported by ChromaDB
            sanitized[key] = value
        elif isinstance(value, list):
            # Convert lists to strings or skip if empty
            if value:
                # Convert non-empty lists to comma-separated strings
                try:
                    # Try to join if all elements are strings/numbers
                    str_items = [str(item) for item in value if item is not None]
                    if str_items:
                        sanitized[key] = ", ".join(str_items)
                except Exception:
                    # If conversion fails, skip this metadata
                    pass
            # Skip empty lists entirely
        elif isinstance(value, dict):
            # Convert dicts to JSON strings if not too large
            try:
                json_str = json.dumps(value)
                if len(json_str) < 1000:  # Reasonable size limit
                    sanitized[f"{key}_json"] = json_str
            except Exception:
                # If conversion fails, skip this metadata
                pass
        else:
            # For other types, try to convert to string
            try:
                str_value = str(value)
                if len(str_value) < 500:  # Reasonable size limit
                    sanitized[key] = str_value
            except Exception:
                # If conversion fails, skip this metadata
                pass
    
    return sanitized

def _filter_complex_metadata_from_documents(documents: List[Document]) -> List[Document]:
    """
    Filter complex metadata from documents for ChromaDB compatibility.
    This function creates new Document instances with sanitized metadata.
    """
    filtered_docs = []
    
    for doc in documents:
        # Sanitize the metadata
        sanitized_metadata = _sanitize_metadata(doc.metadata)
        
        # Create a new document with sanitized metadata
        filtered_doc = Document(
            page_content=doc.page_content,
            metadata=sanitized_metadata
        )
        filtered_docs.append(filtered_doc)
    
    return filtered_docs

def _setup_persistent_chromadb(persist_directory: str) -> chromadb.ClientAPI:
    """
    Set up ChromaDB client with persistent storage configuration.
    """
    # Ensure the persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # Configure ChromaDB with persistence
    chroma_settings = ChromaSettings(
        persist_directory=persist_directory,
        is_persistent=True,
        allow_reset=True,
        anonymized_telemetry=False
    )
    
    # Create persistent client
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=chroma_settings
    )
    
    logging.info(f"ChromaDB persistent client initialized at: {persist_directory}")
    return client

def _get_or_create_collection(
    client: chromadb.ClientAPI,
    collection_name: str,
    embedding_function,
    reset_collection: bool = False
) -> chromadb.Collection:
    """
    Get existing collection or create a new one.
    """
    try:
        if reset_collection:
            # Delete existing collection if it exists
            try:
                client.delete_collection(collection_name)
                logging.info(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass  # Collection might not exist
        # Try to get existing collection
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        logging.info(f"Retrieved existing collection: {collection_name}")
    except Exception as e:
        # Try to create collection, handle UniqueConstraintError
        try:
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logging.info(f"Created new collection: {collection_name}")
        except UniqueConstraintError:
            # Collection already exists, get it
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            logging.info(f"Collection {collection_name} already exists, retrieved instead.")
        except Exception as ce:
            logging.error(f"Failed to create or get collection: {collection_name}: {ce}")
            raise
    return collection

class TokenCounterCallbackHandler(BaseCallbackHandler):
    """Callback handler to count tokens used in LLM calls."""
    def __init__(self):
        super().__init__()
        self.total_tokens = 0
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        pass
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        # Gemini doesn't provide token counts directly, estimate based on text length
        # Rough estimate: 4 chars = 1 token for English text
        if response.generations and response.generations[0] and response.generations[0][0]:
            self.total_tokens += len(response.generations[0][0].text) // 4

async def initialize_rag_system(
    documents_data: List[Dict[str, Any]],
    gemini_api_key: str,
    chroma_persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    reset_collection: bool = False,
    repo_full_name: Optional[str] = None  # Add repo name parameter
) -> Union[dict, dict]:
    """Initialize RAG system with quota management."""
    chroma_persist_dir = chroma_persist_dir or settings.chromadb_persist_directory
    chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1024"))
    chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "128"))
    
    if not _validate_documents_data(documents_data):
        logging.error("Invalid documents_data: must be a list of dicts with 'content' key.")
        return {"error": "Invalid documents_data: must be a list of dicts with 'content' key."}
    if not gemini_api_key:
        logging.error("Missing Gemini API key.")
        return {"error": "Missing Gemini API key."}

    # Generate collection name if not provided
    if not collection_name:
        # Use a hash of the first few documents to create a unique collection name
        content_hash = hashlib.md5(
            "".join([doc.get("content", "")[:100] for doc in documents_data[:5]]).encode()
        ).hexdigest()[:8]
        collection_name = f"rag_collection_{content_hash}"
    
    # Sanitize collection name for ChromaDB
    collection_name = _sanitize_collection_name(collection_name)

    logging.info(f"Initializing RAG system with {len(documents_data)} documents")
    logging.info(f"ChromaDB dir: {chroma_persist_dir}, Collection: {collection_name}")
    logging.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    logging.info(f"Gemini key: {_mask_api_key(gemini_api_key)}")

    try:
        # Check quota before proceeding
        if repo_full_name:
            if not await quota_manager.check_quota(repo_full_name):
                logging.warning(f"API quota exceeded for {repo_full_name}")
                return {"error": "API quota exceeded. Please try again later."}
        
        # Prepare documents
        documents = [
            Document(page_content=doc["content"], metadata=doc.get("metadata", {}))
            for doc in documents_data
        ]
        
        # Enhanced text splitter configuration for better context preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or 1500,  # Larger chunks for better context
            chunk_overlap=chunk_overlap or 300,  # More overlap to preserve context
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks  
                "\n#",   # Markdown headers
                "\n##",  # Markdown sub-headers
                "\n###", # Markdown sub-sub-headers
                "```",   # Code blocks
                "class ", # Class definitions
                "def ",   # Function definitions
                "function ", # JS function definitions
                "const ", # Variable declarations
                "let ",   # Variable declarations
                "var ",   # Variable declarations
                "import ", # Import statements
                "from ",  # Import statements
                "package ", # Package declarations
                ". ",     # Sentence endings
                "? ",     # Question endings
                "! ",     # Exclamation endings
                " ",      # Word boundaries
                ""        # Character level (fallback)
            ]
        )
        
        # Process documents with enhanced metadata
        split_docs = []
        for doc_data in documents_data:
            # Create document with enhanced content
            content = doc_data["content"]
            metadata = doc_data.get("metadata", {})
            
            # Enhance content based on document type for better searchability
            enhanced_content = content
            doc_type = metadata.get("type", "")
            
            # Add searchable prefixes for better retrieval
            if doc_type in ["readme", "repository_metadata"]:
                enhanced_content = f"PROJECT OVERVIEW AND SETUP:\n{content}"
            elif doc_type == "file" and metadata.get("file_path", "").endswith((".json", ".py", ".js", ".ts", ".md")):
                file_path = metadata.get("file_path", "")
                language = metadata.get("language", "")
                enhanced_content = f"FILE: {file_path} ({language})\n{content}"
            elif doc_type in ["issue", "pr"]:
                enhanced_content = f"REPOSITORY DISCUSSION:\n{content}"
            
            doc = Document(
                page_content=enhanced_content,
                metadata=_sanitize_metadata(metadata)
            )
            
            # Split into chunks
            chunks = text_splitter.split_documents([doc])
            split_docs.extend(chunks)
        
        # Filter complex metadata from documents
        split_docs = _filter_complex_metadata_from_documents(split_docs)
        
        logging.info(f"Split {len(documents_data)} documents into {len(split_docs)} chunks for vectorization")

        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=gemini_api_key,
            model="models/embedding-001",
            task_type="retrieval_query"  
        )
        
        # Wrap embeddings in a ChromaDB-compatible class
        class ChromaEmbeddingFunction:
            def __init__(self, embeddings):
                self.embeddings = embeddings
            # ChromaDB expects __call__(self, input)
            def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
                if isinstance(input, str):
                    input = [input]
                return self.embeddings.embed_documents(input)
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return self.__call__(texts)
            def embed_query(self, text: str) -> List[float]:
                return self.embeddings.embed_query(text)
        
        chroma_embeddings = ChromaEmbeddingFunction(embeddings)
        
        # Set up persistent ChromaDB client
        chroma_client = _setup_persistent_chromadb(chroma_persist_dir)
        
        # Get or create collection
        collection = _get_or_create_collection(
            chroma_client,
            collection_name,
            chroma_embeddings,
            reset_collection=reset_collection
        )
        
        # Create vector store with persistent collection
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=chroma_embeddings,
            persist_directory=chroma_persist_dir
        )
        
        # Add documents if collection is empty or we're resetting
        if reset_collection or collection.count() == 0:
            logging.info("Adding documents to vector store...")
            try:
                vectorstore.add_documents(split_docs)
                logging.info(f"Added {len(split_docs)} document chunks to collection.")
            except Exception as e:
                # If we get metadata validation errors, try using langchain's metadata filter
                if "Expected metadata value to be a str, int, float or bool" in str(e):
                    logging.warning(f"ChromaDB metadata validation failed, attempting to filter complex metadata: {str(e)}")
                    
                    try:
                        # Import and use langchain's metadata filter as suggested in the error
                        from langchain_community.vectorstores.utils import filter_complex_metadata
                        
                        # Apply additional filtering
                        filtered_docs = filter_complex_metadata(split_docs)
                        logging.info(f"Applied langchain metadata filtering, trying again with {len(filtered_docs)} documents.")
                        
                        vectorstore.add_documents(filtered_docs)
                        logging.info(f"Successfully added {len(filtered_docs)} document chunks after metadata filtering.")
                        
                    except ImportError:
                        logging.error("langchain_community.vectorstores.utils.filter_complex_metadata not available")
                        # Fallback: Create minimal documents with only basic metadata
                        minimal_docs = []
                        for doc in split_docs:
                            minimal_doc = Document(
                                page_content=doc.page_content,
                                metadata={
                                    "source": str(doc.metadata.get("source", "unknown")),
                                    "type": str(doc.metadata.get("type", "document"))
                                }
                            )
                            minimal_docs.append(minimal_doc)
                        
                        vectorstore.add_documents(minimal_docs)
                        logging.info(f"Added {len(minimal_docs)} document chunks with minimal metadata as fallback.")
                        
                    except Exception as filter_error:
                        logging.error(f"Even metadata filtering failed: {str(filter_error)}")
                        # Last resort: Add documents with no metadata
                        try:
                            no_metadata_docs = [
                                Document(page_content=doc.page_content, metadata={})
                                for doc in split_docs
                            ]
                            vectorstore.add_documents(no_metadata_docs)
                            logging.info(f"Added {len(no_metadata_docs)} document chunks with no metadata as last resort.")
                        except Exception as final_error:
                            logging.error(f"Failed to add documents even without metadata: {str(final_error)}")
                            raise  # Re-raise the original error
                else:
                    # For other types of errors, re-raise immediately
                    raise
        else:
            logging.info(f"Using existing collection with {collection.count()} documents.")
        
        # Create enhanced retriever with better configuration for all question types
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Use MMR search for better diversity in results
            search_kwargs={
                "k": min(12, max(6, collection.count() // 10)),  # Dynamic k based on collection size, more docs for complex questions
                "lambda_mult": 0.7,  # Balance between relevance and diversity
                "fetch_k": min(30, collection.count())  # Fetch more candidates before MMR filtering
            }
        )
        
        # Initialize conversation memory with larger capacity
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            max_token_limit=4000  # Increased memory capacity
        )
        
        # Initialize LLM with token counter
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  
            google_api_key=gemini_api_key,
            temperature=0.3,  
            max_output_tokens=64000, 
            convert_system_message_to_human=True,
            callbacks=[TokenCounterCallbackHandler()]
        )
        
        # Create conversational retrieval chain with custom prompt
        custom_prompt = _create_custom_prompt()
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        
        logging.info("RAG system initialized successfully.")
        
        # Update quota usage if repo name provided
        if repo_full_name:
            # Estimate embedding tokens: ~3 tokens per word
            embedding_tokens = sum(len(doc["content"].split()) * 3 for doc in documents_data)
            await quota_manager.update_usage(repo_full_name, embedding_tokens)
        
        return {
            "qa_chain": qa_chain,
            "memory": memory,
            "collection_name": collection_name,
            "collection_count": collection.count(),
            "vectorstore": vectorstore,
            "chroma_client": chroma_client,
            "token_counter": TokenCounterCallbackHandler()  # Include token counter in return
        }
        
    except (GoogleGenerativeAIError, PermissionDenied, ResourceExhausted, InvalidArgument) as e:
        # Handle Google API specific errors
        categorized_error = _categorize_google_api_error(e)
        logging.error(f"Google API error during RAG initialization: {categorized_error}")
        
        return {
            "error": str(categorized_error),
            "error_type": type(categorized_error).__name__,
            "fallback_available": True,
            "suggestions": _get_error_suggestions(categorized_error)
        }
        
    except Exception as e:
        logging.exception("Failed to initialize RAG system.")
        return {
            "error": f"Failed to initialize RAG system: {str(e)}",
            "error_type": "UnknownError",
            "fallback_available": False,
            "suggestions": ["Check logs for more details", "Verify all configuration parameters"]
        }

def _create_custom_prompt():
    """Create a custom prompt template for better responses."""
    from langchain.prompts import PromptTemplate
    
    template = """You are a helpful AI assistant that answers questions about GitHub repositories. You have access to comprehensive information about the repository including source code, documentation, configuration files, and project history.

When answering questions about the repository:

FOR PROJECT EXPLANATION QUESTIONS (what is this project, what does it do, what framework/language):
- Look for package.json, requirements.txt, setup.py, Cargo.toml, go.mod and similar configuration files to identify frameworks and dependencies
- Examine README files and documentation for project descriptions and setup instructions
- Analyze source code structure to understand the project type (web app, library, CLI tool, etc.)
- Identify the main programming languages from file extensions and imports
- Provide clear, comprehensive explanations about the project's purpose and technology stack
- Include setup/installation instructions when available

FOR TECHNICAL QUESTIONS:
- Reference specific files, functions, or code sections when relevant
- Provide code examples when helpful
- Explain technical concepts clearly

IMPORTANT GUIDELINES:
- Base your answers strictly on the actual repository content provided
- Never mention "knowledge base", "vector database", "embeddings", or other internal technical details
- If you cannot find specific information in the repository, clearly state what information is not available
- Be comprehensive but well-organized with clear sections and bullet points
- Focus on practical, actionable information
- When discussing frameworks or technologies, be specific about versions when available

Context from the repository:
{context}

Chat History:
{chat_history}

Question: {question}

Answer based on the repository content:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"]
    )

async def query_rag_system(
    qa_chain_dict: dict,
    query: str,
    repo_full_name: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    github_client: Optional[Any] = None
) -> Union[str, Tuple[str, Dict]]:
    """Query the RAG system with quota management."""
    if not qa_chain_dict or "qa_chain" not in qa_chain_dict:
        logging.error("qa_chain_dict missing 'qa_chain'.")
        error_msg = "RAG system not initialized."
        if repo_full_name:
            # Return dummy usage stats for consistency
            return error_msg, {"tokens_remaining": 0, "tokens_used": 0}
        return error_msg
        
    if not query or not isinstance(query, str):
        logging.error("Query must be a non-empty string.")
        error_msg = "Query must be a non-empty string."
        if repo_full_name:
            # Return dummy usage stats for consistency
            return error_msg, {"tokens_remaining": 0, "tokens_used": 0}
        return error_msg
    
    try:
        # Check quota before proceeding
        if repo_full_name:
            if not await quota_manager.check_quota(repo_full_name):
                error_msg = "API quota exceeded. Please try again later."
                # Return current usage stats even when quota exceeded
                usage_stats = await quota_manager.get_usage_stats(repo_full_name)
                return error_msg, usage_stats
        
        qa_chain = qa_chain_dict["qa_chain"]
        token_counter = qa_chain_dict.get("token_counter")
        
        # Reset token counter
        if token_counter:
            token_counter.total_tokens = 0
        
        # Format chat history
        formatted_history = []
        if chat_history:
            for message in chat_history:
                formatted_history.append((message["role"], message["content"]))

        # Execute query
        result = await qa_chain.ainvoke({"question": query, "chat_history": formatted_history})
        
        answer = result.get("answer", "No answer generated.")
        source_docs = result.get("source_documents", [])
        
        # Update quota usage if repo name provided
        if repo_full_name and token_counter:
            await quota_manager.update_usage(repo_full_name, token_counter.total_tokens)
        
        # Return answer and usage stats if repo name provided
        if repo_full_name:
            usage_stats = await quota_manager.get_usage_stats(repo_full_name)
            return answer, usage_stats
        
        return answer
        
    except Exception as e:
        logging.exception("Failed to query RAG system.")
        error_msg = "I encountered an unexpected error while processing your request. The issue has been logged."
        if repo_full_name:
            # Return current usage stats even on error
            try:
                usage_stats = await quota_manager.get_usage_stats(repo_full_name)
            except:
                usage_stats = {"tokens_remaining": 0, "tokens_used": 0}
            return error_msg, usage_stats
        return error_msg

async def get_collection_info(qa_chain_dict: dict) -> Dict[str, Any]:
    """
    Get information about the current collection.
    """
    if not qa_chain_dict or "chroma_client" not in qa_chain_dict:
        return {"error": "ChromaDB client not available"}
    
    try:
        collection_name = qa_chain_dict.get("collection_name", "unknown")
        chroma_client = qa_chain_dict["chroma_client"]
        collection = chroma_client.get_collection(collection_name)
        
        return {
            "collection_name": collection_name,
            "document_count": collection.count(),
            "metadata": collection.metadata
        }
    except Exception as e:
        logging.exception("Failed to get collection info.")
        return {"error": f"Failed to get collection info: {str(e)}"}

async def delete_collection(collection_name: str, persist_directory: str) -> bool:
    """
    Delete a ChromaDB collection and its data.
    """
    try:
        client = _setup_persistent_chromadb(persist_directory)
        client.delete_collection(collection_name)
        logging.info(f"Successfully deleted collection: {collection_name}")
        return True
    except Exception as e:
        logging.exception(f"Failed to delete collection {collection_name}")
        return False

async def list_collections(persist_directory: str) -> List[Dict[str, Any]]:
    """
    List all available collections in the ChromaDB instance.
    """
    try:
        client = _setup_persistent_chromadb(persist_directory)
        collections = client.list_collections()
        
        result = []
        for collection in collections:
            result.append({
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            })
        
        return result
    except Exception as e:
        logging.exception("Failed to list collections")
        return []