import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from app.config import settings
import os
import chromadb
from chromadb.config import Settings as ChromaSettings
import hashlib
import shutil
import logging
import asyncio
from app.core.quota_manager import quota_manager

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
        
    except Exception:
        # Collection doesn't exist, create it
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        logging.info(f"Created new collection: {collection_name}")
    
    return collection

class TokenCounterCallbackHandler:
    """Callback handler to count tokens used in LLM calls."""
    def __init__(self):
        self.total_tokens = 0
    
    def on_llm_start(self, *args, **kwargs):
        pass
    
    def on_llm_end(self, response, *args, **kwargs):
        # Gemini doesn't provide token counts directly, estimate based on text length
        # Rough estimate: 4 chars = 1 token for English text
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
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " "],
            keep_separator=True,
            add_start_index=True
        )
        split_docs = text_splitter.split_documents(documents)
        logging.info(f"Split into {len(split_docs)} chunks.")
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=gemini_api_key,
            model="models/embedding-001"
        )
        
        # Set up persistent ChromaDB client
        chroma_client = _setup_persistent_chromadb(chroma_persist_dir)
        
        # Get or create collection
        collection = _get_or_create_collection(
            chroma_client,
            collection_name,
            embeddings,
            reset_collection=reset_collection
        )
        
        # Create vector store with persistent collection
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=chroma_persist_dir
        )
        
        # Add documents if collection is empty or we're resetting
        if reset_collection or collection.count() == 0:
            logging.info("Adding documents to vector store...")
            vectorstore.add_documents(split_docs)
            logging.info(f"Added {len(split_docs)} document chunks to collection.")
        else:
            logging.info(f"Using existing collection with {collection.count()} documents.")
        
        # Create retriever with improved configuration
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 6,  # Return top 6 most relevant chunks
                "fetch_k": 20,  # Fetch top 20 then filter to 6
            }
        )
        
        # Initialize conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
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
        
        # Create conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={
                "prompt": _create_custom_prompt()
            }
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
        
    except Exception as e:
        logging.exception("Failed to initialize RAG system.")
        return {"error": f"Failed to initialize RAG system: {str(e)}"}

def _create_custom_prompt():
    """Create a custom prompt template for better responses."""
    from langchain.prompts import PromptTemplate
    
    template = """You are a helpful AI assistant for a GitHub repository. Use the following pieces of context to answer the question at the end. 

Context from repository (code, documentation, issues, and discussions):
{context}

Chat History:
{chat_history}

Current Question: {question}

Instructions:
1. Provide accurate, helpful responses based on the repository context
2. Reference specific files, issues, or code sections when relevant
3. If you don't know something, say so clearly
4. For code questions, provide examples when possible
5. Be concise but thorough
6. If suggesting solutions, explain the reasoning

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"]
    )

async def query_rag_system(
    qa_chain_dict: dict,
    query: str,
    repo_full_name: Optional[str] = None
) -> Union[str, Tuple[str, Dict]]:
    """Query the RAG system with quota management."""
    if not qa_chain_dict or "qa_chain" not in qa_chain_dict:
        logging.error("qa_chain_dict missing 'qa_chain'.")
        return "RAG system not initialized."
    if not query or not isinstance(query, str):
        logging.error("Query must be a non-empty string.")
        return "Query must be a non-empty string."
    
    try:
        # Check quota before proceeding
        if repo_full_name:
            if not await quota_manager.check_quota(repo_full_name):
                return "API quota exceeded. Please try again later."
        
        qa_chain = qa_chain_dict["qa_chain"]
        token_counter = qa_chain_dict.get("token_counter")
        
        # Reset token counter
        if token_counter:
            token_counter.total_tokens = 0
        
        # Execute query
        result = await qa_chain.ainvoke({"question": query})
        
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
        return f"I encountered an error while processing your question: {str(e)}"

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