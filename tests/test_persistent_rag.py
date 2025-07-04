"""
Tests for persistent ChromaDB functionality and collection management.
"""
import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.rag_system import (
    _sanitize_collection_name,
    _setup_persistent_chromadb,
    initialize_rag_system,
    get_collection_info,
    delete_collection,
    list_collections
)

class MockEmbeddingFunction:
    """Mock embedding function that follows ChromaDB's interface."""
    def __call__(self, input):
        # Return mock embeddings - list of lists representing vector embeddings
        if isinstance(input, list):
            return [[0.1, 0.2, 0.3] for _ in input]
        else:
            return [[0.1, 0.2, 0.3]]

class TestPersistentChromaDB:
    
    def test_sanitize_collection_name(self):
        """Test collection name sanitization."""
        # Basic cases
        assert _sanitize_collection_name("owner/repo") == "owner_repo"
        assert _sanitize_collection_name("my-org/my-repo") == "my_org_my_repo"
        assert _sanitize_collection_name("user123/project-456") == "user123_project_456"
        
        # Special characters
        assert _sanitize_collection_name("owner/repo@2024") == "owner_repo_2024"
        assert _sanitize_collection_name("org/repo-name.git") == "org_repo_name_git"
        
        # Long names
        long_name = "very-long-organization-name/very-long-repository-name-that-exceeds-limits"
        result = _sanitize_collection_name(long_name)
        assert len(result) <= 63
        assert result.startswith("very_long_organization_name_very_long_repository")
        
        # Short names
        assert len(_sanitize_collection_name("a/b")) >= 3
        
        # Names starting with non-alphanumeric
        assert _sanitize_collection_name("-org/repo").startswith("repo_")

    def test_setup_persistent_chromadb(self):
        """Test ChromaDB persistent client setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = _setup_persistent_chromadb(temp_dir)
            assert client is not None
            assert os.path.exists(temp_dir)
            
            # Verify client is persistent
            assert hasattr(client, 'list_collections')

    @pytest.mark.asyncio
    async def test_initialize_rag_system_with_persistence(self):
        """Test RAG system initialization with persistent storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock documents
            documents_data = [
                {
                    "content": "This is a test document about Python programming.",
                    "metadata": {"type": "test", "source": "test.py"}
                },
                {
                    "content": "This document covers FastAPI development.",
                    "metadata": {"type": "test", "source": "api.py"}
                }
            ]
            
            # Mock Gemini API and other components
            with patch('app.core.rag_system.GoogleGenerativeAIEmbeddings') as mock_embeddings, \
                 patch('app.core.rag_system.ChatGoogleGenerativeAI') as mock_llm, \
                 patch('app.core.rag_system.ConversationalRetrievalChain') as mock_chain, \
                 patch('app.core.rag_system.Chroma') as mock_chroma:
                
                # Mock embeddings to return proper embedding function
                mock_embeddings_instance = MockEmbeddingFunction()
                mock_embeddings.return_value = mock_embeddings_instance
                
                # Mock Chroma vectorstore
                mock_vectorstore = Mock()
                mock_vectorstore.add_documents = Mock()
                mock_vectorstore.as_retriever = Mock(return_value=Mock())
                mock_chroma.return_value = mock_vectorstore
                
                # Mock LLM
                mock_llm_instance = Mock()
                mock_llm.return_value = mock_llm_instance
                
                # Mock chain
                mock_chain_instance = Mock()
                mock_chain.from_llm.return_value = mock_chain_instance
                
                # Test initialization
                result = await initialize_rag_system(
                    documents_data=documents_data,
                    gemini_api_key="test_api_key",
                    chroma_persist_dir=temp_dir,
                    collection_name="test_collection"
                )
                
                # Verify success
                assert "error" not in result
                assert "qa_chain" in result
                assert "memory" in result
                assert "collection_name" in result
                assert "vectorstore" in result
                assert "chroma_client" in result

    @pytest.mark.asyncio
    async def test_collection_lifecycle(self):
        """Test complete collection lifecycle: create, info, delete."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collection_name = "test_lifecycle_collection"
            
            # Mock components for RAG initialization
            with patch('app.core.rag_system.GoogleGenerativeAIEmbeddings') as mock_embeddings, \
                 patch('app.core.rag_system.ChatGoogleGenerativeAI') as mock_llm, \
                 patch('app.core.rag_system.ConversationalRetrievalChain') as mock_chain, \
                 patch('app.core.rag_system.Chroma') as mock_chroma:
                
                # Setup mocks
                mock_embeddings.return_value = MockEmbeddingFunction()
                mock_llm.return_value = Mock()
                mock_chain.from_llm.return_value = Mock()
                
                mock_vectorstore = Mock()
                mock_vectorstore.add_documents = Mock()
                mock_vectorstore.as_retriever = Mock(return_value=Mock())
                mock_chroma.return_value = mock_vectorstore
                
                # 1. Create collection
                documents_data = [{
                    "content": "Test document for lifecycle testing.",
                    "metadata": {"type": "test"}
                }]
                
                rag_result = await initialize_rag_system(
                    documents_data=documents_data,
                    gemini_api_key="test_api_key",
                    chroma_persist_dir=temp_dir,
                    collection_name=collection_name
                )
                
                assert "error" not in rag_result
                
                # 2. Get collection info (mock the client and collection)
                mock_collection = Mock()
                mock_collection.count.return_value = 1
                mock_collection.metadata = {"test": "data"}
                
                rag_result["chroma_client"] = Mock()
                rag_result["chroma_client"].get_collection.return_value = mock_collection
                
                info = await get_collection_info(rag_result)
                assert info["collection_name"] == collection_name
                assert "document_count" in info
                
                # 3. List collections
                collections = await list_collections(temp_dir)
                assert isinstance(collections, list)
                
                # 4. Delete collection
                success = await delete_collection(collection_name, temp_dir)
                assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_existing_collection_reuse(self):
        """Test that existing collections are properly reused."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collection_name = "test_reuse_collection"
            
            with patch('app.core.rag_system.GoogleGenerativeAIEmbeddings') as mock_embeddings, \
                 patch('app.core.rag_system.ChatGoogleGenerativeAI') as mock_llm, \
                 patch('app.core.rag_system.ConversationalRetrievalChain') as mock_chain, \
                 patch('app.core.rag_system.Chroma') as mock_chroma:
                
                # Setup mocks
                mock_embeddings.return_value = MockEmbeddingFunction()
                mock_llm.return_value = Mock()
                mock_chain.from_llm.return_value = Mock()
                
                mock_vectorstore = Mock()
                mock_vectorstore.add_documents = Mock()
                mock_vectorstore.as_retriever = Mock(return_value=Mock())
                mock_chroma.return_value = mock_vectorstore
                
                documents_data = [{
                    "content": "Initial document.",
                    "metadata": {"type": "test"}
                }]
                
                # First initialization
                result1 = await initialize_rag_system(
                    documents_data=documents_data,
                    gemini_api_key="test_api_key",
                    chroma_persist_dir=temp_dir,
                    collection_name=collection_name
                )
                
                assert "error" not in result1
                
                # Second initialization with same collection name (should reuse)
                result2 = await initialize_rag_system(
                    documents_data=documents_data,
                    gemini_api_key="test_api_key",
                    chroma_persist_dir=temp_dir,
                    collection_name=collection_name,
                    reset_collection=False
                )
                
                assert "error" not in result2
                assert result2["collection_name"] == collection_name

    @pytest.mark.asyncio
    async def test_collection_reset(self):
        """Test collection reset functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collection_name = "test_reset_collection"
            
            with patch('app.core.rag_system.GoogleGenerativeAIEmbeddings') as mock_embeddings, \
                 patch('app.core.rag_system.ChatGoogleGenerativeAI') as mock_llm, \
                 patch('app.core.rag_system.ConversationalRetrievalChain') as mock_chain, \
                 patch('app.core.rag_system.Chroma') as mock_chroma:
                
                # Setup mocks
                mock_embeddings.return_value = MockEmbeddingFunction()
                mock_llm.return_value = Mock()
                mock_chain.from_llm.return_value = Mock()
                
                mock_vectorstore = Mock()
                mock_vectorstore.add_documents = Mock()
                mock_vectorstore.as_retriever = Mock(return_value=Mock())
                mock_chroma.return_value = mock_vectorstore
                
                documents_data = [{
                    "content": "Original document.",
                    "metadata": {"type": "test"}
                }]
                
                # First initialization
                result1 = await initialize_rag_system(
                    documents_data=documents_data,
                    gemini_api_key="test_api_key",
                    chroma_persist_dir=temp_dir,
                    collection_name=collection_name
                )
                
                assert "error" not in result1
                
                # Reset collection with new data
                new_documents_data = [{
                    "content": "New document after reset.",
                    "metadata": {"type": "test"}
                }]
                
                result2 = await initialize_rag_system(
                    documents_data=new_documents_data,
                    gemini_api_key="test_api_key",
                    chroma_persist_dir=temp_dir,
                    collection_name=collection_name,
                    reset_collection=True
                )
                
                assert "error" not in result2
                assert result2["collection_name"] == collection_name

    @pytest.mark.asyncio
    async def test_invalid_documents_data(self):
        """Test handling of invalid documents data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test empty documents
            result1 = await initialize_rag_system(
                documents_data=[],
                gemini_api_key="test_api_key",
                chroma_persist_dir=temp_dir
            )
            assert "error" in result1
            
            # Test invalid document structure
            result2 = await initialize_rag_system(
                documents_data=[{"invalid": "structure"}],
                gemini_api_key="test_api_key",
                chroma_persist_dir=temp_dir
            )
            assert "error" in result2
            
            # Test non-list input
            result3 = await initialize_rag_system(
                documents_data="not a list",
                gemini_api_key="test_api_key",
                chroma_persist_dir=temp_dir
            )
            assert "error" in result3

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        """Test handling of missing API key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_data = [{
                "content": "Test document.",
                "metadata": {"type": "test"}
            }]
            
            result = await initialize_rag_system(
                documents_data=documents_data,
                gemini_api_key="",  # Empty API key
                chroma_persist_dir=temp_dir
            )
            
            assert "error" in result
            assert "Missing Gemini API key" in result["error"]

    @pytest.mark.asyncio
    async def test_error_handling_in_collection_operations(self):
        """Test error handling in collection operations."""
        # Test with non-existent directory
        non_existent_dir = "/non/existent/path"
        
        # This should handle the error gracefully
        collections = await list_collections(non_existent_dir)
        assert collections == []
        
        # Test deleting non-existent collection
        success = await delete_collection("non_existent", non_existent_dir)
        assert success is False

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 