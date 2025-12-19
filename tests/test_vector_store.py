"""
Unit tests for vector store
"""
import pytest
import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from indexing.vector_store import VectorStore


@pytest.fixture
def temp_db():
    """Create a temporary database directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_vector_store_initialization(temp_db):
    """Test vector store can be initialized"""
    store = VectorStore(collection_name="test", persist_path=temp_db)
    assert store is not None
    assert store.collection is not None


def test_add_and_query_documents(temp_db):
    """Test adding documents and querying them"""
    store = VectorStore(collection_name="test", persist_path=temp_db)
    
    # Add test documents
    documents = [
        "def hello_world(): print('Hello')",
        "def add(a, b): return a + b",
        "class Calculator: pass"
    ]
    metadatas = [
        {"file_path": "test1.py", "name": "hello_world", "type": "function", "start_line": 1, "end_line": 1, "language": "python"},
        {"file_path": "test2.py", "name": "add", "type": "function", "start_line": 1, "end_line": 1, "language": "python"},
        {"file_path": "test3.py", "name": "Calculator", "type": "class", "start_line": 1, "end_line": 1, "language": "python"}
    ]
    ids = ["doc1", "doc2", "doc3"]
    
    store.add_documents(documents, metadatas, ids)
    
    # Query for "add numbers"
    results = store.query("add numbers", n_results=2)
    
    assert results is not None
    assert 'documents' in results
    assert len(results['documents'][0]) > 0
    # Should find the add function
    found_add = any('add' in doc for doc in results['documents'][0])
    assert found_add, "Should find 'add' function when searching for 'add numbers'"


def test_query_empty_store(temp_db):
    """Test querying an empty store"""
    store = VectorStore(collection_name="test", persist_path=temp_db)
    results = store.query("test query", n_results=5)
    
    assert results is not None
    assert 'documents' in results
    # Should return empty results
    assert len(results['documents'][0]) == 0


def test_add_empty_documents(temp_db):
    """Test adding empty document list"""
    store = VectorStore(collection_name="test", persist_path=temp_db)
    # Should not raise error
    store.add_documents([], [], [])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
