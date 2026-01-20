"""
Integration tests for end-to-end workflows
"""
import pytest
import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from indexing.indexer import CodeIndexer
from storage.vector_store import VectorStore
from retrieval.search_engine import SearchEngine


@pytest.fixture
def sample_project():
    """Create a sample project directory"""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample Python file
    python_file = os.path.join(temp_dir, "sample.py")
    with open(python_file, 'w') as f:
        f.write('''
def greet(name):
    """Greet a person by name"""
    return f"Hello, {name}!"

class Math:
    """Math utilities"""
    def multiply(self, a, b):
        """Multiply two numbers"""
        return a * b
''')
    
    # Create sample C file
    c_file = os.path.join(temp_dir, "sample.c")
    with open(c_file, 'w') as f:
        f.write('''
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
''')
    
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_db():
    """Create a temporary database directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Windows: ChromaDB may lock files, use ignore_errors
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_index_sample_project(sample_project, temp_db):
    """Test indexing a complete project"""
    # Change the default db path temporarily
    import storage.vector_store
    original_init = storage.vector_store.VectorStore.__init__
    
    def mock_init(self, collection_name="code_chunks", persist_path=None):
        original_init(self, collection_name, persist_path or temp_db)
    
    storage.vector_store.VectorStore.__init__ = mock_init
    
    try:
        indexer = CodeIndexer(sample_project, persist_path=temp_db)
        indexer.index_project()

        # Verify indexing worked by querying (hybrid)
        engine = SearchEngine(VectorStore(persist_path=temp_db))
        results = engine.hybrid_search("greet function", n_results=5)

        assert len(results) > 0
        found_greet = any("greet" in res["content"].lower() for res in results)
        assert found_greet, "Should find 'greet' function after indexing"
        
    finally:
        # Restore original
        storage.vector_store.VectorStore.__init__ = original_init


def test_search_after_indexing(sample_project, temp_db):
    """Test searching for different types of code"""
    import storage.vector_store
    original_init = storage.vector_store.VectorStore.__init__
    
    def mock_init(self, collection_name="code_chunks", persist_path=None):
        original_init(self, collection_name, persist_path or temp_db)
    
        storage.vector_store.VectorStore.__init__ = mock_init
    
    try:
        # Index the project
        indexer = CodeIndexer(sample_project, persist_path=temp_db)
        indexer.index_project()
        
        # Search for different things via hybrid search
        engine = SearchEngine(VectorStore(persist_path=temp_db))
        
        results = engine.hybrid_search("multiply two numbers", n_results=5)
        assert len(results) > 0
        
        results = engine.hybrid_search("recursive factorial", n_results=5)
        assert len(results) > 0
        
        results = engine.hybrid_search("Math class", n_results=5)
        assert len(results) > 0
        
    finally:
        storage.vector_store.VectorStore.__init__ = original_init


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
