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
from indexing.vector_store import VectorStore


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
    shutil.rmtree(temp_dir)


def test_index_sample_project(sample_project, temp_db):
    """Test indexing a complete project"""
    # Change the default db path temporarily
    import indexing.vector_store
    original_init = indexing.vector_store.VectorStore.__init__
    
    def mock_init(self, collection_name="code_chunks", persist_path=None):
        original_init(self, collection_name, persist_path or temp_db)
    
    indexing.vector_store.VectorStore.__init__ = mock_init
    
    try:
        indexer = CodeIndexer(sample_project)
        indexer.index_project()
        
        # Verify indexing worked by querying
        store = VectorStore(persist_path=temp_db)
        results = store.query("greet function", n_results=5)
        
        assert len(results['documents'][0]) > 0
        # Should find the greet function
        found_greet = any('greet' in doc.lower() for doc in results['documents'][0])
        assert found_greet, "Should find 'greet' function after indexing"
        
    finally:
        # Restore original
        indexing.vector_store.VectorStore.__init__ = original_init


def test_search_after_indexing(sample_project, temp_db):
    """Test searching for different types of code"""
    import indexing.vector_store
    original_init = indexing.vector_store.VectorStore.__init__
    
    def mock_init(self, collection_name="code_chunks", persist_path=None):
        original_init(self, collection_name, persist_path or temp_db)
    
    indexing.vector_store.VectorStore.__init__ = mock_init
    
    try:
        # Index the project
        indexer = CodeIndexer(sample_project)
        indexer.index_project()
        
        # Search for different things
        store = VectorStore(persist_path=temp_db)
        
        # Search for math operations
        results = store.query("multiply two numbers", n_results=5)
        assert len(results['documents'][0]) > 0
        
        # Search for recursive function
        results = store.query("recursive factorial", n_results=5)
        assert len(results['documents'][0]) > 0
        
        # Search for class
        results = store.query("Math class", n_results=5)
        assert len(results['documents'][0]) > 0
        
    finally:
        indexing.vector_store.VectorStore.__init__ = original_init


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
