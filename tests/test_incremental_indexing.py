"""
Integration tests for incremental indexing behavior
"""
import os
import sys
import tempfile
import shutil
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from indexing.indexer import CodeIndexer
from indexing.vector_store import VectorStore


def _write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def test_incremental_indexing_update_and_delete():
    temp_project = tempfile.mkdtemp()
    temp_db = tempfile.mkdtemp()
    try:
        sample_path = os.path.join(temp_project, "sample.py")
        _write_file(sample_path, "def hello():\n    return 'hello'\n")

        indexer = CodeIndexer(temp_project, persist_path=temp_db)
        indexer.index_project()

        store = VectorStore(persist_path=temp_db)
        rel_path = os.path.relpath(sample_path, temp_project)

        initial = store.collection.get(where={"file_path": rel_path})
        assert len(initial["ids"]) > 0

        # Modify file content
        _write_file(sample_path, "def goodbye():\n    return 'bye'\n")
        indexer.index_project()
        store = VectorStore(persist_path=temp_db)

        updated = store.collection.get(where={"file_path": rel_path})
        assert len(updated["ids"]) > 0
        assert any("goodbye" in doc for doc in updated["documents"])

        # Verify registry updated
        registry_path = os.path.join(temp_db, "index_registry.json")
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        assert rel_path in registry["files"]

        # Delete file and reindex
        os.remove(sample_path)
        indexer.index_project()
        store = VectorStore(persist_path=temp_db)

        deleted = store.collection.get(where={"file_path": rel_path})
        assert len(deleted["ids"]) == 0
    finally:
        shutil.rmtree(temp_project, ignore_errors=True)
        shutil.rmtree(temp_db, ignore_errors=True)
