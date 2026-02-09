"""Tests for profiling.builder.ProfileBuilder."""

import pytest
from unittest.mock import MagicMock

import networkx as nx

from profiling.builder import ProfileBuilder


def _make_vector_store(metadatas=None):
    """Create a mock VectorStore with get_all_documents()."""
    store = MagicMock()
    if metadatas is None:
        store.get_all_documents.return_value = {}
    else:
        store.get_all_documents.return_value = {
            "ids": [f"id_{i}" for i in range(len(metadatas))],
            "metadatas": metadatas,
            "documents": [f"doc_{i}" for i in range(len(metadatas))],
        }
    return store


def _make_graph_store(nodes=None, edges=None):
    """Create a mock GraphStore with a real NetworkX DiGraph."""
    store = MagicMock()
    g = nx.DiGraph()
    for n_id, attrs in (nodes or []):
        g.add_node(n_id, **attrs)
    for src, tgt in (edges or []):
        g.add_edge(src, tgt)
    store.graph = g
    return store


PROJECT = "/home/user/project"


class TestEmptyStores:

    def test_empty_vector_store(self):
        vs = _make_vector_store()
        gs = _make_graph_store()
        profile = ProfileBuilder(vs, gs, PROJECT).build()

        assert profile.project_root == PROJECT
        assert profile.total_files == 0
        assert profile.total_symbols == 0
        assert profile.language_stats == {}
        assert profile.module_map == []
        assert profile.entry_points == []
        assert profile.graph_stats.total_nodes == 0


class TestLanguageStats:

    def test_counts_by_language(self):
        metas = [
            {"file_path": f"{PROJECT}/src/foo.py", "project_root": PROJECT, "language": "python", "name": "foo", "type": "function"},
            {"file_path": f"{PROJECT}/src/bar.py", "project_root": PROJECT, "language": "python", "name": "bar", "type": "function"},
            {"file_path": f"{PROJECT}/src/baz.c", "project_root": PROJECT, "language": "c", "name": "baz", "type": "function"},
        ]
        vs = _make_vector_store(metas)
        gs = _make_graph_store()
        profile = ProfileBuilder(vs, gs, PROJECT).build()

        assert "python" in profile.language_stats
        assert "c" in profile.language_stats
        assert profile.language_stats["python"].file_count == 2
        assert profile.language_stats["python"].symbol_count == 2
        assert profile.language_stats["c"].file_count == 1
        assert ".py" in profile.language_stats["python"].extensions
        assert ".c" in profile.language_stats["c"].extensions


class TestEntryPointDetection:

    def test_main_function(self):
        metas = [
            {"file_path": f"{PROJECT}/src/app.py", "project_root": PROJECT, "language": "python", "name": "main", "type": "function"},
        ]
        vs = _make_vector_store(metas)
        gs = _make_graph_store()
        profile = ProfileBuilder(vs, gs, PROJECT).build()

        assert len(profile.entry_points) >= 1
        reasons = [ep.reason for ep in profile.entry_points]
        assert any("main function" in r for r in reasons)

    def test_entry_point_filenames(self):
        metas = [
            {"file_path": f"{PROJECT}/cli.py", "project_root": PROJECT, "language": "python", "name": "run", "type": "function"},
            {"file_path": f"{PROJECT}/__main__.py", "project_root": PROJECT, "language": "python", "name": "", "type": "module"},
        ]
        vs = _make_vector_store(metas)
        gs = _make_graph_store()
        profile = ProfileBuilder(vs, gs, PROJECT).build()

        assert len(profile.entry_points) == 2
        paths = [ep.file_path for ep in profile.entry_points]
        assert f"{PROJECT}/cli.py" in paths
        assert f"{PROJECT}/__main__.py" in paths


class TestModuleMap:

    def test_grouped_by_file_sorted_by_line(self):
        metas = [
            {"file_path": f"{PROJECT}/src/a.py", "project_root": PROJECT, "language": "python", "name": "func_b", "type": "function", "start_line": 20, "signature": "def func_b()"},
            {"file_path": f"{PROJECT}/src/a.py", "project_root": PROJECT, "language": "python", "name": "func_a", "type": "function", "start_line": 5, "signature": "def func_a()"},
            {"file_path": f"{PROJECT}/src/b.py", "project_root": PROJECT, "language": "python", "name": "ClassX", "type": "class", "start_line": 1, "signature": "class ClassX"},
        ]
        vs = _make_vector_store(metas)
        gs = _make_graph_store()
        profile = ProfileBuilder(vs, gs, PROJECT).build()

        assert len(profile.module_map) == 2
        # First file should be a.py (sorted by path)
        a_file = profile.module_map[0]
        assert "a.py" in a_file.relative_path
        assert a_file.symbols[0].name == "func_a"  # line 5 first
        assert a_file.symbols[1].name == "func_b"  # line 20 second


class TestGraphStats:

    def test_in_out_degree(self):
        nodes = [
            ("A", {"file_path": f"{PROJECT}/a.py", "name": "A"}),
            ("B", {"file_path": f"{PROJECT}/b.py", "name": "B"}),
            ("C", {"file_path": f"{PROJECT}/c.py", "name": "C"}),
        ]
        edges = [("A", "B"), ("A", "C"), ("C", "B")]
        vs = _make_vector_store([
            {"file_path": f"{PROJECT}/a.py", "project_root": PROJECT, "language": "python", "name": "A", "type": "function"},
        ])
        gs = _make_graph_store(nodes, edges)
        profile = ProfileBuilder(vs, gs, PROJECT).build()

        assert profile.graph_stats.total_nodes == 3
        assert profile.graph_stats.total_edges == 3
        # B has highest in-degree (2)
        assert profile.graph_stats.most_called[0]["name"] == "B"
        # A has highest out-degree (2)
        assert profile.graph_stats.most_calling[0]["name"] == "A"

    def test_empty_graph(self):
        vs = _make_vector_store([
            {"file_path": f"{PROJECT}/a.py", "project_root": PROJECT, "language": "python", "name": "f", "type": "function"},
        ])
        gs = _make_graph_store()
        profile = ProfileBuilder(vs, gs, PROJECT).build()
        assert profile.graph_stats.total_nodes == 0
        assert profile.graph_stats.total_edges == 0


class TestDirectoryTree:

    def test_tree_structure(self):
        metas = [
            {"file_path": f"{PROJECT}/src/a.py", "project_root": PROJECT, "language": "python", "name": "f", "type": "function"},
            {"file_path": f"{PROJECT}/src/sub/b.py", "project_root": PROJECT, "language": "python", "name": "g", "type": "function"},
            {"file_path": f"{PROJECT}/README.md", "project_root": PROJECT, "language": "text", "name": "doc", "type": "file"},
        ]
        vs = _make_vector_store(metas)
        gs = _make_graph_store()
        profile = ProfileBuilder(vs, gs, PROJECT).build()

        tree = profile.directory_tree
        assert tree is not None
        child_names = {c.name for c in tree.children}
        assert "src" in child_names
        assert "README.md" in child_names
