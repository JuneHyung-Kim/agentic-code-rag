"""Shared test configuration and fixtures for agent tests."""

import sys
import os

# Add src/ to sys.path so that agent, tools, config modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from agent.graph import define_graph


@pytest.fixture
def compiled_graph():
    """Return a compiled LangGraph workflow (no LLM calls)."""
    return define_graph()


@pytest.fixture
def graph_topology(compiled_graph):
    """Return the DrawableGraph for topology inspection."""
    return compiled_graph.get_graph()
