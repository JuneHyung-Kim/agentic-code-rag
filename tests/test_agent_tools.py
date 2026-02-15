"""Tests for agent tool registration — no LLM calls required."""

import pytest

from langchain_core.tools import BaseTool

from agent.tools import get_tools, reset_tools


EXPECTED_TOOL_NAMES = {
    "search_codebase",
    "read_file",
    "find_files",
    "grep_codebase",
    "list_directory",
    "get_callers",
    "get_callees",
    "get_symbol_definition",
    "get_call_chain",
    "get_module_summary",
}


class TestToolRegistration:
    """Verify tool definitions and singleton behavior."""

    def test_get_tools_returns_five_tools(self):
        tools = get_tools()
        assert len(tools) == 10

    def test_tool_names(self):
        tools = get_tools()
        names = {t.name for t in tools}
        assert names == EXPECTED_TOOL_NAMES

    def test_tools_are_base_tool_instances(self):
        tools = get_tools()
        for t in tools:
            assert isinstance(t, BaseTool), f"{t.name} is not a BaseTool"

    def test_tool_has_description(self):
        tools = get_tools()
        for t in tools:
            assert t.description, f"{t.name} has no description"
            assert len(t.description) > 10, f"{t.name} description too short"

    def test_reset_tools(self):
        """reset_tools() should clear the singleton cache without error."""
        reset_tools()
        # After reset, get_tools() should still work (recreates singletons lazily)
        tools = get_tools()
        assert len(tools) == 10
        reset_tools()  # clean up
