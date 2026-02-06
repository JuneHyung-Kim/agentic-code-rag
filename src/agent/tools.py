"""Tool registry, singleton management, and dispatch for the agent.

Centralises all tool-related logic so that nodes.py only needs to call
``tool_registry.execute(name, input)`` without knowing about individual tools.
"""

import logging
from typing import Any, Callable, Dict, Optional

from config import config
from tools.search_tool import SearchTool
from tools.structure import FileSystemTools
from tools.related import RelatedCodeTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Manages tool metadata, singleton instances, and dispatch."""

    def __init__(self) -> None:
        self._instances: Dict[str, Any] = {}
        self._descriptors: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, Callable[..., str]] = {}
        self._register_builtins()

    # -- Registration helpers --------------------------------------------------

    def _register_builtins(self) -> None:
        """Register the built-in tool set."""
        self.register(
            name="search_codebase",
            description=(
                "Search the codebase using hybrid semantic + keyword search. "
                "Use for finding code by concepts, function names, or keywords."
            ),
            parameters={
                "query": "The search query string",
                "n_results": "(optional) Number of results to return, default 5",
            },
            handler=self._handle_search,
        )
        self.register(
            name="read_file",
            description="Read the contents of a specific file. Use when you know the exact file path.",
            parameters={"file_path": "The path to the file to read"},
            handler=self._handle_read_file,
        )
        self.register(
            name="list_directory",
            description="List files and subdirectories in a directory. Use to explore project structure.",
            parameters={"path": "The directory path to list (use '.' for root)"},
            handler=self._handle_list_dir,
        )
        self.register(
            name="get_callers",
            description="Find functions that call the given function. Use to understand who uses a function.",
            parameters={"function_name": "The name of the function to find callers for"},
            handler=self._handle_get_callers,
        )
        self.register(
            name="get_callees",
            description="Find functions that the given function calls. Use to understand dependencies.",
            parameters={"function_name": "The name of the function to find callees for"},
            handler=self._handle_get_callees,
        )
        self.register(
            name="finish",
            description=(
                "Signal that you have gathered enough information for the current plan step. "
                "Use when observations provide sufficient context."
            ),
            parameters={"summary": "A brief summary of what was found"},
            handler=self._handle_finish,
        )

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, str],
        handler: Callable[..., str],
    ) -> None:
        """Register a tool with its metadata and handler."""
        self._descriptors[name] = {
            "description": description,
            "parameters": parameters,
        }
        self._handlers[name] = handler

    # -- Tool instance singletons ---------------------------------------------

    def _get_search_tool(self) -> SearchTool:
        if "search" not in self._instances:
            self._instances["search"] = SearchTool()
        return self._instances["search"]

    def _get_fs_tool(self) -> FileSystemTools:
        if "fs" not in self._instances:
            self._instances["fs"] = FileSystemTools(config.project_root)
        return self._instances["fs"]

    def _get_related_tool(self) -> RelatedCodeTool:
        if "related" not in self._instances:
            self._instances["related"] = RelatedCodeTool()
        return self._instances["related"]

    def reset(self) -> None:
        """Reset all tool singletons. Useful for testing or when project_root changes."""
        self._instances.clear()

    # -- Handlers --------------------------------------------------------------

    def _handle_search(self, tool_input: Dict[str, Any]) -> str:
        query = tool_input.get("query", "")
        n_results = tool_input.get("n_results", 5)
        return self._get_search_tool().search_codebase(query, n_results=n_results)

    def _handle_read_file(self, tool_input: Dict[str, Any]) -> str:
        return self._get_fs_tool().read_file(tool_input.get("file_path", ""))

    def _handle_list_dir(self, tool_input: Dict[str, Any]) -> str:
        return self._get_fs_tool().list_dir(tool_input.get("path", "."))

    def _handle_get_callers(self, tool_input: Dict[str, Any]) -> str:
        return self._get_related_tool().get_callers(tool_input.get("function_name", ""))

    def _handle_get_callees(self, tool_input: Dict[str, Any]) -> str:
        return self._get_related_tool().get_callees(tool_input.get("function_name", ""))

    def _handle_finish(self, tool_input: Dict[str, Any]) -> str:
        summary = tool_input.get("summary", "Research complete.")
        return f"[FINISH] {summary}"

    # -- Public API ------------------------------------------------------------

    def format_for_prompt(self) -> str:
        """Format all registered tools as a human-readable string for LLM prompts."""
        lines = []
        for name, info in self._descriptors.items():
            params = ", ".join(f"{k}: {v}" for k, v in info["parameters"].items())
            lines.append(f"- {name}({params}): {info['description']}")
        return "\n".join(lines)

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool by name. Returns the result string or an error message."""
        handler = self._handlers.get(tool_name)
        if handler is None:
            return f"Unknown tool: {tool_name}. Available tools: {list(self._descriptors.keys())}"
        try:
            return handler(tool_input)
        except Exception as e:
            return f"Tool execution error ({tool_name}): {e}"


# Module-level singleton
tool_registry = ToolRegistry()
