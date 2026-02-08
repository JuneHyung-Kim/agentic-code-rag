"""LangGraph-native tool definitions using @tool decorator.

Provides get_tools() factory that returns List[BaseTool] for use with
bind_tools() and ToolNode().
"""

import logging
from typing import Any, Dict, List

from langchain_core.tools import BaseTool, tool

from config import config
from tools.search_tool import SearchTool
from tools.structure import FileSystemTools
from tools.related import RelatedCodeTool
from tools.symbol import SymbolTool

logger = logging.getLogger(__name__)

# -- Singleton tool instances -------------------------------------------------

_instances: Dict[str, Any] = {}


def _get_search_tool() -> SearchTool:
    if "search" not in _instances:
        _instances["search"] = SearchTool()
    return _instances["search"]


def _get_fs_tool() -> FileSystemTools:
    if "fs" not in _instances:
        _instances["fs"] = FileSystemTools(config.project_root)
    return _instances["fs"]


def _get_related_tool() -> RelatedCodeTool:
    if "related" not in _instances:
        _instances["related"] = RelatedCodeTool()
    return _instances["related"]


def _get_symbol_tool() -> SymbolTool:
    if "symbol" not in _instances:
        _instances["symbol"] = SymbolTool()
    return _instances["symbol"]


def reset_tools() -> None:
    """Reset all tool singletons. Useful for testing or when project_root changes."""
    _instances.clear()


# -- @tool definitions --------------------------------------------------------


@tool
def search_codebase(query: str, n_results: int = 5) -> str:
    """Search the codebase using hybrid semantic + keyword search.
    Use for finding code by concepts, function names, or keywords."""
    return _get_search_tool().search_codebase(query, n_results=n_results)


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a specific file. Use when you know the exact file path."""
    return _get_fs_tool().read_file(file_path)


@tool
def list_directory(path: str = ".") -> str:
    """List files and subdirectories in a directory. Use to explore project structure."""
    return _get_fs_tool().list_dir(path)


@tool
def get_callers(function_name: str) -> str:
    """Find functions that call the given function. Use to understand who uses a function."""
    return _get_related_tool().get_callers(function_name)


@tool
def get_callees(function_name: str) -> str:
    """Find functions that the given function calls. Use to understand dependencies."""
    return _get_related_tool().get_callees(function_name)


@tool
def get_symbol_definition(symbol_name: str, symbol_type: str = "") -> str:
    """Look up the exact definition of a symbol by name. Returns source location and code.
    Use when you need the precise definition rather than a semantic search."""
    return _get_symbol_tool().get_symbol_definition(
        symbol_name, symbol_type=symbol_type or None
    )


@tool
def get_call_chain(
    function_name: str, direction: str = "callees", max_depth: int = 3
) -> str:
    """Trace a multi-hop call chain from a function. direction is 'callees' (outgoing)
    or 'callers' (incoming). Use to understand deep call hierarchies."""
    return _get_related_tool().get_call_chain(
        function_name, direction=direction, max_depth=max_depth
    )


@tool
def get_module_summary(path: str) -> str:
    """Get a high-level summary of all symbols in a file or directory.
    Use to understand the structure of a module without reading every file."""
    return _get_symbol_tool().get_module_summary(path)


# -- Public API ---------------------------------------------------------------


def get_tools() -> List[BaseTool]:
    """Return the list of all agent tools for bind_tools() and ToolNode()."""
    return [
        search_codebase,
        read_file,
        list_directory,
        get_callers,
        get_callees,
        get_symbol_definition,
        get_call_chain,
        get_module_summary,
    ]
