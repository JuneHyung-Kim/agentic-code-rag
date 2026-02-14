"""LangGraph-native tool definitions using @tool decorator.

Provides get_tools() factory that returns List[BaseTool] for use with
bind_tools() and ToolNode().

All tools return Tuple[str, dict] with response_format="content_and_artifact".
The first element is LLM-readable text; the second is a structured artifact
with entities and relationships for working memory extraction.
"""

import logging
import re
from typing import Any, Dict, List, Tuple

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


# -- Artifact helpers ---------------------------------------------------------

def _empty_artifact() -> Dict[str, list]:
    """Return an empty artifact dict."""
    return {"entities": [], "relationships": []}


def _parse_search_entities(text: str) -> List[Dict[str, str]]:
    """Extract entity dicts from search result text."""
    entities = []
    # Pattern: "File: /path/to/file.py" lines
    for match in re.finditer(r"File:\s*(.+)", text):
        file_path = match.group(1).strip()
        entities.append({"name": file_path, "type": "file", "location": file_path})
    return entities


def _parse_graph_entities(text: str, relation_type: str) -> Tuple[List[Dict], List[Dict]]:
    """Extract entities and relationships from caller/callee text output."""
    entities = []
    relationships = []
    # Pattern: "  - name (type) in file_path"
    for match in re.finditer(r"-\s+(\S+)\s+\((\w+)\)\s+in\s+(.+)", text):
        name, sym_type, file_path = match.group(1), match.group(2), match.group(3).strip()
        entities.append({"name": name, "type": sym_type, "location": file_path})
    return entities, relationships


def _parse_caller_relationships(function_name: str, text: str) -> List[Dict]:
    """Extract 'calls' relationships from get_callers output."""
    rels = []
    for match in re.finditer(r"-\s+(\S+)\s+\(\w+\)\s+in", text):
        caller = match.group(1)
        rels.append({"source": caller, "type": "calls", "target": function_name})
    return rels


def _parse_callee_relationships(function_name: str, text: str) -> List[Dict]:
    """Extract 'calls' relationships from get_callees output."""
    rels = []
    for match in re.finditer(r"-\s+(\S+)\s+\(\w+\)\s+in", text):
        callee = match.group(1)
        rels.append({"source": function_name, "type": "calls", "target": callee})
    return rels


def _parse_symbol_entities(text: str) -> List[Dict[str, str]]:
    """Extract entity dicts from symbol definition text."""
    entities = []
    current: Dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("Name:"):
            current["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Type:"):
            current["type"] = line.split(":", 1)[1].strip()
        elif line.startswith("File:"):
            current["location"] = line.split(":", 1)[1].strip()
        elif line.startswith("--- Match") and current:
            if current.get("name"):
                entities.append(dict(current))
            current = {}
    if current.get("name"):
        entities.append(dict(current))
    return entities


def _parse_call_chain_relationships(text: str) -> Tuple[List[Dict], List[Dict]]:
    """Extract entities and relationships from call chain text."""
    entities = []
    relationships = []
    # Parse indented entries
    lines = []
    for match in re.finditer(r"(\s*)(\S+)\s+\((\w+)\)\s+in\s+(.+)", text):
        indent = len(match.group(1))
        name = match.group(2)
        sym_type = match.group(3)
        file_path = match.group(4).strip()
        depth = indent // 2
        entities.append({"name": name, "type": sym_type, "location": file_path})
        lines.append((depth, name))

    # Build parent-child relationships from indent levels
    for i in range(1, len(lines)):
        child_depth, child_name = lines[i]
        # Find parent: closest preceding entry with depth == child_depth - 1
        for j in range(i - 1, -1, -1):
            parent_depth, parent_name = lines[j]
            if parent_depth == child_depth - 1:
                relationships.append({"source": parent_name, "type": "calls", "target": child_name})
                break

    return entities, relationships


def _parse_module_entities(text: str) -> List[Dict[str, str]]:
    """Extract entity dicts from module summary text."""
    entities = []
    current_file = ""
    for line in text.splitlines():
        line = line.strip()
        if line.endswith(":") and not line.startswith("["):
            current_file = line.rstrip(":")
        elif line.startswith("["):
            # Pattern: [type] name (Lstart-end) — signature
            match = re.match(r"\[(\w+)\]\s+(\S+)", line)
            if match:
                entities.append({
                    "name": match.group(2),
                    "type": match.group(1),
                    "location": current_file,
                })
    return entities


# -- @tool definitions --------------------------------------------------------


@tool(response_format="content_and_artifact")
def search_codebase(query: str, n_results: int = 5) -> Tuple[str, dict]:
    """Search the codebase using hybrid semantic + keyword search.
    Use for finding code by concepts, function names, or keywords."""
    text = _get_search_tool().search_codebase(query, n_results=n_results)
    entities = _parse_search_entities(text)
    return text, {"entities": entities, "relationships": []}


@tool(response_format="content_and_artifact")
def read_file(file_path: str) -> Tuple[str, dict]:
    """Read the contents of a specific file. Use when you know the exact file path."""
    text = _get_fs_tool().read_file(file_path)
    return text, _empty_artifact()


@tool(response_format="content_and_artifact")
def list_directory(path: str = ".") -> Tuple[str, dict]:
    """List files and subdirectories in a directory. Use to explore project structure."""
    text = _get_fs_tool().list_dir(path)
    return text, _empty_artifact()


@tool(response_format="content_and_artifact")
def get_callers(function_name: str) -> Tuple[str, dict]:
    """Find functions that call the given function. Use to understand who uses a function."""
    text = _get_related_tool().get_callers(function_name)
    entities, _ = _parse_graph_entities(text, "caller")
    relationships = _parse_caller_relationships(function_name, text)
    # Add the target function itself as entity
    if entities:
        entities.insert(0, {"name": function_name, "type": "function", "location": ""})
    return text, {"entities": entities, "relationships": relationships}


@tool(response_format="content_and_artifact")
def get_callees(function_name: str) -> Tuple[str, dict]:
    """Find functions that the given function calls. Use to understand dependencies."""
    text = _get_related_tool().get_callees(function_name)
    entities, _ = _parse_graph_entities(text, "callee")
    relationships = _parse_callee_relationships(function_name, text)
    if entities:
        entities.insert(0, {"name": function_name, "type": "function", "location": ""})
    return text, {"entities": entities, "relationships": relationships}


@tool(response_format="content_and_artifact")
def get_symbol_definition(symbol_name: str, symbol_type: str = "") -> Tuple[str, dict]:
    """Look up the exact definition of a symbol by name. Returns source location and code.
    Use when you need the precise definition rather than a semantic search."""
    text = _get_symbol_tool().get_symbol_definition(
        symbol_name, symbol_type=symbol_type or None
    )
    entities = _parse_symbol_entities(text)
    return text, {"entities": entities, "relationships": []}


@tool(response_format="content_and_artifact")
def get_call_chain(
    function_name: str, direction: str = "callees", max_depth: int = 3
) -> Tuple[str, dict]:
    """Trace a multi-hop call chain from a function. direction is 'callees' (outgoing)
    or 'callers' (incoming). Use to understand deep call hierarchies."""
    text = _get_related_tool().get_call_chain(
        function_name, direction=direction, max_depth=max_depth
    )
    entities, relationships = _parse_call_chain_relationships(text)
    return text, {"entities": entities, "relationships": relationships}


@tool(response_format="content_and_artifact")
def get_module_summary(path: str) -> Tuple[str, dict]:
    """Get a high-level summary of all symbols in a file or directory.
    Use to understand the structure of a module without reading every file."""
    text = _get_symbol_tool().get_module_summary(path)
    entities = _parse_module_entities(text)
    return text, {"entities": entities, "relationships": []}


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
