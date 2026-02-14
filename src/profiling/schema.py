"""Pydantic data models for codebase profiles."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class LanguageStats(BaseModel):
    file_count: int = 0
    symbol_count: int = 0
    extensions: List[str] = Field(default_factory=list)


class DirectoryNode(BaseModel):
    name: str
    type: str = "directory"  # "directory" or "file"
    children: List["DirectoryNode"] = Field(default_factory=list)


class SymbolSummary(BaseModel):
    name: str
    type: str  # "function", "class", "method", etc.
    start_line: int = 0
    signature: str = ""


class FileSummary(BaseModel):
    relative_path: str
    language: str = ""
    symbols: List[SymbolSummary] = Field(default_factory=list)


class EntryPoint(BaseModel):
    file_path: str
    symbol_name: str = ""
    reason: str = ""


class KeyModule(BaseModel):
    relative_path: str
    symbol_count: int = 0
    total_in_degree: int = 0
    total_out_degree: int = 0
    role: str = ""  # e.g. "most symbols", "most called", "hub"


class GraphStats(BaseModel):
    total_nodes: int = 0
    total_edges: int = 0
    most_called: List[Dict] = Field(default_factory=list)   # highest in-degree
    most_calling: List[Dict] = Field(default_factory=list)  # highest out-degree


class CodebaseProfile(BaseModel):
    project_root: str
    generated_at: str = ""
    profile_version: int = 1
    total_files: int = 0
    total_symbols: int = 0
    language_stats: Dict[str, LanguageStats] = Field(default_factory=dict)
    directory_tree: Optional[DirectoryNode] = None
    module_map: List[FileSummary] = Field(default_factory=list)
    entry_points: List[EntryPoint] = Field(default_factory=list)
    key_modules: List[KeyModule] = Field(default_factory=list)
    graph_stats: GraphStats = Field(default_factory=GraphStats)
    ai_summary: Optional[str] = None
