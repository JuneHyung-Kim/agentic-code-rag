from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# TODO: [Future Plan] Implement 'File Context Chunk'
# Currently, we index at the symbol level (functions/classes).
# In the future, consider adding a 'file_summary' node that captures:
# - Module-level docstrings
# - Global variables summary
# - Full list of imports/dependencies
# This will improve answers for "What does this file do?" type queries.

@dataclass
class CodeNode:
    """Source code entity representation for indexing."""
    
    # Basic Identification
    type: str                # 'function', 'class', 'struct', 'method', 'enum', 'typedef', 'macro', 'global_var', 'function_decl'
    name: str                # Identifier name
    file_path: str
    
    # Location
    start_line: int          # 0-indexed
    end_line: int            # 0-indexed
    
    # Content
    content: str             # Raw source code
    language: str            # 'python', 'c', 'cpp'
    
    # Contextual Metadata (Key for RAG quality)
    docstring: Optional[str] = None
    arguments: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    signature: Optional[str] = None
    
    # Structural Relationships
    parent_name: Optional[str] = None    # e.g., class name
    
    # Dependencies
    imports: List[str] = field(default_factory=list)    # import/include statements
    
    # Extension
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def node_id(self) -> str:
        """Generate unique ID: filepath:name:line"""
        return f"{self.file_path}:{self.name}:{self.start_line}"
