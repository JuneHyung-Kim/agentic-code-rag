from typing import List
from .base_strategy import BaseStrategy
from ..schema import CodeNode
from ..storage.graph_store import GraphStore
from utils.logger import logger
import os

class GraphStrategy(BaseStrategy):
    def __init__(self, graph_store: GraphStore, project_root: str):
        self.graph_store = graph_store
        self.project_root = project_root

    def index(self, file_path: str, nodes: List[CodeNode]) -> bool:
        if not nodes:
            return True
        
        try:
            # 1. Add nodes
            for node in nodes:
                if not node.id:
                    continue
                    
                self.graph_store.add_node(
                    node.id,
                    file_path=os.path.abspath(file_path),
                    name=node.name,
                    type=node.type,
                    start_line=node.start_line
                )
                
                # 2. Add structural edges (parent-child)
                # ...
                
                # 3. Add Call Graph edges
                if node.function_calls:
                    for called_func_name in node.function_calls:
                        # Logic to resolve `called_func_name` to a node_id is complex.
                        # It requires symbol resolution (which file is it defined in?).
                        # We don't have that info yet.
                        # We can store a "soft edge" or "pending edge" pointing to a name.
                        # Or we add an edge to a "symbol_node" named `clean_func_name`.
                        self.graph_store.add_edge(node.id, called_func_name, type="calls_by_name")

            return True
        except Exception as e:
            logger.error(f"GraphStrategy failed to index {file_path}: {e}")
            return False

    def delete(self, file_path: str) -> bool:
        try:
            self.graph_store.delete_by_file(os.path.abspath(file_path))
            return True
        except Exception as e:
            logger.error(f"GraphStrategy failed to delete {file_path}: {e}")
            return False
