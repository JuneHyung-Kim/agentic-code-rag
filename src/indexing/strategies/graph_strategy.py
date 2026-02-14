from typing import Dict, List
from collections import defaultdict
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

    def resolve_edges(self):
        """Resolve bare-name call edges to real node IDs after all files are indexed."""
        graph = self.graph_store.graph

        # Build name → [node_id] lookup from nodes that have a file_path
        name_to_ids: Dict[str, List[str]] = defaultdict(list)
        for node_id, data in graph.nodes(data=True):
            if data.get("file_path"):
                name = data.get("name", "")
                if name:
                    name_to_ids[name].append(node_id)

        # Find all calls_by_name edges
        edges_to_remove = []
        edges_to_add = []
        for u, v, data in graph.edges(data=True):
            if data.get("type") != "calls_by_name":
                continue
            # v is a bare function name
            targets = name_to_ids.get(v, [])
            if targets:
                edges_to_remove.append((u, v))
                for target_id in targets:
                    if target_id != u:  # skip self-calls
                        edges_to_add.append((u, target_id, {"type": "calls"}))

        # Apply changes
        for u, v in edges_to_remove:
            graph.remove_edge(u, v)
            # Remove orphaned bare-name node if it has no remaining edges
            if graph.has_node(v) and not graph.nodes[v].get("file_path"):
                if graph.degree(v) == 0:
                    graph.remove_node(v)

        for u, v, data in edges_to_add:
            if not graph.has_edge(u, v):
                graph.add_edge(u, v, **data)

        resolved = len(edges_to_remove)
        new_edges = len(edges_to_add)
        unresolved = sum(
            1 for _, _, d in graph.edges(data=True) if d.get("type") == "calls_by_name"
        )
        logger.info(
            f"Edge resolution: {resolved} bare-name edges resolved → "
            f"{new_edges} real edges added, {unresolved} unresolved remain"
        )

    def delete(self, file_path: str) -> bool:
        try:
            self.graph_store.delete_by_file(os.path.abspath(file_path))
            return True
        except Exception as e:
            logger.error(f"GraphStrategy failed to delete {file_path}: {e}")
            return False
