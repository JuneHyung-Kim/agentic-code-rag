import networkx as nx
from typing import List, Dict, Any, Optional
from utils.logger import logger

class GraphStore:
    """
    In-memory graph store using NetworkX.
    Stores call graphs and file dependencies.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_node(self, node_id: str, **attrs):
        """Add or update a node with attributes."""
        self.graph.add_node(node_id, **attrs)

    def add_edge(self, source: str, target: str, type: str = "calls"):
        """Add an edge between nodes."""
        self.graph.add_edge(source, target, type=type)

    def get_context(self, node_id: str, depth: int = 1) -> List[str]:
        """
        Retrieve context (neighbors) for a given node.
        Returns a list of neighbor node IDs.
        """
        if node_id not in self.graph:
            return []

        # Get successors (called functions) and predecessors (callers)
        # For simple context, we just return immediate neighbors
        neighbors = set()
        try:
            # Outgoing edges (calls)
            neighbors.update(self.graph.successors(node_id))
            # Incoming edges (called by)
            neighbors.update(self.graph.predecessors(node_id))
        except Exception:
            pass

        return list(neighbors)

    def get_callers(self, function_name: str) -> List[Dict[str, Any]]:
        """
        Find functions that call the given function.
        Returns list of caller info with node attributes.
        """
        callers = []
        # Find nodes matching the function name
        target_nodes = [
            n for n in self.graph.nodes()
            if n.endswith(f":{function_name}") or n == function_name
        ]

        for target in target_nodes:
            for caller in self.graph.predecessors(target):
                node_data = self.graph.nodes.get(caller, {})
                callers.append({
                    "node_id": caller,
                    "file_path": node_data.get("file_path", ""),
                    "name": node_data.get("name", caller),
                    "type": node_data.get("type", "unknown")
                })
        return callers

    def get_callees(self, function_name: str) -> List[Dict[str, Any]]:
        """
        Find functions that the given function calls.
        Returns list of callee info with node attributes.
        """
        callees = []
        # Find nodes matching the function name
        source_nodes = [
            n for n in self.graph.nodes()
            if n.endswith(f":{function_name}") or n == function_name
        ]

        for source in source_nodes:
            for callee in self.graph.successors(source):
                node_data = self.graph.nodes.get(callee, {})
                callees.append({
                    "node_id": callee,
                    "file_path": node_data.get("file_path", ""),
                    "name": node_data.get("name", callee),
                    "type": node_data.get("type", "unknown")
                })
        return callees

    def delete_by_file(self, file_path: str):
        """Remove all nodes belonging to a specific file."""
        nodes_to_remove = [
            n for n, data in self.graph.nodes(data=True)
            if data.get("file_path") == file_path
        ]
        self.graph.remove_nodes_from(nodes_to_remove)

    def clear(self):
        self.graph.clear()

    def save(self, path: str):
        """Save graph to disk."""
        import pickle
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "wb") as f:
                pickle.dump(self.graph, f)
            logger.info(f"Graph saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def load(self, path: str):
        """Load graph from disk."""
        import pickle
        import os
        if not os.path.exists(path):
            logger.info(f"No existing graph found at {path}, starting fresh.")
            return

        try:
            with open(path, "rb") as f:
                self.graph = pickle.load(f)
            logger.info(f"Graph loaded from {path} with {self.graph.number_of_nodes()} nodes")
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            self.graph = nx.DiGraph()

# Singleton instance
_graph_store_instance = None

def get_graph_store() -> GraphStore:
    global _graph_store_instance
    if _graph_store_instance is None:
        _graph_store_instance = GraphStore()
    return _graph_store_instance
