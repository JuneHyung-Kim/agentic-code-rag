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

    def delete_by_file(self, file_path: str):
        """Remove all nodes belonging to a specific file."""
        nodes_to_remove = [
            n for n, data in self.graph.nodes(data=True)
            if data.get("file_path") == file_path
        ]
        self.graph.remove_nodes_from(nodes_to_remove)

    def clear(self):
        self.graph.clear()

# Singleton instance
_graph_store_instance = None

def get_graph_store() -> GraphStore:
    global _graph_store_instance
    if _graph_store_instance is None:
        _graph_store_instance = GraphStore()
    return _graph_store_instance
