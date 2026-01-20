from typing import Dict, Any
from storage.vector_store import VectorStore
from retrieval.search_engine import SearchEngine

class SearchTool:
    """
    External interface for agents to perform code searches.
    Delegates actual work to the SearchEngine in the retrieval layer.
    """
    def __init__(self):
        # In a generic app, these might be singletons or injected
        self.vector_store = VectorStore()
        self.search_engine = SearchEngine(self.vector_store)

    def search_codebase(self, query: str, n_results: int = 5) -> str:
        """
        Tool entry point for searching code.
        """
        results = self.search_engine.hybrid_search(query, n_results=n_results)
        
        if not results:
            return "No relevant code found."

        formatted = []
        for i, res in enumerate(results):
            meta = res['metadata']
            formatted.append(
                f"Result {i+1} (Score: {res['score']:.2f}):\n"
                f"File: {meta.get('file_path')}\n"
                f"Line: {meta.get('start_line')}-{meta.get('end_line')}\n"
                f"Content:\n{res['content']}\n"
            )
        return "\n".join(formatted)

    def get_tool_definition(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_codebase",
                "description": "Search code using hybrid (semantic+keyword) strategy.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "n_results": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        }