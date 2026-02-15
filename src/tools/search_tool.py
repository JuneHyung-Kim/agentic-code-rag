from typing import Dict, Any
from indexing.storage.vector_store import VectorStore
from retrieval.search_engine import get_search_engine

class SearchTool:
    """
    External interface for agents to perform code searches.
    Delegates actual work to the SearchEngine in the retrieval layer.
    """
    def __init__(self):
        # Use singleton engine
        self.search_engine = get_search_engine()

    @staticmethod
    def _make_snippet(content: str, max_lines: int = 5, max_chars: int = 300) -> str:
        """Truncate content to a short snippet for LLM-friendly output."""
        lines = content.splitlines()
        snippet_lines = lines[:max_lines]
        snippet = "\n".join(snippet_lines)
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars]
        if len(lines) > max_lines or len(content) > max_chars:
            snippet += "\n..."
        return snippet

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
            snippet = self._make_snippet(res['content'])
            name = meta.get('name', '')
            node_type = meta.get('type', '')
            name_line = f"Name: {name} [{node_type}]\n" if name else ""
            formatted.append(
                f"Result {i+1} (Score: {res['score']:.2f}):\n"
                f"File: {meta.get('file_path')}\n"
                f"{name_line}"
                f"Lines: {meta.get('start_line')}-{meta.get('end_line')}\n"
                f"Snippet:\n{snippet}\n"
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