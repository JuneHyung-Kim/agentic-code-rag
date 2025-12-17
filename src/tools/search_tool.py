from typing import List, Dict, Any
from indexing.vector_store import VectorStore

class SearchTool:
    def __init__(self):
        self.vector_store = VectorStore()

    def search_codebase(self, query: str, n_results: int = 5) -> str:
        """
        Search the codebase for relevant code snippets based on a natural language query.
        
        Args:
            query (str): The search query describing what you are looking for.
            n_results (int): Number of results to return.
            
        Returns:
            str: A formatted string containing the search results.
        """
        results = self.vector_store.query(query, n_results)
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant code found."

        formatted_results = []
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            formatted_results.append(
                f"Result {i+1}:\n"
                f"File: {meta['file_path']}\n"
                f"Type: {meta['type']}\n"
                f"Name: {meta['name']}\n"
                f"Line: {meta['start_line']}-{meta['end_line']}\n"
                f"Content:\n{doc}\n"
            )
            
        return "\n".join(formatted_results)

    def get_tool_definition(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_codebase",
                "description": "Search the codebase for relevant code snippets using semantic search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query describing the code functionality or concept."
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5).",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }
