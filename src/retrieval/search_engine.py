from typing import List, Dict, Any
from indexing.storage.vector_store import VectorStore
from indexing.storage.keyword_store import KeywordStore
from utils.logger import logger

class SearchEngine:
    """
    Retrieval Business Logic.
    Implements Hybrid Search combining Semantic Search and Keyword Search.
    """
    def __init__(self, vector_store: VectorStore = None):
        if vector_store is None:
            from indexing.storage.vector_store import get_vector_store
            self.vector_store = get_vector_store()
        else:
            self.vector_store = vector_store
            
        from indexing.storage.keyword_store import get_keyword_store
        self.keyword_store = get_keyword_store()
        
        # Sync keyword store with vector store data on init
        self._sync_indices()

    def _sync_indices(self):
        """Syncs the in-memory keyword store with persistent vector store data."""
        try:
            data = self.vector_store.get_all_documents()
            if data and data['documents']:
                self.keyword_store.build_index(data['documents'], data['ids'])
        except Exception as e:
            logger.error(f"Failed to sync keyword index: {e}")

    def hybrid_search(
        self,
        query: str,
        n_results: int = 5,
        alpha: float = 0.7,
        project_root: str = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search logic.

        Args:
            query (str): The search query.
            n_results (int): Number of results to return.
            alpha (float): Weight for vector score (0.0~1.0).
                           1.0 = Pure Vector, 0.0 = Pure Keyword.
            project_root (str, optional): Filter results by project root path.
                           If None, searches across all projects.
        """
        # 1. Retrieval: Vector Search (Get more candidates for reranking)
        k_candidates = n_results * 2
        vector_res = self.vector_store.query(
            query,
            n_results=k_candidates,
            where_filter={"project_root": project_root} if project_root else None
        )
        
        if not vector_res['ids'] or not vector_res['ids'][0]:
            return []

        # 2. Retrieval: Keyword Search
        bm25_scores = self.keyword_store.search(query)
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0

        # 3. Reranking Logic
        combined_results = []
        ids = vector_res['ids'][0]
        distances = vector_res['distances'][0]
        metas = vector_res['metadatas'][0]
        docs = vector_res['documents'][0]

        for i, doc_id in enumerate(ids):
            # Normalize Vector Distance (L2) -> Similarity [0, 1]
            vec_score = 1.0 / (1.0 + distances[i])
            
            # Normalize BM25 Score -> [0, 1]
            raw_bm_score = bm25_scores.get(doc_id, 0.0)
            norm_bm_score = raw_bm_score / max_bm25 if max_bm25 > 0 else 0.0
            
            # Weighted Fusion
            final_score = (alpha * vec_score) + ((1 - alpha) * norm_bm_score)
            
            combined_results.append({
                "id": doc_id,
                "content": docs[i],
                "metadata": metas[i],
                "score": final_score
            })

        # 4. Sort and Limit
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        return combined_results[:n_results]

# Singleton instance
_search_engine_instance = None

def get_search_engine() -> SearchEngine:
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = SearchEngine()
    return _search_engine_instance
