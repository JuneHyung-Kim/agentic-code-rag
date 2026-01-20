import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from utils.logger import logger

class KeywordStore:
    """
    In-memory store for keyword-based search using BM25.
    Responsible for maintaining the BM25 index data structure.
    """
    def __init__(self):
        self.bm25 = None
        self.doc_ids = []
        self._is_ready = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text specifically for C/C++ code analysis.
        Splits by non-alphanumeric characters to capture variable names.
        """
        # Regex to split by non-word characters but keep underscores/digits
        return [word.lower() for word in re.split(r'[^a-zA-Z0-9_]+', text) if word]

    def build_index(self, documents: List[str], doc_ids: List[str]):
        """
        Build or rebuild the BM25 index.
        """
        if not documents:
            logger.warning("No documents provided to build BM25 index.")
            self._is_ready = False
            return

        logger.info(f"Building BM25 index for {len(documents)} documents...")
        
        try:
            self.doc_ids = doc_ids
            tokenized_corpus = [self._tokenize(doc) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._is_ready = True
            logger.info("BM25 index built successfully.")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self._is_ready = False

    def search(self, query: str) -> Dict[str, float]:
        """
        Return BM25 scores for the query.
        Returns: Dict[doc_id, score]
        """
        if not self._is_ready or not self.bm25:
            # logger.warning("BM25 index is not ready.") # Reduce noise
            return {}

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        score_map = {}
        for doc_id, score in zip(self.doc_ids, scores):
            if score > 0:
                score_map[doc_id] = score
                
        return score_map