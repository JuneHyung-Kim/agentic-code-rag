import os
import pickle
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
        self.documents = {} # id -> text mapping
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
        Build or rebuild the BM25 index from scratch.
        """
        if not documents:
            logger.warning("No documents provided to build BM25 index.")
            self._is_ready = False
            return

        logger.info(f"Building BM25 index for {len(documents)} documents...")
        
        try:
            self.documents = {doc_id: doc for doc_id, doc in zip(doc_ids, documents)}
            self._rebuild_index()
            logger.info("BM25 index built successfully.")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self._is_ready = False

    def add_documents(self, documents: List[str], doc_ids: List[str]):
        """Add new documents and rebuild index."""
        if not documents: return
        
        for doc, doc_id in zip(documents, doc_ids):
            self.documents[doc_id] = doc
        
        self._rebuild_index()

    def delete_documents(self, doc_ids: List[str]):
        """Remove documents and rebuild index."""
        if not doc_ids: return
        
        changed = False
        for doc_id in doc_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
                changed = True
        
        if changed:
            self._rebuild_index()

    def _rebuild_index(self):
        """Internal method to rebuild BM25 from self.documents"""
        if not self.documents:
            self.bm25 = None
            self._is_ready = False
            return

        tokenized_corpus = [self._tokenize(doc) for doc in self.documents.values()]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._is_ready = True

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
        doc_ids = list(self.documents.keys())
        for doc_id, score in zip(doc_ids, scores):
            if score > 0:
                score_map[doc_id] = score
                
        return score_map
    
    def save(self, path: str):
        """Save keyword index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "wb") as f:
                # We save documents. BM25 object is not easily picklable if it contains lambdas,
                # but rank_bm25 objects usually are. 
                # Ideally we just save documents and rebuild, but rebuilding is fast enough?
                # For safety, let's save documents and rebuild on load to avoid pickle issues with version changes of rank_bm25
                # Actually, rebuilding 10k docs is fast.
                pickle.dump(self.documents, f)
            logger.info(f"Keyword index documents saved to {path} ({len(self.documents)} docs)")
        except Exception as e:
            logger.error(f"Failed to save keyword index: {e}")

    def load(self, path: str):
        """Load keyword index from disk."""
        if not os.path.exists(path):
            logger.info(f"No existing keyword index found at {path}")
            return

        try:
            with open(path, "rb") as f:
                self.documents = pickle.load(f)
            logger.info(f"Keyword documents loaded from {path}. Rebuilding index...")
            self._rebuild_index()
        except Exception as e:
            logger.error(f"Failed to load keyword index: {e}")
            self.documents = {}
            self._is_ready = False

# Singleton instance
_keyword_store_instance = None

def get_keyword_store() -> KeywordStore:
    global _keyword_store_instance
    if _keyword_store_instance is None:
        _keyword_store_instance = KeywordStore()
    return _keyword_store_instance
