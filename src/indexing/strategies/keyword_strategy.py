import os
from typing import List
from .base_strategy import BaseStrategy
from ..schema import CodeNode
from ..storage.keyword_store import KeywordStore
from utils.logger import logger

class KeywordStrategy(BaseStrategy):
    def __init__(self, keyword_store: KeywordStore):
        self.keyword_store = keyword_store

    def index(self, file_path: str, nodes: List[CodeNode]) -> bool:
        if not nodes:
            return True
        
        try:
            documents = []
            doc_ids = []
            
            for node in nodes:
                if not node.id:
                    continue
                
                documents.append(node.content) # Keyword search usually on content
                doc_ids.append(node.id)

            self.keyword_store.add_documents(documents, doc_ids)
            return True
        except Exception as e:
            logger.error(f"KeywordStrategy failed to index {file_path}: {e}")
            return False

    def delete(self, file_path: str) -> bool:
        # Delete documents matching this file by filtering on ID prefix (abs_path:...).
        try:
            # Helper to find IDs matching file path
            ids_to_remove = [
                doc_id for doc_id in self.keyword_store.documents.keys() 
                if doc_id.startswith(f"{os.path.abspath(file_path)}:")
            ]
            self.keyword_store.delete_documents(ids_to_remove)
            return True
        except Exception as e:
            logger.error(f"KeywordStrategy failed to delete {file_path}: {e}")
            return False
