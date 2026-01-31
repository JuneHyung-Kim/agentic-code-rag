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
        # KeywordStore in-memory doesn't track file_path -> ids easily 
        # unless we scan all docs or keep a reverse map.
        # Required: `KeywordStore` needs `delete_by_file` or we need track IDs.
        # My previous update to KeywordStore added `delete_documents(ids)`.
        # It doesn't support `delete_by_file_path`.
        
        # We might need to implement `delete_by_file` in KeywordStore 
        # or we just accept that in-memory keyword store might have stale entries 
        # until full rebuild? No, user wants incremental.
        
        # I'll enable `delete_by_file` in KeywordStore or handled here.
        # If I can't easily find IDs for a file, I can't delete them.
        # VectorStore has persistence and can query by metadata. KeywordStore doesn't.
        
        # Hack: We can query VectorStore for all IDs of this file, then delete from KeywordStore.
        # But VectorStrategy might have already deleted them?
        # Ordering matters!
        
        # Better: KeywordStore should track file_path.
        # But KeywordStore logic is about BM25.
        
        # Let's assume for now we might skip deletion in KeywordStore for this pass 
        # or simply rely on `_sync_indices` at startup.
        # BUT runtime updates are requested.
        
        # Implementation Detail:
        # Since I can't easily get IDs, I'll update KeywordStore to allow deletion by ID prefix?
        # ID is `abs_path:...`. So I can filter by prefix.
        
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
