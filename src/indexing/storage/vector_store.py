import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from utils.logger import logger
from config import config

class VectorStore:
    """
    Data Access Object (DAO) for ChromaDB.
    Handles low-level DB operations: add, delete, query, get.
    """
    def __init__(self, collection_name="code_chunks", persist_path="./db"):
        logger.info(f"Initializing vector store at {persist_path}")
        self.collection_name = collection_name
        self.persist_path = persist_path
        self.client = chromadb.PersistentClient(path=persist_path)
        self.ef = self._get_embedding_function()
        
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.ef
            )
        except ValueError:
            # Simple retry logic for embedding conflict
            self.client.delete_collection(name=collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.ef
            )

    def _get_embedding_function(self):
        """Factory method for embedding function."""
        provider = config.embedding_provider
        if provider == "openai":
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=config.openai_api_key, model_name=config.embedding_model
            )
        elif provider == "gemini":
            return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=config.gemini_api_key
            )
        elif provider == "ollama":
            try:
                import chromadb.utils.embedding_functions as ef
                return ef.OllamaEmbeddingFunction(
                    url=f"{config.ollama_base_url}/api/embeddings",
                    model_name=config.embedding_model
                )
            except AttributeError:
                from utils.ollama_embedding import OllamaEmbeddingFunction
                return OllamaEmbeddingFunction(
                    base_url=config.ollama_base_url,
                    model_name=config.embedding_model
                )
        return embedding_functions.DefaultEmbeddingFunction()

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        if not documents: return
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def query(self, query_text: str, n_results: int = 5, where_filter: Dict = None):
        kwargs = {
            "query_texts": [query_text],
            "n_results": n_results
        }
        if where_filter:
            kwargs["where"] = where_filter
        return self.collection.query(**kwargs)

    def get_all_documents(self) -> Dict[str, Any]:
        """Retrieve all documents for syncing with other stores."""
        count = self.collection.count()
        if count == 0: return {}
        return self.collection.get(limit=count)

    def delete_by_file_path(self, file_path: str):
        self.collection.delete(where={"file_path": file_path})
        
    def reset_collection(self):
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name, embedding_function=self.ef
        )

# Singleton instance
_vector_store_instance = None

def get_vector_store() -> VectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance