import sys
import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Any
from utils.logger import logger
from config import config

class VectorStore:
    def __init__(self, collection_name="code_chunks", persist_path="./db"):
        logger.info(f"Initializing vector store at {persist_path}")
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # Get embedding function based on configuration
        self.ef = self._get_embedding_function()
        
        # Try to get or create collection, handle embedding function conflicts
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.ef
            )
        except ValueError as e:
            if "embedding function" in str(e).lower() and "conflict" in str(e).lower():
                logger.warning(f"Embedding function conflict detected. Deleting old collection '{collection_name}'...")
                try:
                    self.client.delete_collection(name=collection_name)
                    logger.info(f"Deleted old collection. Creating new one with current embedding function...")
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        embedding_function=self.ef
                    )
                except Exception as delete_err:
                    logger.error(f"Failed to delete/recreate collection: {delete_err}")
                    raise
            else:
                raise
    
    def _get_embedding_function(self):
        """Get embedding function based on EMBEDDING_PROVIDER configuration."""
        provider = config.embedding_provider
        
        if provider == "openai":
            logger.info(f"Using OpenAI Embeddings ({config.embedding_model})")
            config.validate_embedding_config()
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=config.openai_api_key,
                model_name=config.embedding_model
            )
        
        elif provider == "gemini":
            logger.info("Using Gemini Embeddings")
            config.validate_embedding_config()
            return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=config.gemini_api_key
            )
        
        elif provider == "ollama":
            logger.info(f"Using Ollama Embeddings ({config.embedding_model} at {config.ollama_base_url})")
            try:
                # Ollama uses OpenAI-compatible API
                import chromadb.utils.embedding_functions as ef
                return ef.OllamaEmbeddingFunction(
                    url=f"{config.ollama_base_url}/api/embeddings",
                    model_name=config.embedding_model
                )
            except AttributeError:
                # Fallback: use custom implementation if ChromaDB doesn't have OllamaEmbeddingFunction
                logger.warning("ChromaDB doesn't have built-in Ollama support. Using custom implementation.")
                from utils.ollama_embedding import OllamaEmbeddingFunction
                return OllamaEmbeddingFunction(
                    base_url=config.ollama_base_url,
                    model_name=config.embedding_model
                )
        
        elif provider == "default":
            logger.info("Using default embeddings (Sentence Transformers)")
            logger.warning("Default embeddings are slower and may have lower quality.")
            logger.warning("For better results, set EMBEDDING_PROVIDER to 'openai', 'gemini', or 'ollama'")
            return embedding_functions.DefaultEmbeddingFunction()
        
        else:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        if not documents:
            logger.debug("No documents to add")
            return
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.debug(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {type(e).__name__}: {e}")
            raise

    def query(self, query_text: str, n_results: int = 5):
        try:
            logger.debug(f"Querying vector store: '{query_text[:50]}...' (n_results={n_results})")
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            found = len(results['documents'][0]) if results['documents'] else 0
            logger.debug(f"Found {found} results")
            return results
        except Exception as e:
            logger.error(f"Query failed: {type(e).__name__}: {e}")
            raise
