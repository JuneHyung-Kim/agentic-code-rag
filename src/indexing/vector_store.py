import sys
import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Any
from utils.logger import logger

class VectorStore:
    def __init__(self, collection_name="code_chunks", persist_path="./db"):
        logger.info(f"Initializing vector store at {persist_path}")
        self.client = chromadb.PersistentClient(path=persist_path)
        
        openai_key = os.getenv("OPENAI_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        if openai_key:
            logger.info("Using OpenAI Embeddings (text-embedding-3-small)")
            self.ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-3-small"
            )
        elif gemini_key:
            logger.info("Using Gemini Embeddings")
            self.ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=gemini_key
            )
        else:
            logger.warning("No API key found. Using default embeddings (Sentence Transformers)")
            logger.warning("For better results, set OPENAI_API_KEY or GEMINI_API_KEY in .env")
            self.ef = embedding_functions.DefaultEmbeddingFunction()
            
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef
        )

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
