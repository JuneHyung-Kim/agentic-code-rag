import sys
import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, collection_name="code_chunks", persist_path="./db"):
        self.client = chromadb.PersistentClient(path=persist_path)
        
        openai_key = os.getenv("OPENAI_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        if openai_key:
            print("Using OpenAI Embeddings")
            self.ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-3-small"
            )
        elif gemini_key:
            print("Using Gemini Embeddings")
            self.ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=gemini_key
            )
        else:
            print("Using Default Embeddings (Sentence Transformers)")
            self.ef = embedding_functions.DefaultEmbeddingFunction()
            
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef
        )

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        if not documents:
            return
            
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_text: str, n_results: int = 5):
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
