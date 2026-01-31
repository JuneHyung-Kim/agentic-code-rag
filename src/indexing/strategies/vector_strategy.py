import os
import hashlib
import json
from typing import List
from .base_strategy import BaseStrategy
from ..schema import CodeNode
from ..storage.vector_store import VectorStore
from utils.logger import logger

class VectorStrategy(BaseStrategy):
    def __init__(self, vector_store: VectorStore, project_root: str):
        self.vector_store = vector_store
        self.project_root = project_root

    def index(self, file_path: str, nodes: List[CodeNode]) -> bool:
        if not nodes:
            return True

        documents = []
        metadatas = []
        ids = []

        abs_path = os.path.abspath(file_path)
        rel_path = os.path.relpath(file_path, self.project_root)

        for _, node in enumerate(nodes):
            # Construct embedding text
            parts = []
            if node.docstring:
                parts.append(f"Docstring: {node.docstring}")
            if node.signature:
                parts.append(f"Signature: {node.signature}")
            if node.return_type:
                parts.append(f"Returns: {node.return_type}")
            if node.arguments:
                parts.append(f"Parameters: {', '.join(node.arguments)}")
            parts.append(f"Code:\n{node.content}")
            embed_text = "\n\n".join(parts)

            documents.append(embed_text)

            # Build metadata
            metadata = {
                'file_path': abs_path,
                'project_root': self.project_root,
                'relative_path': rel_path,
                'name': node.name,
                'type': node.type,
                'language': node.language,
                'start_line': node.start_line,
                'end_line': node.end_line
            }
            
            if node.parent_name:
                metadata['parent_name'] = node.parent_name
            if node.signature:
                metadata['signature'] = node.signature
            if node.return_type:
                metadata['return_type'] = node.return_type
            if node.imports:
                metadata['imports'] = ", ".join(node.imports)[:1000]
            if node.arguments:
                metadata['arguments'] = json.dumps(node.arguments)
            # Flatten function_calls for metadata if needed
            if node.function_calls:
                metadata['function_calls'] = ", ".join(node.function_calls)[:1000]

            metadatas.append(metadata)
            
            # Use centrally assigned ID
            if not node.id:
                 # Fallback if ID generation missed (should not happen)
                 logger.warning(f"Node {node.name} has no ID, skipping")
                 continue
                 
            ids.append(node.id)
            
        try:
            self.vector_store.add_documents(documents, metadatas, ids)
            return True
        except Exception as e:
            logger.error(f"VectorStrategy failed to index {file_path}: {e}")
            return False

    def delete(self, file_path: str) -> bool:
        try:
            self.vector_store.delete_by_file_path(file_path)
            return True
        except Exception as e:
            logger.error(f"VectorStrategy failed to delete {file_path}: {e}")
            return False
