import os
import json
import hashlib
from typing import List, Dict, Any
from tqdm import tqdm
from .parser import CodeParser
from .vector_store import VectorStore
from .schema import CodeNode
from utils.logger import logger

class CodeIndexer:
    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)
        if not os.path.exists(self.root_path):
            raise FileNotFoundError(f"Project path does not exist: {self.root_path}")
        if not os.path.isdir(self.root_path):
            raise NotADirectoryError(f"Project path is not a directory: {self.root_path}")
        
        logger.info(f"Initializing indexer for: {self.root_path}")
        self.parser = CodeParser()
        self.vector_store = VectorStore()
        
    def index_project(self):
        logger.info(f"Starting indexing for project: {self.root_path}")
        
        # 1. Collect all files to scan
        all_files = []
        for root, _, files in os.walk(self.root_path):
            # Skip hidden directories and build artifacts
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue
            if any(skip in root for skip in ['build', 'venv', '__pycache__', 'node_modules', '.git', 'dist', 'egg-info']):
                continue

            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        
        logger.info(f"Found {len(all_files)} files to scan")
        
        # 2. Index files with progress bar
        indexed_count = 0
        skipped_count = 0
        error_count = 0
        
        for file_path in tqdm(all_files, desc="Indexing files"):
            result = self._index_file(file_path)
            if result == "indexed":
                indexed_count += 1
            elif result == "skipped":
                skipped_count += 1
            else:
                error_count += 1
        
        # 3. Report results
        logger.info(f"Indexing complete. Indexed: {indexed_count}, Skipped: {skipped_count}, Errors: {error_count}")
        print(f"\n✅ Indexing complete!")
        print(f"   📁 Indexed: {indexed_count} files")
        print(f"   ⏭️  Skipped: {skipped_count} files")
        print(f"   ❌ Errors: {error_count} files")
                
    def _index_file(self, file_path: str) -> str:
        """
        Parses and indexes a single file.
        Returns: 'indexed', 'skipped', or 'error'
        """
        # 1. Filter by extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.py', '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx']:
            return "skipped"

        try:
            # 2. Read file content (Handle encoding issues)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for {file_path}, trying with errors='replace'")
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    code = f.read()
            
            # 3. Parse file using the new CodeParser
            # This returns a list of CodeNode objects containing rich metadata
            nodes: List[CodeNode] = self.parser.parse_file(file_path, code)
            
            if not nodes:
                logger.debug(f"No definitions found in {file_path}")
                return "skipped"

            documents = []
            metadatas = []
            ids = []
            
            rel_path = os.path.relpath(file_path, self.root_path)

            for i, node in enumerate(nodes):
                # 4. Construct embedding text
                # Inject docstring/signature into the content to improve semantic search
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
                
                # 5. Build metadata (Flatten lists to strings for vector DB compatibility)
                metadata = {
                    'file_path': rel_path,
                    'name': node.name,
                    'type': node.type,
                    'language': node.language,
                    'start_line': node.start_line,
                    'end_line': node.end_line
                }
                
                # Add optional fields if they exist
                if node.parent_name:
                    metadata['parent_name'] = node.parent_name

                if node.signature:
                    metadata['signature'] = node.signature
                if node.return_type:
                    metadata['return_type'] = node.return_type

                # Join lists into strings (e.g., ['pandas', 'numpy'] -> "pandas, numpy")
                if node.imports:
                    metadata['imports'] = ", ".join(node.imports)[:1000] # Truncate for safety
                if node.arguments:
                    metadata['arguments'] = json.dumps(node.arguments)
                
                metadatas.append(metadata)
                
                # 6. Generate unique ID
                # Format: "path:kind:name:line:hash"
                sig_seed = "|".join([
                    node.signature or "",
                    node.return_type or "",
                    ",".join(node.arguments) if node.arguments else ""
                ])
                hash_input = f"{node.content}\n{sig_seed}".encode("utf-8", errors="replace")
                short_hash = hashlib.sha1(hash_input).hexdigest()[:10]
                base_id = f"{rel_path}:{node.type}:{node.name}:{node.start_line}:{short_hash}"

                unique_id = base_id
                suffix = 1
                while unique_id in ids:
                    suffix += 1
                    unique_id = f"{base_id}:{suffix}"
                ids.append(unique_id)
                
            # 7. Add to Vector Store
            if documents:
                self.vector_store.add_documents(documents, metadatas, ids)
                return "indexed"
                
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {type(e).__name__}: {e}")
            return "error"
        
        return "skipped"
