import os
from typing import List
from tqdm import tqdm
from .parser import CodeParser
from .vector_store import VectorStore
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
        
        # Collect all files first for progress bar
        all_files = []
        for root, _, files in os.walk(self.root_path):
            # Skip hidden directories and build artifacts
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue
            if any(skip in root for skip in ['build', 'venv', '__pycache__', 'node_modules', '.git']):
                continue

            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        
        logger.info(f"Found {len(all_files)} files to scan")
        
        # Index with progress bar
        indexed_count = 0
        skipped_count = 0
        error_count = 0
        
        for file_path in tqdm(all_files, desc="Indexing files"):
            result = self._index_file(file_path)
            if result == "indexed":
                indexed_count += 1
            elif result == "skipped":
                skipped_count += 1
            else:  # error
                error_count += 1
        
        logger.info(f"Indexing complete. Indexed: {indexed_count}, Skipped: {skipped_count}, Errors: {error_count}")
        print(f"\n✅ Indexing complete!")
        print(f"   📁 Indexed: {indexed_count} files")
        print(f"   ⏭️  Skipped: {skipped_count} files")
        print(f"   ❌ Errors: {error_count} files")
                
    def _index_file(self, file_path: str) -> str:
        """Index a single file. Returns: 'indexed', 'skipped', or 'error'"""
        ext = os.path.splitext(file_path)[1]
        lang = None
        if ext == '.py':
            lang = 'python'
        elif ext in ['.c', '.h']:
            lang = 'c'
        elif ext in ['.cpp', '.hpp', '.cc', '.cxx']:
            lang = 'cpp'
            
        if not lang:
            return "skipped"

        try:
            # Try UTF-8 first, then with error handling
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for {file_path}, trying with errors='replace'")
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    code = f.read()
                
            definitions = self.parser.extract_definitions(code, lang)
            
            if not definitions:
                logger.debug(f"No definitions found in {file_path}")
                return "skipped"

            documents = []
            metadatas = []
            ids = []
            
            # Relative path for cleaner metadata
            rel_path = os.path.relpath(file_path, self.root_path)

            for i, defn in enumerate(definitions):
                documents.append(defn['content'])
                metadatas.append({
                    'file_path': rel_path,
                    'name': defn['name'],
                    'type': defn['type'],
                    'start_line': defn['start_line'],
                    'end_line': defn['end_line'],
                    'language': lang
                })
                # Create a unique ID
                ids.append(f"{rel_path}:{defn['name']}:{i}")
                
            if documents:
                self.vector_store.add_documents(documents, metadatas, ids)
                logger.info(f"Indexed {len(documents)} definitions from {rel_path}")
                return "indexed"
                
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return "error"
        except PermissionError:
            logger.error(f"Permission denied: {file_path}")
            return "error"
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {type(e).__name__}: {e}")
            return "error"
        
        return "skipped"
