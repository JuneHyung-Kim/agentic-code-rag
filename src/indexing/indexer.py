import os
from typing import List
from .parser import CodeParser
from .vector_store import VectorStore

class CodeIndexer:
    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)
        self.parser = CodeParser()
        self.vector_store = VectorStore()
        
    def index_project(self):
        print(f"Indexing project at {self.root_path}...")
        count = 0
        for root, _, files in os.walk(self.root_path):
            # Skip hidden directories and build artifacts
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue
            if 'build' in root or 'venv' in root or '__pycache__' in root:
                continue

            for file in files:
                file_path = os.path.join(root, file)
                if self._index_file(file_path):
                    count += 1
        print(f"Indexing complete. Processed {count} files.")
                
    def _index_file(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1]
        lang = None
        if ext == '.py':
            lang = 'python'
        elif ext in ['.c', '.h']:
            lang = 'c'
        elif ext in ['.cpp', '.hpp', '.cc', '.cxx']:
            lang = 'cpp'
            
        if not lang:
            return False

        try:
            # Use errors='ignore' to skip encoding issues in binary/weird files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
                
            definitions = self.parser.extract_definitions(code, lang)
            
            if not definitions:
                return False

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
                print(f"Indexed {len(documents)} definitions from {rel_path}")
                return True
                
        except Exception as e:
            print(f"Failed to index {file_path}: {e}")
            return False
        
        return False
