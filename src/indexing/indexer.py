import os
import hashlib
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from utils.logger import logger
from .parser import CodeParser
from .schema import CodeNode
from .file_registry import (
    SCHEMA_VERSION,
    load_registry,
    save_registry,
    build_file_record,
    get_project_files,
    update_project_files,
)
from .storage.vector_store import get_vector_store
from .storage.keyword_store import get_keyword_store
from .storage.graph_store import get_graph_store
from .strategies.vector_strategy import VectorStrategy
from .strategies.keyword_strategy import KeywordStrategy
from .strategies.graph_strategy import GraphStrategy

class CodeIndexer:
    def __init__(self, root_path: str, persist_path: str = "./db"):
        self.root_path = os.path.abspath(root_path)
        if not os.path.exists(self.root_path):
            raise FileNotFoundError(f"Project path does not exist: {self.root_path}")
        
        logger.info(f"Initializing indexer for: {self.root_path}")
        self.persist_path = os.path.abspath(persist_path)
        self.registry_path = os.path.join(self.persist_path, "index_registry.json")
        self.parser = CodeParser()
        self.supported_exts = {'.py', '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx'}
        
        # Initialize Stores (Singletons)
        self.vector_store = get_vector_store()
        self.keyword_store = get_keyword_store()
        self.graph_store = get_graph_store()
        
        # Initialize Strategies
        self.strategies = [
            VectorStrategy(self.vector_store, self.root_path),
            KeywordStrategy(self.keyword_store),
            GraphStrategy(self.graph_store, self.root_path)
        ]
        
    def index_project(self):
        logger.info(f"Starting indexing for project: {self.root_path}")

        # 1. Collect files
        all_files = self._discover_files()
        logger.info(f"Found {len(all_files)} files to scan")

        # 2. Load registry
        registry = load_registry(self.registry_path)
        if registry is None:
            registry = {
                "schema_version": SCHEMA_VERSION,
                "projects": {}
            }
            logger.info("No existing registry found, creating new one")

        # 3. Get existing files
        registry_files = get_project_files(registry, self.root_path)

        # 4. Build current file records
        current_files = {}
        current_paths = {}
        for file_path in all_files:
            abs_path = os.path.abspath(file_path)
            current_paths[abs_path] = file_path
            current_files[abs_path] = build_file_record(file_path)

        # 5. Compute changes
        added = [p for p in current_files.keys() if p not in registry_files]
        deleted = [p for p in registry_files.keys() if p not in current_files]
        modified = [
            p for p in current_files.keys()
            if p in registry_files and current_files[p]["sha1"] != registry_files[p].get("sha1")
        ]

        logger.info(f"Changes detected: Added={len(added)}, Modified={len(modified)}, Deleted={len(deleted)}")

        # 6. Process Deletions
        for abs_path in deleted + modified:
            self._delete_file(abs_path)

        # 7. Process Additions/Modifications
        target_paths = [current_paths[p] for p in added + modified]
        indexed_count, skipped_count, error_count = self._index_files(target_paths)

        # 8. Update Registry
        update_project_files(registry, self.root_path, current_files)
        save_registry(self.registry_path, registry)

        self._report_results(indexed_count, skipped_count, error_count)
                
    def _delete_file(self, abs_path: str):
        """Delegate deletion to all strategies."""
        for strategy in self.strategies:
            strategy.delete(abs_path)

    def _index_file(self, file_path: str) -> str:
        """Parses and indexes a single file using all strategies."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_exts:
            return "skipped"

        try:
            # 1. Read Content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                code = f.read()
            
            # 2. Parse
            nodes = self.parser.parse_file(file_path, code)
            
            if not nodes:
                return "skipped"

            # 3. Generate IDs centrally
            # This ensures all strategies use the same ID for the same node
            self._assign_node_ids(nodes)

            # 4. Delegate to Strategies
            for strategy in self.strategies:
                strategy.index(file_path, nodes)
            
            return "indexed"

        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")
            return "error"

    def _assign_node_ids(self, nodes: List[CodeNode]):
        """Generates and assigns unique IDs to nodes."""
        ids = []
        for node in nodes:
            sig_seed = "|".join([
                node.signature or "",
                node.return_type or "",
                ",".join(node.arguments) if node.arguments else ""
            ])
            hash_input = f"{node.content}\n{sig_seed}".encode("utf-8", errors="replace")
            short_hash = hashlib.sha1(hash_input).hexdigest()[:10]
            base_id = f"{node.file_path}:{node.type}:{node.name}:{node.start_line}:{short_hash}"

            unique_id = base_id
            suffix = 1
            while unique_id in ids:
                suffix += 1
                unique_id = f"{base_id}:{suffix}"
            ids.append(unique_id)
            node.id = unique_id

    def _index_files(self, file_paths: List[str]) -> Tuple[int, int, int]:
        indexed_count = 0
        skipped_count = 0
        error_count = 0

        for file_path in tqdm(file_paths, desc="Indexing files"):
            result = self._index_file(file_path)
            if result == "indexed":
                indexed_count += 1
            elif result == "skipped":
                skipped_count += 1
            else:
                error_count += 1

        return indexed_count, skipped_count, error_count

    def _discover_files(self) -> List[str]:
        all_files = []
        for root, _, files in os.walk(self.root_path):
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue
            if any(skip in root for skip in ['build', 'venv', '__pycache__', 'node_modules', '.git', 'dist', 'egg-info']):
                continue

            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file_path)[1].lower()
                if ext in self.supported_exts:
                    all_files.append(file_path)
        return all_files

    def _report_results(self, indexed_count: int, skipped_count: int, error_count: int) -> None:
        logger.info(f"Indexing complete. Indexed: {indexed_count}, Skipped: {skipped_count}, Errors: {error_count}")
        print(f"\n[OK] Indexing complete!")
        print(f"   [FILE] Indexed: {indexed_count} files")
        print(f"   [SKIP] Skipped: {skipped_count} files")
        print(f"   [ERR] Errors: {error_count} files")
