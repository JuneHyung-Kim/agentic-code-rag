import fnmatch
import os
from typing import Dict, Any, List

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist", ".tox", ".mypy_cache", ".pytest_cache", "db"}


class FileSystemTools:
    """
    Tools for navigating and reading the filesystem.
    """
    def __init__(self, project_root: str = "./"):
        self.project_root = os.path.abspath(project_root)

    def list_dir(self, directory: str = ".") -> str:
        """
        List contents of a directory.
        """
        try:
            target_path = os.path.abspath(os.path.join(self.project_root, directory))
            if not target_path.startswith(self.project_root):
                return f"Error: Access denied. {self.project_root}"
            if not os.path.exists(target_path): return f"Error: Not found: {directory}"
            if not os.path.isdir(target_path): return f"Error: Not a dir: {directory}"
            
            items = os.listdir(target_path)
            return "\n".join([f"[{'DIR' if os.path.isdir(os.path.join(target_path, i)) else 'FILE'}] {i}" for i in items])
        except Exception as e:
            return f"Error: {e}"

    def read_file(self, file_path: str, start_line: int = 1, end_line: int = 0) -> str:
        """
        Read content of a file.
        """
        try:
            target_path = os.path.abspath(os.path.join(self.project_root, file_path))
            if not target_path.startswith(self.project_root):
                return f"Error: Access denied."
            if not os.path.exists(target_path): return f"Error: Not found."
            
            with open(target_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            if end_line <= 0: end_line = len(lines)
            return "".join(lines[start_line-1:end_line])
        except Exception as e:
            return f"Error: {e}"

    def find_files(self, pattern: str, path: str = ".", max_results: int = 50) -> str:
        """Find files matching a glob pattern (e.g. '*.py', 'test_*.c') recursively."""
        try:
            base = os.path.abspath(os.path.join(self.project_root, path))
            if not base.startswith(self.project_root):
                return "Error: Access denied."

            matches: List[str] = []
            for root, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for fname in files:
                    if fnmatch.fnmatch(fname, pattern):
                        full = os.path.join(root, fname)
                        rel = os.path.relpath(full, self.project_root)
                        matches.append(rel)
                        if len(matches) >= max_results:
                            break
                if len(matches) >= max_results:
                    break

            if not matches:
                return f"No files matching '{pattern}' found."
            header = f"Found {len(matches)} file(s) matching '{pattern}':\n"
            return header + "\n".join(matches)
        except Exception as e:
            return f"Error: {e}"
