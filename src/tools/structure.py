import os
from typing import Dict, Any, List

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

    def read_file(self, file_path: str, start_line: int = 1, end_line: int = None) -> str:
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
            
            if end_line is None: end_line = len(lines)
            return "".join(lines[start_line-1:end_line])
        except Exception as e:
            return f"Error: {e}"
