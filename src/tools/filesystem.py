import os
from typing import List

class FileSystemTools:
    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)

    def list_files(self, directory: str = ".") -> List[str]:
        """List files in a directory relative to the project root."""
        target_dir = os.path.join(self.root_path, directory)
        if not os.path.exists(target_dir):
            return [f"Error: Directory {directory} does not exist."]
        
        try:
            return os.listdir(target_dir)
        except Exception as e:
            return [f"Error: {str(e)}"]

    def read_file(self, file_path: str) -> str:
        """Read the content of a file."""
        target_path = os.path.join(self.root_path, file_path)
        if not os.path.exists(target_path):
            return f"Error: File {file_path} does not exist."
        
        try:
            with open(target_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
