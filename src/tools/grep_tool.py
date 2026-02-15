import fnmatch
import os
import re
from typing import List

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist", ".tox", ".mypy_cache", ".pytest_cache", "db"}


class GrepTool:
    """Regex pattern search across project files."""

    def __init__(self, project_root: str = "./"):
        self.project_root = os.path.abspath(project_root)

    @staticmethod
    def _is_binary(file_path: str) -> bool:
        """Check if a file is binary by looking for null bytes in the first 512 bytes."""
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(512)
            return b"\x00" in chunk
        except OSError:
            return True

    def grep(
        self,
        pattern: str,
        max_results: int = 30,
        context_lines: int = 1,
        glob_filter: str = "",
    ) -> str:
        """Search for a regex pattern across project files.

        Args:
            pattern: Regular expression to search for.
            max_results: Maximum number of matches to return.
            context_lines: Number of context lines before and after each match.
            glob_filter: Optional glob to filter filenames (e.g. '*.py').

        Returns:
            Formatted string of matches with file, line number, and context.
        """
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        matches: List[str] = []
        match_count = 0

        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in sorted(files):
                if glob_filter and not fnmatch.fnmatch(fname, glob_filter):
                    continue
                full_path = os.path.join(root, fname)
                if self._is_binary(full_path):
                    continue
                rel_path = os.path.relpath(full_path, self.project_root)

                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                except OSError:
                    continue

                for i, line in enumerate(lines):
                    if regex.search(line):
                        match_count += 1
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        context = "".join(lines[start:end]).rstrip("\n")
                        matches.append(f"[{match_count}] {rel_path}:{i + 1}\n    {context}")
                        if match_count >= max_results:
                            break
                if match_count >= max_results:
                    break
            if match_count >= max_results:
                break

        if not matches:
            return f"No matches found for pattern '{pattern}'."
        header = f"Found {match_count} match(es) for '{pattern}':\n\n"
        return header + "\n\n".join(matches)
