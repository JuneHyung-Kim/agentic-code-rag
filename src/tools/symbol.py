import os
from typing import Optional

from indexing.storage.vector_store import get_vector_store


class SymbolTool:
    """Tools for symbol-level queries against the vector store metadata."""

    def __init__(self):
        self.vector_store = get_vector_store()

    # -- public API -----------------------------------------------------------

    def get_symbol_definition(
        self, symbol_name: str, symbol_type: Optional[str] = None
    ) -> str:
        """Return definition location and source code for *symbol_name*."""
        if symbol_type:
            where = {"$and": [{"name": symbol_name}, {"type": symbol_type}]}
        else:
            where = {"name": symbol_name}

        result = self.vector_store.get_by_metadata(where)
        if not result:
            return (
                f"No definition found for '{symbol_name}'"
                + (f" (type={symbol_type})" if symbol_type else "")
                + ". The symbol may not be indexed."
            )

        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        documents = result.get("documents", [])

        lines = [f"Found {len(ids)} definition(s) for '{symbol_name}':"]
        for i, (meta, doc) in enumerate(zip(metadatas, documents)):
            name = meta.get("name", symbol_name)
            sym_type = meta.get("type", "unknown")
            file_path = meta.get("file_path", "unknown")
            start = meta.get("start_line", "?")
            end = meta.get("end_line", "?")
            signature = meta.get("signature", "")

            lines.append(f"\n--- Match {i + 1} ---")
            lines.append(f"Name: {name}")
            lines.append(f"Type: {sym_type}")
            lines.append(f"File: {file_path}")
            lines.append(f"Lines: {start}-{end}")
            if signature:
                lines.append(f"Signature: {signature}")
            if doc:
                content = doc[:2000] + "..." if len(doc) > 2000 else doc
                lines.append(f"Content:\n{content}")

        return "\n".join(lines)

    def get_module_summary(self, path: str) -> str:
        """Return a high-level symbol summary for a file or directory *path*."""
        abs_path = os.path.abspath(path)

        if os.path.isfile(abs_path):
            result = self.vector_store.get_by_metadata({"file_path": abs_path})
        else:
            # Directory: fetch all then filter by prefix
            result = self.vector_store.get_all_documents()
            if result:
                result = self._filter_by_prefix(result, abs_path)

        if not result:
            return f"No indexed symbols found for '{path}'."

        return self._format_summary(result, path)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _filter_by_prefix(result: dict, prefix: str) -> dict:
        """Keep only entries whose file_path starts with *prefix*."""
        ids, metas, docs = [], [], []
        for i, meta in enumerate(result.get("metadatas", [])):
            fp = meta.get("file_path", "")
            if fp.startswith(prefix):
                ids.append(result["ids"][i])
                metas.append(meta)
                docs.append(result["documents"][i] if result.get("documents") else "")
        if not ids:
            return {}
        return {"ids": ids, "metadatas": metas, "documents": docs}

    @staticmethod
    def _format_summary(result: dict, display_path: str) -> str:
        """Group symbols by file then by type."""
        from collections import defaultdict

        by_file: dict = defaultdict(list)
        for meta in result.get("metadatas", []):
            fp = meta.get("file_path", "unknown")
            by_file[fp].append(meta)

        lines = [f"Module summary for '{display_path}' ({len(result['ids'])} symbols):"]
        for fp in sorted(by_file):
            lines.append(f"\n  {fp}:")
            by_type: dict = defaultdict(list)
            for m in by_file[fp]:
                by_type[m.get("type", "unknown")].append(m)
            for sym_type in sorted(by_type):
                for m in by_type[sym_type]:
                    name = m.get("name", "?")
                    sig = m.get("signature", "")
                    start = m.get("start_line", "?")
                    end = m.get("end_line", "?")
                    detail = f" — {sig}" if sig else ""
                    lines.append(
                        f"    [{sym_type}] {name} (L{start}-{end}){detail}"
                    )

        return "\n".join(lines)
