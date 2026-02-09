"""Deterministic profile extraction from VectorStore + GraphStore."""

import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from profiling.schema import (
    CodebaseProfile,
    DirectoryNode,
    EntryPoint,
    FileSummary,
    GraphStats,
    LanguageStats,
    SymbolSummary,
)


_ENTRY_POINT_FILENAMES = {"cli.py", "__main__.py", "main.py", "app.py"}


class ProfileBuilder:
    def __init__(self, vector_store, graph_store, project_root: str):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.project_root = os.path.abspath(project_root)

    def build(self) -> CodebaseProfile:
        docs = self.vector_store.get_all_documents()
        metas: List[Dict[str, Any]] = docs.get("metadatas", []) if docs else []

        # Filter to this project
        metas = [
            m for m in metas
            if m.get("project_root", "") == self.project_root
               or m.get("file_path", "").startswith(self.project_root)
        ]

        language_stats = self._extract_language_stats(metas)
        module_map = self._extract_module_map(metas)
        entry_points = self._detect_entry_points(metas)
        graph_stats = self._extract_graph_stats()
        directory_tree = self._build_directory_tree(metas)

        files = set(m.get("file_path", "") for m in metas)

        return CodebaseProfile(
            project_root=self.project_root,
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_files=len(files),
            total_symbols=len(metas),
            language_stats=language_stats,
            directory_tree=directory_tree,
            module_map=module_map,
            entry_points=entry_points,
            graph_stats=graph_stats,
        )

    def _extract_language_stats(self, metas: List[Dict]) -> Dict[str, LanguageStats]:
        lang_files: Dict[str, set] = defaultdict(set)
        lang_symbols: Dict[str, int] = defaultdict(int)
        lang_exts: Dict[str, set] = defaultdict(set)

        for m in metas:
            lang = m.get("language", "unknown")
            fp = m.get("file_path", "")
            lang_files[lang].add(fp)
            lang_symbols[lang] += 1
            ext = os.path.splitext(fp)[1]
            if ext:
                lang_exts[lang].add(ext)

        return {
            lang: LanguageStats(
                file_count=len(lang_files[lang]),
                symbol_count=lang_symbols[lang],
                extensions=sorted(lang_exts[lang]),
            )
            for lang in lang_files
        }

    def _extract_module_map(self, metas: List[Dict]) -> List[FileSummary]:
        by_file: Dict[str, List[Dict]] = defaultdict(list)
        file_lang: Dict[str, str] = {}

        for m in metas:
            fp = m.get("file_path", "")
            by_file[fp].append(m)
            if not file_lang.get(fp):
                file_lang[fp] = m.get("language", "")

        result = []
        for fp in sorted(by_file):
            symbols_meta = sorted(by_file[fp], key=lambda x: x.get("start_line", 0))
            symbols = [
                SymbolSummary(
                    name=sm.get("name", ""),
                    type=sm.get("type", ""),
                    start_line=sm.get("start_line", 0),
                    signature=sm.get("signature", ""),
                )
                for sm in symbols_meta
            ]
            rel_path = os.path.relpath(fp, self.project_root) if fp else fp
            result.append(FileSummary(
                relative_path=rel_path,
                language=file_lang.get(fp, ""),
                symbols=symbols,
            ))
        return result

    def _detect_entry_points(self, metas: List[Dict]) -> List[EntryPoint]:
        entry_points = []
        seen = set()

        for m in metas:
            fp = m.get("file_path", "")
            name = m.get("name", "")
            sym_type = m.get("type", "")

            # Heuristic 1: main function
            if name == "main" and sym_type == "function" and fp not in seen:
                entry_points.append(EntryPoint(
                    file_path=fp, symbol_name=name, reason="main function",
                ))
                seen.add(fp)

            # Heuristic 2: known entry-point filenames
            basename = os.path.basename(fp)
            if basename in _ENTRY_POINT_FILENAMES and fp not in seen:
                entry_points.append(EntryPoint(
                    file_path=fp, symbol_name=name or basename, reason=f"entry-point file ({basename})",
                ))
                seen.add(fp)

        return entry_points

    def _extract_graph_stats(self) -> GraphStats:
        graph = self.graph_store.graph

        # Filter nodes belonging to this project
        project_nodes = [
            n for n, data in graph.nodes(data=True)
            if data.get("file_path", "").startswith(self.project_root)
        ]
        if not project_nodes:
            return GraphStats(
                total_nodes=0,
                total_edges=0,
            )

        project_set = set(project_nodes)
        project_edges = [
            (u, v) for u, v in graph.edges()
            if u in project_set and v in project_set
        ]

        # In-degree (most called)
        in_deg: Dict[str, int] = defaultdict(int)
        out_deg: Dict[str, int] = defaultdict(int)
        for u, v in project_edges:
            in_deg[v] += 1
            out_deg[u] += 1

        most_called = [
            {"node_id": n, "name": graph.nodes[n].get("name", n), "in_degree": d}
            for n, d in sorted(in_deg.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        most_calling = [
            {"node_id": n, "name": graph.nodes[n].get("name", n), "out_degree": d}
            for n, d in sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        return GraphStats(
            total_nodes=len(project_nodes),
            total_edges=len(project_edges),
            most_called=most_called,
            most_calling=most_calling,
        )

    def _build_directory_tree(self, metas: List[Dict]) -> DirectoryNode:
        root = DirectoryNode(name=os.path.basename(self.project_root) or self.project_root)

        for m in metas:
            fp = m.get("file_path", "")
            if not fp:
                continue
            rel = os.path.relpath(fp, self.project_root)
            parts = rel.split(os.sep)
            self._insert_path(root, parts)

        return root

    def _insert_path(self, node: DirectoryNode, parts: List[str]) -> None:
        if not parts:
            return
        name = parts[0]
        is_file = len(parts) == 1

        for child in node.children:
            if child.name == name:
                if not is_file:
                    self._insert_path(child, parts[1:])
                return

        new_node = DirectoryNode(name=name, type="file" if is_file else "directory")
        node.children.append(new_node)
        if not is_file:
            self._insert_path(new_node, parts[1:])
