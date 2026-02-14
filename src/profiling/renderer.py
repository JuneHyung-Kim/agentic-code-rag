"""Markdown rendering for codebase profiles."""

from profiling.schema import CodebaseProfile, DirectoryNode


def render_full_markdown(profile: CodebaseProfile) -> str:
    """Render a complete, human-readable markdown document."""
    lines = [f"# Codebase Profile: {profile.project_root}", ""]

    lines.append(f"Generated: {profile.generated_at}  ")
    lines.append(f"Profile version: {profile.profile_version}")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append(f"- **Files**: {profile.total_files}")
    lines.append(f"- **Symbols**: {profile.total_symbols}")
    lines.append("")

    # Language stats
    if profile.language_stats:
        lines.append("## Languages")
        lines.append("| Language | Files | Symbols | Extensions |")
        lines.append("|----------|-------|---------|------------|")
        for lang, stats in sorted(profile.language_stats.items()):
            exts = ", ".join(stats.extensions)
            lines.append(f"| {lang} | {stats.file_count} | {stats.symbol_count} | {exts} |")
        lines.append("")

    # Entry points
    if profile.entry_points:
        lines.append("## Entry Points")
        for ep in profile.entry_points:
            lines.append(f"- `{ep.file_path}` — {ep.symbol_name} ({ep.reason})")
        lines.append("")

    # Directory tree
    if profile.directory_tree:
        lines.append("## Directory Structure")
        lines.append("```")
        _render_tree(profile.directory_tree, lines, prefix="")
        lines.append("```")
        lines.append("")

    # Module map
    if profile.module_map:
        lines.append("## Module Map")
        for fm in profile.module_map:
            lines.append(f"### `{fm.relative_path}` ({fm.language})")
            if fm.summary:
                lines.append(f"> {fm.summary}")
                lines.append("")
            for sym in fm.symbols:
                sig = f" — `{sym.signature}`" if sym.signature else ""
                lines.append(f"- {sym.type} **{sym.name}** (L{sym.start_line}){sig}")
            lines.append("")

    # Key modules
    if profile.key_modules:
        lines.append("## Key Modules")
        lines.append("| File | Symbols | In-degree | Out-degree | Role |")
        lines.append("|------|---------|-----------|------------|------|")
        for km in profile.key_modules:
            lines.append(
                f"| `{km.relative_path}` | {km.symbol_count} "
                f"| {km.total_in_degree} | {km.total_out_degree} | {km.role} |"
            )
        lines.append("")

    # Graph stats
    gs = profile.graph_stats
    if gs.total_nodes > 0:
        lines.append("## Call Graph")
        lines.append(f"- **Nodes**: {gs.total_nodes}")
        lines.append(f"- **Edges**: {gs.total_edges}")
        if gs.most_called:
            lines.append("- **Most called**:")
            for item in gs.most_called:
                lines.append(f"  - {item['name']} (in-degree: {item['in_degree']})")
        if gs.most_calling:
            lines.append("- **Most calling**:")
            for item in gs.most_calling:
                lines.append(f"  - {item['name']} (out-degree: {item['out_degree']})")
        lines.append("")

    # AI summary
    if profile.ai_summary:
        lines.append("## AI Summary")
        lines.append(profile.ai_summary)
        lines.append("")

    return "\n".join(lines)


def _render_tree(node: DirectoryNode, lines: list, prefix: str) -> None:
    lines.append(f"{prefix}{node.name}/" if node.type == "directory" else f"{prefix}{node.name}")
    sorted_children = sorted(node.children, key=lambda c: (c.type == "file", c.name))
    for i, child in enumerate(sorted_children):
        is_last = i == len(sorted_children) - 1
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        if child.type == "directory":
            lines.append(f"{prefix}{connector}{child.name}/")
            _render_tree_children(child, lines, prefix + extension)
        else:
            lines.append(f"{prefix}{connector}{child.name}")


def _render_tree_children(node: DirectoryNode, lines: list, prefix: str) -> None:
    sorted_children = sorted(node.children, key=lambda c: (c.type == "file", c.name))
    for i, child in enumerate(sorted_children):
        is_last = i == len(sorted_children) - 1
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        if child.type == "directory":
            lines.append(f"{prefix}{connector}{child.name}/")
            _render_tree_children(child, lines, prefix + extension)
        else:
            lines.append(f"{prefix}{connector}{child.name}")


def render_prompt_context(profile: CodebaseProfile, max_length: int = 6000) -> str:
    """Render a condensed version for LLM prompt injection.

    If an AI-generated summary exists, it is included in full (up to max_length)
    rather than being truncated to 300 chars.
    """
    # If ai_summary is available, use it as the primary context
    if profile.ai_summary:
        header = f"Project: {profile.project_root}\n"
        result = header + profile.ai_summary
        if len(result) > max_length:
            result = result[:max_length - 3] + "..."
        return result

    # Fallback: stats-based summary
    lines = [f"Project: {profile.project_root}"]
    lines.append(f"Files: {profile.total_files}, Symbols: {profile.total_symbols}")

    # Languages summary
    if profile.language_stats:
        lang_parts = [
            f"{lang} ({s.file_count} files, {s.symbol_count} symbols)"
            for lang, s in sorted(profile.language_stats.items())
        ]
        lines.append(f"Languages: {', '.join(lang_parts)}")

    # Entry points
    if profile.entry_points:
        lines.append("Entry points:")
        for ep in profile.entry_points:
            lines.append(f"  - {ep.file_path}: {ep.symbol_name} ({ep.reason})")

    # Top-level directories
    if profile.directory_tree and profile.directory_tree.children:
        dirs = [c.name for c in profile.directory_tree.children if c.type == "directory"]
        if dirs:
            lines.append(f"Top directories: {', '.join(sorted(dirs))}")

    # Key modules
    if profile.key_modules:
        lines.append("Key modules:")
        for km in profile.key_modules[:5]:
            lines.append(f"  - {km.relative_path} ({km.role})")

    # Graph highlights
    gs = profile.graph_stats
    if gs.total_nodes > 0:
        lines.append(f"Call graph: {gs.total_nodes} nodes, {gs.total_edges} edges")
        if gs.most_called:
            names = [item["name"] for item in gs.most_called[:3]]
            lines.append(f"Most called: {', '.join(names)}")

    result = "\n".join(lines)
    if len(result) > max_length:
        result = result[:max_length - 3] + "..."
    return result
