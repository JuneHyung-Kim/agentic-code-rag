"""Tests for profiling.renderer."""

import pytest

from profiling.schema import (
    CodebaseProfile,
    DirectoryNode,
    EntryPoint,
    FileSummary,
    GraphStats,
    LanguageStats,
    SymbolSummary,
)
from profiling.renderer import render_full_markdown, render_prompt_context


def _make_profile(**overrides) -> CodebaseProfile:
    defaults = dict(
        project_root="/home/user/project",
        generated_at="2025-01-01T00:00:00+00:00",
        total_files=3,
        total_symbols=10,
        language_stats={
            "python": LanguageStats(file_count=2, symbol_count=8, extensions=[".py"]),
            "c": LanguageStats(file_count=1, symbol_count=2, extensions=[".c"]),
        },
        directory_tree=DirectoryNode(
            name="project",
            children=[
                DirectoryNode(name="src", children=[
                    DirectoryNode(name="main.py", type="file"),
                ]),
                DirectoryNode(name="README.md", type="file"),
            ],
        ),
        module_map=[
            FileSummary(
                relative_path="src/main.py",
                language="python",
                symbols=[
                    SymbolSummary(name="main", type="function", start_line=1, signature="def main()"),
                ],
            ),
        ],
        entry_points=[
            EntryPoint(file_path="/home/user/project/src/main.py", symbol_name="main", reason="main function"),
        ],
        graph_stats=GraphStats(
            total_nodes=5,
            total_edges=8,
            most_called=[{"name": "helper", "in_degree": 4}],
            most_calling=[{"name": "main", "out_degree": 3}],
        ),
    )
    defaults.update(overrides)
    return CodebaseProfile(**defaults)


class TestRenderFullMarkdown:

    def test_contains_sections(self):
        md = render_full_markdown(_make_profile())
        assert "# Codebase Profile" in md
        assert "## Overview" in md
        assert "## Languages" in md
        assert "## Entry Points" in md
        assert "## Directory Structure" in md
        assert "## Module Map" in md
        assert "## Call Graph" in md

    def test_contains_language_data(self):
        md = render_full_markdown(_make_profile())
        assert "python" in md
        assert ".py" in md

    def test_ai_summary_included(self):
        md = render_full_markdown(_make_profile(ai_summary="This is a test project."))
        assert "## AI Summary" in md
        assert "This is a test project." in md

    def test_no_ai_summary_when_none(self):
        md = render_full_markdown(_make_profile())
        assert "## AI Summary" not in md


class TestRenderPromptContext:

    def test_compact_output(self):
        ctx = render_prompt_context(_make_profile())
        assert len(ctx) < 2000

    def test_contains_key_info(self):
        ctx = render_prompt_context(_make_profile())
        assert "project" in ctx.lower()
        assert "python" in ctx.lower()
        assert "main" in ctx.lower()

    def test_max_length_truncation(self):
        ctx = render_prompt_context(_make_profile(), max_length=50)
        assert len(ctx) <= 50
        assert ctx.endswith("...")

    def test_includes_graph_highlights(self):
        ctx = render_prompt_context(_make_profile())
        assert "5 nodes" in ctx
        assert "helper" in ctx
