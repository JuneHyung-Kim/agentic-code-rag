"""Tests for profiling.profile_store."""

import json
import os
import tempfile

import pytest

from profiling.schema import CodebaseProfile, LanguageStats
from profiling.profile_store import (
    save_profile,
    load_profile,
    load_prompt_context,
    get_codebase_context,
    reset_profile_cache,
)


def _make_profile() -> CodebaseProfile:
    return CodebaseProfile(
        project_root="/tmp/test-project",
        generated_at="2025-01-01T00:00:00+00:00",
        total_files=2,
        total_symbols=5,
        language_stats={
            "python": LanguageStats(file_count=2, symbol_count=5, extensions=[".py"]),
        },
    )


class TestSaveLoad:

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _make_profile()
            save_profile(profile, persist_path=tmpdir)

            loaded = load_profile(persist_path=tmpdir)
            assert loaded is not None
            assert loaded.project_root == profile.project_root
            assert loaded.total_files == profile.total_files
            assert loaded.total_symbols == profile.total_symbols
            assert "python" in loaded.language_stats

    def test_json_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_profile(_make_profile(), persist_path=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "codebase_profile.json"))

    def test_md_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_profile(_make_profile(), persist_path=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "codebase_profile.md"))


class TestLoadNonexistent:

    def test_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_profile(persist_path=tmpdir)
            assert result is None


class TestPromptContext:

    def test_from_saved_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_profile(_make_profile(), persist_path=tmpdir)
            ctx = load_prompt_context(persist_path=tmpdir)
            assert ctx is not None
            assert "python" in ctx.lower()

    def test_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = load_prompt_context(persist_path=tmpdir)
            assert ctx is None


class TestCaching:

    def setup_method(self):
        reset_profile_cache()

    def teardown_method(self):
        reset_profile_cache()

    def test_cache_returns_same(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_profile(_make_profile(), persist_path=tmpdir)
            ctx1 = get_codebase_context(persist_path=tmpdir)
            ctx2 = get_codebase_context(persist_path=tmpdir)
            assert ctx1 == ctx2

    def test_cache_returns_none_when_no_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = get_codebase_context(persist_path=tmpdir)
            assert ctx is None
