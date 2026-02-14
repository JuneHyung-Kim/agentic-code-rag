"""Load, save, and cache codebase profiles."""

import json
import os
from typing import Optional

from profiling.schema import CodebaseProfile
from profiling.renderer import render_full_markdown, render_prompt_context

_DEFAULT_PERSIST_PATH = "./db"

_profile_cache: Optional[str] = None  # cached prompt context string
_cache_loaded: bool = False


def save_profile(profile: CodebaseProfile, persist_path: str = _DEFAULT_PERSIST_PATH) -> None:
    """Write profile as JSON and Markdown to persist_path."""
    os.makedirs(persist_path, exist_ok=True)

    json_path = os.path.join(persist_path, "codebase_profile.json")
    md_path = os.path.join(persist_path, "codebase_profile.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profile.model_dump(), f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_full_markdown(profile))


def load_profile(persist_path: str = _DEFAULT_PERSIST_PATH) -> Optional[CodebaseProfile]:
    """Load profile from JSON. Returns None if not found."""
    json_path = os.path.join(persist_path, "codebase_profile.json")
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return CodebaseProfile(**data)


def load_prompt_context(persist_path: str = _DEFAULT_PERSIST_PATH) -> Optional[str]:
    """Load profile and return the best available context.

    Prefers the AI-generated summary (ai_summary) when available,
    falling back to the stats-based render_prompt_context().
    """
    profile = load_profile(persist_path)
    if profile is None:
        return None
    if profile.ai_summary:
        return profile.ai_summary
    return render_prompt_context(profile)


def get_codebase_context(persist_path: str = _DEFAULT_PERSIST_PATH) -> Optional[str]:
    """Singleton-cached prompt context for injection into agent nodes."""
    global _profile_cache, _cache_loaded
    if _cache_loaded:
        return _profile_cache

    _profile_cache = load_prompt_context(persist_path)
    _cache_loaded = True
    return _profile_cache


def reset_profile_cache() -> None:
    """Clear the singleton cache. Call after re-init or for testing."""
    global _profile_cache, _cache_loaded
    _profile_cache = None
    _cache_loaded = False
