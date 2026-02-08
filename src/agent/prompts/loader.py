"""YAML → ChatPromptTemplate loader with shared component resolution."""

from pathlib import Path
from typing import Dict

import yaml
from langchain_core.prompts import ChatPromptTemplate

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_cache: Dict[str, ChatPromptTemplate] = {}
_shared_cache: Dict[str, str] = {}


def _load_shared() -> Dict[str, str]:
    """Load shared components from _shared.yaml (cached)."""
    if _shared_cache:
        return _shared_cache

    shared_path = _TEMPLATES_DIR / "_shared.yaml"
    with open(shared_path, "r") as f:
        data = yaml.safe_load(f)

    _shared_cache.update(data)
    return _shared_cache


def load_prompt(name: str) -> ChatPromptTemplate:
    """Load a YAML template and return a ChatPromptTemplate.

    Resolves shared component placeholders (e.g. ``{planning_guidelines}``)
    from ``_shared.yaml`` before constructing the template. LangChain template
    variables (like ``{input}``, ``{findings}``) are preserved.

    Args:
        name: Template name without extension (e.g. "plan").

    Returns:
        A ChatPromptTemplate ready for ``.format_messages()`` or chain use.
    """
    if name in _cache:
        return _cache[name]

    shared = _load_shared()

    template_path = _TEMPLATES_DIR / f"{name}.yaml"
    with open(template_path, "r") as f:
        data = yaml.safe_load(f)

    messages = []
    for msg in data["messages"]:
        role = msg["role"]
        content = msg["content"]

        # Resolve shared components — these use Python str.format_map
        # but we must avoid touching LangChain variables like {input}.
        # Shared keys are resolved first; remaining {…} are kept for LangChain.
        content = _resolve_shared(content, shared)

        messages.append((role, content))

    prompt = ChatPromptTemplate.from_messages(messages)
    _cache[name] = prompt
    return prompt


def _resolve_shared(template: str, shared: Dict[str, str]) -> str:
    """Replace {shared_key} placeholders while preserving LangChain variables.

    Only keys present in the shared dict are substituted. All other
    ``{...}`` placeholders are left intact for LangChain.
    """
    for key, value in shared.items():
        placeholder = "{" + key + "}"
        if placeholder in template:
            template = template.replace(placeholder, value)
    return template


def reset_cache() -> None:
    """Clear all caches. Useful for testing."""
    _cache.clear()
    _shared_cache.clear()
