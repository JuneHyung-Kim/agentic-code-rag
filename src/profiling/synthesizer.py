"""Two-step LLM pipeline for codebase profile synthesis."""

import json
import logging
import re
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.model import get_model
from profiling.schema import CodebaseProfile, FileSummary

logger = logging.getLogger(__name__)

_BATCH_SIZE = 25

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_MODULE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior software engineer. For each file listed below, write a "
     "one-sentence summary describing its purpose based on the file path and "
     "symbol names.\n\n"
     "Reply with ONLY a JSON object mapping file path to summary string.\n"
     "Example: {{\"src/utils.py\": \"Utility helpers for string formatting\"}}"),
    ("user", "{file_list}"),
])

_ARCHITECTURE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior software architect. Given the following codebase information, "
     "generate a structured project overview in Markdown, similar to a CLAUDE.md file.\n\n"
     "Include these sections:\n"
     "## Project Overview\n"
     "A brief description of what this project does.\n\n"
     "## Architecture\n"
     "How the codebase is organized and the main data/control flows.\n\n"
     "## Key Modules\n"
     "The most important files and their roles.\n\n"
     "## Entry Points\n"
     "Where execution starts.\n\n"
     "## Key Relationships\n"
     "Important dependencies and call patterns between modules.\n\n"
     "Be specific, factual, and concise. Do not invent information not present in the input."),
    ("user", "{context}"),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _trivial_summary(fm: FileSummary) -> str | None:
    """Return a rule-based summary for trivial files, or None."""
    basename = fm.relative_path.rsplit("/", 1)[-1] if "/" in fm.relative_path else fm.relative_path

    if len(fm.symbols) == 0:
        if basename == "__init__.py":
            return "package initializer"
        return "empty module" if fm.language == "python" else None

    if len(fm.symbols) == 1:
        sym = fm.symbols[0]
        return f"{sym.name} {sym.type} definition"

    if basename == "__init__.py" and all(s.type in ("import", "variable") for s in fm.symbols):
        return "package initializer with re-exports"

    return None


def _format_file_for_batch(fm: FileSummary) -> str:
    """Format a single file entry for the batch prompt."""
    symbols = ", ".join(f"{s.name}({s.type})" for s in fm.symbols)
    return f"File: {fm.relative_path} ({fm.language})\nSymbols: {symbols}"


# ---------------------------------------------------------------------------
# Step 1: Batch module summaries
# ---------------------------------------------------------------------------

def _step1_module_summaries(profile: CodebaseProfile) -> int:
    """Generate per-file summaries. Returns count of files summarized."""
    count = 0

    # First pass: trivial files (no LLM needed)
    non_trivial: List[FileSummary] = []
    for fm in profile.module_map:
        trivial = _trivial_summary(fm)
        if trivial is not None:
            fm.summary = trivial
            count += 1
        else:
            non_trivial.append(fm)

    if not non_trivial:
        return count

    # Second pass: LLM batches
    model = get_model()
    chain = _MODULE_SUMMARY_PROMPT | model | StrOutputParser()

    for i in range(0, len(non_trivial), _BATCH_SIZE):
        batch = non_trivial[i : i + _BATCH_SIZE]
        file_list = "\n\n".join(_format_file_for_batch(fm) for fm in batch)

        try:
            raw = chain.invoke({"file_list": file_list})
            cleaned = _strip_json_fences(raw)
            summaries = json.loads(cleaned)

            for fm in batch:
                if fm.relative_path in summaries:
                    fm.summary = summaries[fm.relative_path]
                    count += 1
        except Exception as e:
            logger.warning(f"Module summary batch {i // _BATCH_SIZE + 1} failed: {e}")

    return count


# ---------------------------------------------------------------------------
# Step 2: Architecture summary
# ---------------------------------------------------------------------------

def _build_architecture_context(profile: CodebaseProfile) -> str:
    """Build the context string for the architecture LLM call."""
    parts: List[str] = []

    # Module summaries
    summarized = [fm for fm in profile.module_map if fm.summary]
    if summarized:
        parts.append("## Module Summaries")
        for fm in summarized:
            parts.append(f"- `{fm.relative_path}`: {fm.summary}")

    # Key modules
    if profile.key_modules:
        parts.append("\n## Key Modules (by graph analysis)")
        for km in profile.key_modules:
            parts.append(
                f"- `{km.relative_path}` — {km.role} "
                f"(symbols: {km.symbol_count}, in: {km.total_in_degree}, out: {km.total_out_degree})"
            )

    # Entry points
    if profile.entry_points:
        parts.append("\n## Entry Points")
        for ep in profile.entry_points:
            parts.append(f"- `{ep.file_path}`: {ep.symbol_name} ({ep.reason})")

    # Graph stats
    gs = profile.graph_stats
    if gs.total_nodes > 0:
        parts.append(f"\n## Call Graph Stats")
        parts.append(f"- Nodes: {gs.total_nodes}, Edges: {gs.total_edges}")
        if gs.most_called:
            names = [f"{item['name']} (in:{item['in_degree']})" for item in gs.most_called[:5]]
            parts.append(f"- Most called: {', '.join(names)}")
        if gs.most_calling:
            names = [f"{item['name']} (out:{item['out_degree']})" for item in gs.most_calling[:5]]
            parts.append(f"- Most calling: {', '.join(names)}")

    # Language stats
    if profile.language_stats:
        parts.append("\n## Language Stats")
        for lang, stats in sorted(profile.language_stats.items()):
            parts.append(f"- {lang}: {stats.file_count} files, {stats.symbol_count} symbols")

    parts.append(f"\n## Summary Stats")
    parts.append(f"- Total files: {profile.total_files}")
    parts.append(f"- Total symbols: {profile.total_symbols}")

    return "\n".join(parts)


def _step2_architecture_summary(profile: CodebaseProfile) -> str | None:
    """Generate architecture summary. Returns None on failure."""
    context = _build_architecture_context(profile)
    model = get_model()
    chain = _ARCHITECTURE_PROMPT | model | StrOutputParser()

    try:
        return chain.invoke({"context": context})
    except Exception as e:
        logger.warning(f"Architecture summary generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_profile(profile: CodebaseProfile) -> CodebaseProfile:
    """Run the two-step LLM synthesis pipeline on a profile.

    Mutates and returns the same profile object:
    - Step 1: populates FileSummary.summary for each module
    - Step 2: populates profile.ai_summary with CLAUDE.md-style overview

    Each step is independent — failure in one does not block the other.
    """
    # Step 1: Module summaries
    try:
        count = _step1_module_summaries(profile)
        total = len(profile.module_map)
        logger.info(f"Module summaries: {count}/{total} files")
        print(f"  Module summaries: {count}/{total} files")
    except Exception as e:
        logger.warning(f"Step 1 (module summaries) failed: {e}")
        print(f"  Module summaries failed: {e}")

    # Step 2: Architecture summary
    try:
        summary = _step2_architecture_summary(profile)
        if summary:
            profile.ai_summary = summary
            print("  Architecture summary generated.")
        else:
            print("  Architecture summary empty.")
    except Exception as e:
        logger.warning(f"Step 2 (architecture summary) failed: {e}")
        print(f"  Architecture summary failed: {e}")

    return profile
