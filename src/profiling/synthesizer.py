"""Optional LLM-based summary generation for codebase profiles."""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.model import get_model
from profiling.schema import CodebaseProfile
from profiling.renderer import render_full_markdown

logger = logging.getLogger(__name__)

_MAX_INPUT_CHARS = 10000

_SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior software architect. Given a codebase profile, write a "
     "concise narrative summary (3-5 paragraphs) covering: purpose, architecture, "
     "key modules, entry points, and notable patterns. Be specific and factual."),
    ("user", "{profile_markdown}"),
])


def synthesize_summary(profile: CodebaseProfile) -> str:
    """Generate a narrative summary using the LLM.

    Returns empty string on failure (graceful degradation).
    """
    try:
        md = render_full_markdown(profile)
        if len(md) > _MAX_INPUT_CHARS:
            md = md[:_MAX_INPUT_CHARS] + "\n\n... (truncated)"

        chain = _SUMMARIZE_PROMPT | get_model() | StrOutputParser()
        return chain.invoke({"profile_markdown": md})
    except Exception as e:
        logger.warning(f"Profile summary generation failed: {e}")
        return ""
