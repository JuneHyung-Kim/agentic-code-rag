import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from config import config

logger = logging.getLogger(__name__)

_model_instance = None
_model_with_tools_instance = None


def get_model():
    """Get or create the LLM instance (singleton pattern)."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    if config.chat_provider == "gemini":
        _model_instance = ChatGoogleGenerativeAI(
            model=config.chat_model,
            google_api_key=config.gemini_api_key,
            temperature=0,
            convert_system_message_to_human=True,
        )
    elif config.chat_provider == "ollama":
        _model_instance = ChatOllama(
            base_url=config.ollama_base_url,
            model=config.chat_model,
            temperature=0,
        )
    else:
        raise ValueError(
            f"Unsupported chat provider: {config.chat_provider}. "
            "Must be 'gemini' or 'ollama'."
        )

    return _model_instance


def get_model_with_tools():
    """Get LLM instance with tools bound (singleton pattern)."""
    global _model_with_tools_instance
    if _model_with_tools_instance is not None:
        return _model_with_tools_instance

    from agent.tools import get_tools

    _model_with_tools_instance = get_model().bind_tools(get_tools())
    return _model_with_tools_instance


def reset_model():
    """Reset the model singletons. Useful for testing or dynamic config changes."""
    global _model_instance, _model_with_tools_instance
    _model_instance = None
    _model_with_tools_instance = None
