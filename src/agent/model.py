import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from config import config

logger = logging.getLogger(__name__)

_model_instance = None
_model_with_tools_instance = None
_auto_approve = False


class APICallCancelled(Exception):
    """Raised when user declines an API call in confirmation prompt."""
    pass


def _wrap_with_confirmation(model):
    """Monkey-patch model.invoke() to ask user confirmation before each API call.

    Covers all call patterns:
    - prompt | model | parser (LCEL chains call model.invoke internally)
    - model.with_structured_output(Schema) -> RunnableBinding(bound=model) -> bound.invoke()
    - model.bind_tools(tools) -> same RunnableBinding pattern
    - model.invoke(messages) direct calls
    """
    original_invoke = model.invoke

    def confirmed_invoke(*args, **kwargs):
        global _auto_approve
        if _auto_approve:
            return original_invoke(*args, **kwargs)

        messages = args[0] if args else kwargs.get("input", [])
        msg_count = len(messages) if isinstance(messages, list) else "?"
        print(f"\n[API Call] {config.chat_provider}/{config.chat_model} (messages: {msg_count})")
        choice = input("Proceed? [Y/n/all]: ").strip().lower()
        if choice == "n":
            raise APICallCancelled("API call cancelled by user")
        if choice == "all":
            _auto_approve = True
        return original_invoke(*args, **kwargs)

    # Bypass Pydantic's __setattr__ which rejects unknown fields
    object.__setattr__(model, "invoke", confirmed_invoke)
    return model


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

    if config.confirm_api_calls:
        _wrap_with_confirmation(_model_instance)

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
    global _model_instance, _model_with_tools_instance, _auto_approve
    _model_instance = None
    _model_with_tools_instance = None
    _auto_approve = False
