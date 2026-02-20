import os
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables from .env file
load_dotenv()

class AgentConfig:
    def __init__(self):
        # API Keys
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Embedding Model Configuration
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "default")  # openai, gemini, default, ollama
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # or mxbai-embed-large for ollama
        
        # Chat Model Configuration
        self.chat_provider = os.getenv("CHAT_PROVIDER", "gemini")  # openai, gemini, ollama
        self.chat_model = os.getenv("CHAT_MODEL", "gemini-1.5-flash")  # or llama3.2, qwen2.5, etc for ollama
        
        # Ollama Configuration
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Agent loop configuration
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "10"))  # Max planner-executor-refinery cycles
        self.max_executor_steps = int(os.getenv("MAX_EXECUTOR_STEPS", "5"))  # Max ReAct steps per execution

        # Legacy support (deprecated but kept for backward compatibility)
        legacy_provider = os.getenv("MODEL_PROVIDER")
        if legacy_provider and not os.getenv("CHAT_PROVIDER"):
            self.chat_provider = legacy_provider
        legacy_model = os.getenv("MODEL_NAME")
        if legacy_model and not os.getenv("CHAT_MODEL"):
            self.chat_model = legacy_model
            
        self.project_root = os.getenv("PROJECT_ROOT", "./")
        self.confirm_api_calls = os.getenv("CONFIRM_API_CALLS", "true").lower() == "true"
    
    def validate_embedding_config(self) -> None:
        """Validate embedding configuration."""
        if self.embedding_provider not in ["gemini", "default", "ollama"]:
            raise ValueError(
                f"Invalid EMBEDDING_PROVIDER: {self.embedding_provider}. "
                f"Must be one of: gemini, default, ollama"
            )

        if self.embedding_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("EMBEDDING_PROVIDER is 'gemini' but GEMINI_API_KEY is not set")
    
    def validate_chat_config(self) -> None:
        """Validate chat configuration."""
        if self.chat_provider not in ["gemini", "ollama", "claude"]:
            raise ValueError(
                f"Invalid CHAT_PROVIDER: {self.chat_provider}. "
                f"Must be one of: gemini, ollama, claude"
            )

        if self.chat_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("CHAT_PROVIDER is 'gemini' but GEMINI_API_KEY is not set")

        if self.chat_provider == "claude" and not self.anthropic_api_key:
            raise ValueError("CHAT_PROVIDER is 'claude' but ANTHROPIC_API_KEY is not set")

config = AgentConfig()
