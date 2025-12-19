import os
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables from .env file
load_dotenv()

class AgentConfig:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.model_provider = os.getenv("MODEL_PROVIDER", "openai") # openai or gemini
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o")
        self.project_root = os.getenv("PROJECT_ROOT", "./")
        
    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid based on selected provider."""
        if self.model_provider == "openai":
            return bool(self.openai_api_key)
        elif self.model_provider == "gemini":
            return bool(self.gemini_api_key)
        return False
    
    def validate(self) -> None:
        """Validate configuration and raise error if invalid."""
        if self.model_provider not in ["openai", "gemini"]:
            raise ValueError(f"Invalid MODEL_PROVIDER: {self.model_provider}. Must be 'openai' or 'gemini'")
        
        if not self.is_valid:
            raise ValueError(
                f"Missing API key for {self.model_provider}. "
                f"Please set {'OPENAI_API_KEY' if self.model_provider == 'openai' else 'GEMINI_API_KEY'} in .env file"
            )

config = AgentConfig()
