import os
from typing import List, Optional

class AgentConfig:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.model_provider = os.getenv("MODEL_PROVIDER", "openai") # openai or gemini
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o")
        self.project_root = os.getenv("PROJECT_ROOT", "./")
        
    @property
    def is_valid(self) -> bool:
        if self.model_provider == "openai":
            return bool(self.openai_api_key)
        elif self.model_provider == "gemini":
            return bool(self.gemini_api_key)
        return False

config = AgentConfig()
