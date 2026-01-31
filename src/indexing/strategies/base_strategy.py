from abc import ABC, abstractmethod
from typing import List, Any
from ..schema import CodeNode

class BaseStrategy(ABC):
    @abstractmethod
    def index(self, file_path: str, nodes: List[CodeNode]) -> bool:
        """Index a list of nodes for a specific file."""
        pass

    @abstractmethod
    def delete(self, file_path: str) -> bool:
        """Delete entries for a specific file."""
        pass
