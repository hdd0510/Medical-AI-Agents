from abc import abstractmethod
from typing import Dict, Any

class Tool:
    """Base class for all tools that agents can use."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool with given parameters and return results."""
        pass
    
    def get_spec(self) -> Dict[str, Any]:
        """Return the specification of this tool for agent's reasoning."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters(),
            "returns": self._get_returns()
        }
    
    @abstractmethod
    def _get_parameters(self) -> Dict[str, Any]:
        """Get the parameters schema of this tool."""
        pass
    
    @abstractmethod
    def _get_returns(self) -> Dict[str, Any]:
        """Get the return schema of this tool."""
        pass