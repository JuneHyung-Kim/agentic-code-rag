import logging
from typing import Dict, Any, List

from config import config
from agent.graph import define_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeAgent:
    def __init__(self):
        config.validate_chat_config()
        
        # Initialize the LangGraph workflow
        self.app = define_graph()
        
        # We maintain a crude history here if needed, but the graph state handles it mostly.
        # However, for a persistent session across `chat` calls in a CLI, we might want to
        # pass history in. For now, we treat each `chat` call as a fresh inquiry 
        # or we could try to thread it.
        # Given the "stateful" goal, let's treat `chat` as a run.
        self.chat_history = [] 

    def chat(self, user_input: str) -> str:
        """
        Main entry point for the agent.
        """
        inputs = {
            "input": user_input, 
            "chat_history": self.chat_history,
            "plan": [],
            "current_step": 0,
            "findings": {}
        }
        
        try:
            # Run the graph
            # recursion_limit protects against infinite loops
            final_state = self.app.invoke(inputs, config={"recursion_limit": 25})
            
            response = final_state.get("response", "I could not generate a response.")
            
            # Update history (optional, if we want to carry over context, 
            # but current design is per-query focused)
            self.chat_history.append(("user", user_input))
            self.chat_history.append(("ai", response))
            
            return response
            
        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            return f"An error occurred: {str(e)}"

    def reset(self):
        self.chat_history = []


