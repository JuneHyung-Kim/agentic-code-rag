import logging
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage

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
            "findings": {},
            "messages": [],
            "executor_call_count": 0,
        }
        
        try:
            # Run the graph
            # recursion_limit protects against infinite loops
            final_state = self.app.invoke(inputs, config={"recursion_limit": 150})
            
            response = final_state.get("response", "I could not generate a response.")
            
            # Update history for multi-turn conversations
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=response))
            
            return response
            
        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            return f"An error occurred: {str(e)}"

    def reset(self):
        self.chat_history = []


