from typing import List, Dict, Any, TypedDict, Annotated, Literal
import operator

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State for the Code RAG Agent.

    Attributes:
        input (str): The original user query.
        chat_history (List[BaseMessage]): Previous conversation history + current session.
        plan (List[Dict[str, Any]]): Ordered list of Task dicts produced by the planner.
            Each dict has: goal, success_criteria, abort_criteria, suggested_tools?, context_hint?
        current_step (int): Index of the current task in the plan.
        findings (Dict[str, Any]): Key-value store of evidence gathered from tool execution.
                                   Also acts as the 'long-term memory' for the session.
        response (str): The final answer to appear in the chat.
        loop_decision (Literal["CONTINUE", "FINISH"]): Decision from refinery node.
        messages (List[BaseMessage]): ToolNode loop messages (executor_llm ↔ tool_node).
        executor_call_count (int): Number of executor LLM calls in current step (for max_executor_steps).
        iteration_count (int): Number of planner-executor-refinery cycles (for infinite loop prevention).
    """
    input: str
    chat_history: Annotated[List[BaseMessage], add_messages]
    plan: List[Dict[str, Any]]
    current_step: int
    findings: Dict[str, Any]
    response: str
    loop_decision: Literal["CONTINUE", "FINISH"]
    messages: Annotated[List[BaseMessage], add_messages]
    executor_call_count: int
    iteration_count: int
