from typing import List, Dict, Any, TypedDict, Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def empty_working_memory() -> Dict[str, Any]:
    """Return a fresh working memory structure.

    Keys:
        discovered_entities: list of code entities found (functions, classes, files).
        relationships: list of relationships between entities (calls, imports, etc.).
        insights: list of textual insights / observations.
        task_results: list of per-task result summaries.
    """
    return {
        "discovered_entities": [],
        "relationships": [],
        "insights": [],
        "task_results": [],
    }


class AgentState(TypedDict):
    """
    State for the Code RAG Agent.

    Attributes:
        input (str): The original user query.
        plan (List[Dict[str, Any]]): Ordered list of Task dicts produced by the planner.
            Each dict has: goal, success_criteria, abort_criteria, suggested_tools?, context_hint?
        current_step (int): Index of the current task in the plan.
        working_memory (Dict[str, Any]): Structured session memory accumulated across tasks.
            Keys: discovered_entities, relationships, insights, task_results.
        response (str): The final answer to appear in the chat.
        messages (List[BaseMessage]): ToolNode loop messages (executor_llm <-> tool_node).
        executor_call_count (int): Number of executor LLM calls in current step (for max_executor_steps).
        iteration_count (int): Number of planner invocations (for observability).
    """
    input: str
    plan: List[Dict[str, Any]]
    current_step: int
    working_memory: Dict[str, Any]
    response: str
    messages: Annotated[List[BaseMessage], add_messages]
    executor_call_count: int
    iteration_count: int
