import logging

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.nodes import (
    plan_node,
    setup_executor,
    executor_llm_node,
    route_executor,
    aggregate_node,
    refine_node,
    synthesize_node,
)
from agent.tools import get_tools

logger = logging.getLogger(__name__)


def route_after_aggregate(state: AgentState) -> str:
    """Conditional edge after aggregate: loop back for remaining tasks or finish.

    Returns:
        "setup_executor" if there are more tasks to execute.
        "refinery" if all tasks are done.
    """
    current = state.get("current_step", 0)
    plan = state.get("plan", [])

    if current < len(plan):
        logger.info(f"Tasks remaining: {len(plan) - current}, looping to setup_executor")
        return "setup_executor"

    logger.info("All tasks completed, proceeding to refinery")
    return "refinery"


def define_graph():
    """
    Define the LangGraph workflow for the Code RAG Agent.

    Flow:
        planner → setup_executor → executor_llm ──→ [route_executor]
                       ↑               │
                       │         "tools" → tool_node ──┘
                       │         "aggregate"
                       │               ↓
                       │           aggregate → [route_after_aggregate]
                       │              ├─ "setup_executor" → setup_executor (more tasks)
                       │              └─ "refinery" → refinery → synthesizer → END
    """
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("planner", plan_node)
    workflow.add_node("setup_executor", setup_executor)
    workflow.add_node("executor_llm", executor_llm_node)
    workflow.add_node("tool_node", ToolNode(get_tools(), handle_tool_errors=True))
    workflow.add_node("aggregate", aggregate_node)
    workflow.add_node("refinery", refine_node)
    workflow.add_node("synthesizer", synthesize_node)

    # 2. Entry point
    workflow.set_entry_point("planner")

    # 3. Linear edges
    workflow.add_edge("planner", "setup_executor")
    workflow.add_edge("setup_executor", "executor_llm")
    workflow.add_edge("tool_node", "executor_llm")

    # 4. Conditional edge: executor_llm → tools or aggregate
    workflow.add_conditional_edges(
        "executor_llm",
        route_executor,
        {
            "tools": "tool_node",
            "aggregate": "aggregate",
        },
    )

    # 5. Conditional edge: aggregate → setup_executor (more tasks) or refinery (done)
    workflow.add_conditional_edges(
        "aggregate",
        route_after_aggregate,
        {
            "setup_executor": "setup_executor",
            "refinery": "refinery",
        },
    )

    # 6. Linear edge: refinery → synthesizer (no outer loop)
    workflow.add_edge("refinery", "synthesizer")

    # 7. Terminal edge
    workflow.add_edge("synthesizer", END)

    # 8. Compile
    return workflow.compile()
