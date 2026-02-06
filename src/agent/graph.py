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
from config import config

logger = logging.getLogger(__name__)


def should_continue(state: AgentState) -> str:
    """
    Conditional edge function: decides whether to continue planning or synthesize.

    Returns:
        "planner" if more research is needed
        "synthesizer" if ready to generate final answer
    """
    decision = state.get("loop_decision", "FINISH")
    iteration = state.get("iteration_count", 0)

    if iteration >= config.max_iterations:
        logger.warning(f"Max iterations ({config.max_iterations}) reached in edge check")
        return "synthesizer"

    if decision == "CONTINUE":
        logger.info(f"Continuing to planner (iteration {iteration})")
        return "planner"

    logger.info("Proceeding to synthesizer")
    return "synthesizer"


def define_graph():
    """
    Define the LangGraph workflow for the Code RAG Agent.

    Flow:
        planner → setup_executor → executor_llm ──→ [route_executor]
                                       ↑               │
                                       │         "tools" → tool_node ──┘
                                       │         "aggregate"
                                       │               ↓
                                       │           aggregate → refinery → [should_continue]
                                                                 ├─ "planner" → planner
                                                                 └─ "synthesizer" → synthesizer → END
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

    # 5. Linear edge: aggregate → refinery
    workflow.add_edge("aggregate", "refinery")

    # 6. Conditional edge: refinery → planner or synthesizer
    workflow.add_conditional_edges(
        "refinery",
        should_continue,
        {
            "planner": "planner",
            "synthesizer": "synthesizer",
        },
    )

    # 7. Terminal edge
    workflow.add_edge("synthesizer", END)

    # 8. Compile
    return workflow.compile()
