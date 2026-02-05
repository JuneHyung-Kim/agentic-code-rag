import logging
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import plan_node, execute_node, synthesize_node, refine_node
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

    # Safety check for max iterations (also checked in refine_node)
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
        Planner (high-level plan)
            ↓
        ReAct Executor (LLM tool selection → execution → observation → repeat)
            ↓
        Refinery (evaluate if sufficient)
            ↓ CONTINUE        ↓ FINISH
        Planner ←─────────  Synthesizer
    """
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("planner", plan_node)
    workflow.add_node("executor", execute_node)
    workflow.add_node("refinery", refine_node)
    workflow.add_node("synthesizer", synthesize_node)

    # 2. Add Edges
    workflow.set_entry_point("planner")

    # Linear flow: planner → executor → refinery
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "refinery")

    # 3. Conditional Edge: refinery decides to loop or finish
    workflow.add_conditional_edges(
        "refinery",
        should_continue,
        {
            "planner": "planner",
            "synthesizer": "synthesizer"
        }
    )

    # Terminal edge
    workflow.add_edge("synthesizer", END)

    # 4. Compile
    return workflow.compile()
