import logging
from typing import Any, Dict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser

from config import config
from agent.state import AgentState
from agent.model import get_model, get_model_with_tools
from profiling.profile_store import get_codebase_context
from agent.prompts import (
    PLAN_PROMPT,
    EXECUTOR_PROMPT,
    REFINE_PROMPT,
    SYNTHESIZE_PROMPT,
    PlanOutput,
    RefineDecision,
)

logger = logging.getLogger(__name__)


# -- Helper ------------------------------------------------------------------

def _format_findings(findings: Dict[str, Any], max_len: int = 200) -> str:
    return "\n".join(
        f"- {k}: {str(v)[:max_len]}..." for k, v in findings.items()
    )


# -- Nodes -------------------------------------------------------------------

def plan_node(state: AgentState) -> Dict[str, Any]:
    """Create or update the research plan based on user input and existing findings."""
    logger.info("--- PLANNING ---")
    model = get_model()

    findings_str = _format_findings(state.get("findings", {}))

    iteration = state.get("iteration_count", 0)
    if iteration > 0:
        logger.info(f"Replanning (iteration {iteration})")

    codebase_ctx = get_codebase_context() or "No codebase profile available. Run 'init' to generate one."

    chain = PLAN_PROMPT | model.with_structured_output(PlanOutput)

    try:
        result = chain.invoke({
            "input": state["input"],
            "findings": findings_str,
            "codebase_context": codebase_ctx,
        })
        plan = result.steps
        if not plan:
            plan = ["Search for relevant code related to the query"]
        logger.info(f"Generated plan with {len(plan)} steps: {plan}")
    except Exception as e:
        plan = ["Search for relevant code related to the query"]
        logger.warning(f"Plan generation failed: {e}")

    return {
        "plan": plan,
        "current_step": 0,
        "iteration_count": iteration + 1,
    }


def setup_executor(state: AgentState) -> Dict[str, Any]:
    """Initialize messages for the executor LLM ↔ ToolNode loop.

    Clears previous messages via RemoveMessage, then seeds with
    SystemMessage (task context) and HumanMessage (task instruction).
    """
    logger.info("--- SETUP EXECUTOR ---")

    current_step_idx = state.get("current_step", 0)
    plan = state.get("plan", [])

    if not plan or current_step_idx >= len(plan):
        current_step_desc = "No more steps to execute"
    else:
        current_step_desc = plan[current_step_idx]

    logger.info(f"Step {current_step_idx + 1}/{len(plan)}: {current_step_desc}")

    findings_str = _format_findings(state.get("findings", {}), max_len=150)

    # Clear old messages
    remove_msgs = [RemoveMessage(id=m.id) for m in state.get("messages", []) if m.id]

    # Build fresh messages from prompt template
    formatted = EXECUTOR_PROMPT.format_messages(
        current_step=current_step_desc,
        findings=findings_str,
    )
    sys_msg = formatted[0]
    user_msg = formatted[1]

    return {
        "messages": remove_msgs + [sys_msg, user_msg],
        "executor_call_count": 0,
    }


def executor_llm_node(state: AgentState) -> Dict[str, Any]:
    """Invoke the LLM with tools bound. Returns AIMessage (may contain tool_calls)."""
    logger.info("--- EXECUTOR LLM ---")
    model = get_model_with_tools()

    response = model.invoke(state["messages"])

    call_count = state.get("executor_call_count", 0)
    if response.tool_calls:
        logger.info(f"  Tool calls: {[tc['name'] for tc in response.tool_calls]}")
    else:
        logger.info("  No tool calls — will aggregate")

    return {
        "messages": [response],
        "executor_call_count": call_count + 1,
    }


def route_executor(state: AgentState) -> str:
    """Conditional edge: route to tool_node or aggregate.

    Returns 'tools' if the last AIMessage has tool_calls and we haven't
    exceeded max_executor_steps. Otherwise returns 'aggregate'.
    """
    messages = state.get("messages", [])
    call_count = state.get("executor_call_count", 0)
    max_steps = config.max_executor_steps

    if messages:
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls and call_count < max_steps:
            return "tools"

    if call_count >= max_steps:
        logger.warning(f"Max executor steps ({max_steps}) reached, forcing aggregate")

    return "aggregate"


def aggregate_node(state: AgentState) -> Dict[str, Any]:
    """Extract findings from tool messages and advance current_step.

    Collects content from ToolMessage entries and the final AIMessage summary,
    stores them in findings, then clears messages.
    """
    logger.info("--- AGGREGATE ---")

    messages = state.get("messages", [])
    current_step_idx = state.get("current_step", 0)
    plan = state.get("plan", [])
    current_step_desc = plan[current_step_idx] if current_step_idx < len(plan) else "unknown"

    # Collect tool results and the final AI summary
    parts = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.content:
            parts.append(msg.content)
        elif isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            parts.append(msg.content)

    combined = "\n---\n".join(parts) if parts else "No results found"
    finding_key = f"step_{current_step_idx}: {current_step_desc}"

    old_findings = dict(state.get("findings", {}))
    old_findings[finding_key] = combined

    # Clear messages for next step
    remove_msgs = [RemoveMessage(id=m.id) for m in messages if m.id]

    return {
        "findings": old_findings,
        "current_step": current_step_idx + 1,
        "messages": remove_msgs,
    }


def refine_node(state: AgentState) -> Dict[str, Any]:
    """Evaluate findings and decide whether to continue research or finish."""
    logger.info("--- REFINING ---")
    model = get_model()

    iteration = state.get("iteration_count", 0)
    if iteration >= config.max_iterations:
        logger.warning(f"Max iterations ({config.max_iterations}) reached, forcing FINISH")
        return {"loop_decision": "FINISH"}

    findings_str = _format_findings(state.get("findings", {}), max_len=300)
    current = state.get("current_step", 0)
    total = len(state.get("plan", []))

    chain = REFINE_PROMPT | model.with_structured_output(RefineDecision)

    try:
        result = chain.invoke({
            "input": state["input"],
            "findings": findings_str,
            "step": current,
            "total": total,
        })
        decision = result.decision
        reason = result.reason
        logger.info(f"Refinery decision: {decision} - {reason}")
    except Exception as e:
        logger.warning(f"Refinery failed: {e}")
        decision = "CONTINUE" if current < total else "FINISH"

    return {"loop_decision": decision}


def synthesize_node(state: AgentState) -> Dict[str, Any]:
    """Synthesize all findings into a final answer."""
    logger.info("--- SYNTHESIZING ---")
    model = get_model()

    findings_str = "\n".join(
        f"### {k}\n{v}\n" for k, v in state.get("findings", {}).items()
    )

    codebase_ctx = get_codebase_context() or "No codebase profile available. Run 'init' to generate one."

    chain = SYNTHESIZE_PROMPT | model | StrOutputParser()
    response = chain.invoke({
        "input": state["input"],
        "findings": findings_str,
        "codebase_context": codebase_ctx,
    })

    return {"response": response}
