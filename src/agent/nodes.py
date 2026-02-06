import logging
from typing import Any, Dict, List

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import config
from agent.state import AgentState, ExecutorStep
from agent.model import get_model
from agent.prompts import (
    PLAN_PROMPT,
    REACT_SYSTEM,
    REACT_USER,
    REFINE_PROMPT,
    SYNTHESIZE_PROMPT,
)
from agent.tools import tool_registry

logger = logging.getLogger(__name__)


# -- Helper ------------------------------------------------------------------

def _format_findings(findings: Dict[str, Any], max_len: int = 200) -> str:
    return "\n".join(
        f"- {k}: {str(v)[:max_len]}..." for k, v in findings.items()
    )


def _build_react_history(executor_steps: List[ExecutorStep]) -> str:
    if not executor_steps:
        return ""
    parts = []
    for i, step in enumerate(executor_steps):
        parts.append(f"\n--- Step {i + 1} ---")
        parts.append(f"Thought: {step['thought']}")
        parts.append(f"Action: {step['action']}({step['action_input']})")
        parts.append(f"Observation: {step['observation'][:500]}...")
    return "".join(parts)


# -- Nodes -------------------------------------------------------------------

def plan_node(state: AgentState) -> Dict[str, Any]:
    """Create or update the research plan based on user input and existing findings."""
    logger.info("--- PLANNING ---")
    model = get_model()

    findings_str = _format_findings(state.get("findings", {}))

    iteration = state.get("iteration_count", 0)
    if iteration > 0:
        logger.info(f"Replanning (iteration {iteration})")

    chain = PLAN_PROMPT | model | JsonOutputParser()

    try:
        plan = chain.invoke({"input": state["input"], "findings": findings_str})
        if not isinstance(plan, list):
            plan = ["Search for relevant code related to the query"]
        logger.info(f"Generated plan with {len(plan)} steps: {plan}")
    except Exception as e:
        plan = ["Search for relevant code related to the query"]
        logger.warning(f"Plan generation failed: {e}")

    return {
        "plan": plan,
        "current_step": 0,
        "executor_steps": [],
        "iteration_count": iteration + 1,
    }


def execute_node(state: AgentState) -> Dict[str, Any]:
    """Execute plan steps using ReAct-style reasoning with tool calls."""
    logger.info("--- EXECUTING (ReAct) ---")
    model = get_model()

    current_step_idx = state.get("current_step", 0)
    plan = state.get("plan", [])

    if not plan or current_step_idx >= len(plan):
        logger.info("No more steps to execute")
        return {"current_step": current_step_idx}

    current_step_desc = plan[current_step_idx]
    logger.info(f"Step {current_step_idx + 1}/{len(plan)}: {current_step_desc}")

    findings_str = _format_findings(state.get("findings", {}), max_len=150)
    executor_steps: List[ExecutorStep] = list(state.get("executor_steps", []))
    max_react_steps = config.max_executor_steps

    # Build the system prompt with runtime values
    react_system = REACT_SYSTEM.format(
        current_step=current_step_desc,
        tools=tool_registry.format_for_prompt(),
        findings=findings_str,
    )

    for react_step in range(max_react_steps):
        history_str = _build_react_history(executor_steps)

        if history_str:
            history_section = (
                f"Previous actions in this task:{history_str}\n\n"
                "Based on the observations, what should you do next?"
            )
        else:
            history_section = "What is your first action?"

        user_prompt = REACT_USER.format(
            current_step=current_step_desc,
            history_section=history_section,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", react_system),
            ("user", user_prompt),
        ])

        chain = prompt | model | JsonOutputParser()

        try:
            response = chain.invoke({})
            thought = response.get("thought", "")
            action = response.get("action", "finish")
            action_input = response.get("action_input", {})

            logger.info(f"  ReAct step {react_step + 1}: {action}")
            logger.debug(f"  Thought: {thought}")

            observation = tool_registry.execute(action, action_input)

            step_record: ExecutorStep = {
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation,
            }
            executor_steps.append(step_record)

            if action == "finish":
                logger.info("  ReAct loop finished by LLM")
                break

        except Exception as e:
            logger.warning(f"  ReAct step failed: {e}")
            step_record: ExecutorStep = {
                "thought": f"Error: {e}",
                "action": "error",
                "action_input": {},
                "observation": str(e),
            }
            executor_steps.append(step_record)
            break

    # Summarize findings from this execution
    step_findings = [
        s["observation"] for s in executor_steps
        if s["action"] != "error" and s["observation"]
    ]
    combined_findings = "\n---\n".join(step_findings) if step_findings else "No results found"
    finding_key = f"step_{current_step_idx}: {current_step_desc}"

    old_findings = dict(state.get("findings", {}))
    old_findings[finding_key] = combined_findings

    return {
        "findings": old_findings,
        "current_step": current_step_idx + 1,
        "executor_steps": executor_steps,
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

    chain = REFINE_PROMPT | model | JsonOutputParser()

    try:
        decision_data = chain.invoke({
            "input": state["input"],
            "findings": findings_str,
            "step": current,
            "total": total,
        })
        decision = decision_data.get("decision", "FINISH")
        reason = decision_data.get("reason", "")
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

    chain = SYNTHESIZE_PROMPT | model | StrOutputParser()
    response = chain.invoke({"input": state["input"], "findings": findings_str})

    return {"response": response}
