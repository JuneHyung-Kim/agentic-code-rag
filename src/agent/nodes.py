import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser

from config import config
from agent.state import AgentState, empty_working_memory
from agent.model import get_model, get_model_with_tools
from agent.tools import get_tools
from profiling.profile_store import get_codebase_context
from agent.prompts import (
    PLAN_PROMPT,
    EXECUTOR_PROMPT,
    REFINE_PROMPT,
    SYNTHESIZE_PROMPT,
    AGGREGATE_PROMPT,
    Task,
    PlanOutput,
    RefineDecision,
)

logger = logging.getLogger(__name__)

PLAN_LOG_DIR = Path("./logs/plans")


# -- Helpers -----------------------------------------------------------------

def _deep_copy_working_memory(wm: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of working memory to avoid mutating state."""
    return copy.deepcopy(wm)


def _format_working_memory(wm: Dict[str, Any], max_entities: int = 20) -> str:
    """Format working memory into a compact summary for executor/planner context.

    Shows entity count, key relationships, recent insights, and task result summaries.
    """
    parts = []

    entities = wm.get("discovered_entities", [])
    if entities:
        shown = entities[:max_entities]
        names = [e.get("name", str(e)) if isinstance(e, dict) else str(e) for e in shown]
        parts.append(f"Entities ({len(entities)}): {', '.join(names)}")
        if len(entities) > max_entities:
            parts.append(f"  ... and {len(entities) - max_entities} more")

    rels = wm.get("relationships", [])
    if rels:
        shown = rels[:10]
        for r in shown:
            if isinstance(r, dict):
                parts.append(f"  {r.get('source', '?')} --{r.get('type', '?')}--> {r.get('target', '?')}")
            else:
                parts.append(f"  {r}")
        if len(rels) > 10:
            parts.append(f"  ... and {len(rels) - 10} more relationships")

    insights = wm.get("insights", [])
    if insights:
        parts.append("Insights:")
        for ins in insights[-5:]:
            parts.append(f"  - {ins}")

    task_results = wm.get("task_results", [])
    if task_results:
        parts.append("Completed tasks:")
        for tr in task_results:
            parts.append(f"  - [{tr.get('task', '?')}]: {tr.get('summary', '')}")

    return "\n".join(parts) if parts else "No findings yet."


def _format_working_memory_for_synthesis(wm: Dict[str, Any]) -> str:
    """Format working memory as a full dump for the synthesizer node.

    Provides all details without truncation so the synthesizer can produce
    a comprehensive answer.
    """
    parts = []

    entities = wm.get("discovered_entities", [])
    if entities:
        parts.append(f"## Discovered Entities ({len(entities)})")
        for e in entities:
            if isinstance(e, dict):
                name = e.get("name", "unknown")
                etype = e.get("type", "")
                loc = e.get("location", "")
                parts.append(f"- **{name}** ({etype}) {f'@ {loc}' if loc else ''}")
            else:
                parts.append(f"- {e}")

    rels = wm.get("relationships", [])
    if rels:
        parts.append(f"\n## Relationships ({len(rels)})")
        for r in rels:
            if isinstance(r, dict):
                parts.append(f"- {r.get('source', '?')} --{r.get('type', '?')}--> {r.get('target', '?')}")
            else:
                parts.append(f"- {r}")

    insights = wm.get("insights", [])
    if insights:
        parts.append("\n## Insights")
        for ins in insights:
            parts.append(f"- {ins}")

    task_results = wm.get("task_results", [])
    if task_results:
        parts.append("\n## Task Results")
        for tr in task_results:
            parts.append(f"### {tr.get('task', 'Unknown task')}")
            parts.append(tr.get("summary", "No summary"))

    return "\n".join(parts) if parts else "No findings accumulated."


def _save_plan_log(
    query: str,
    tasks: List[Dict[str, Any]],
    iteration: int,
    codebase_ctx: str,
) -> Path:
    """Persist the generated plan to a JSON file for observability."""
    PLAN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = PLAN_LOG_DIR / f"plan_{timestamp}_iter{iteration}.json"

    log_data = {
        "timestamp": timestamp,
        "iteration": iteration,
        "query": query,
        "tasks": tasks,
        "codebase_context_length": len(codebase_ctx),
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Plan saved to {log_path}")
    return log_path


def _task_to_dict(task: Task) -> Dict[str, Any]:
    """Convert a Task pydantic model to a plain dict for state storage."""
    return task.model_dump(exclude_none=True)


def _build_tools_summary() -> str:
    """Build a tool summary string from the live tool registry."""
    lines = []
    for t in get_tools():
        # First sentence of description is enough for the planner
        desc = t.description.split("\n")[0].strip()
        lines.append(f"- {t.name}: {desc}")
    return "\n".join(lines)


def _get_current_task(state: AgentState) -> Dict[str, Any]:
    """Get the current task dict from state, with safe defaults."""
    plan = state.get("plan", [])
    idx = state.get("current_step", 0)
    if not plan or idx >= len(plan):
        return {
            "goal": "No more tasks to execute",
            "success_criteria": "N/A",
            "abort_criteria": "N/A",
        }
    return plan[idx]


def _merge_entities(existing: List[Dict], new_entities: List[Dict]) -> List[Dict]:
    """Merge new entities into existing list, deduplicating by name."""
    seen = {e.get("name") for e in existing if isinstance(e, dict) and e.get("name")}
    merged = list(existing)
    for ent in new_entities:
        name = ent.get("name") if isinstance(ent, dict) else None
        if name and name not in seen:
            seen.add(name)
            merged.append(ent)
        elif not name:
            merged.append(ent)
    return merged


def _merge_relationships(existing: List[Dict], new_rels: List[Dict]) -> List[Dict]:
    """Merge new relationships into existing list, deduplicating by (source, type, target)."""
    seen = set()
    for r in existing:
        if isinstance(r, dict):
            key = (r.get("source"), r.get("type"), r.get("target"))
            seen.add(key)
    merged = list(existing)
    for r in new_rels:
        if isinstance(r, dict):
            key = (r.get("source"), r.get("type"), r.get("target"))
            if key not in seen:
                seen.add(key)
                merged.append(r)
        else:
            merged.append(r)
    return merged


# -- Nodes -------------------------------------------------------------------

def plan_node(state: AgentState) -> Dict[str, Any]:
    """Create a structured research plan based on user input and codebase profile."""
    logger.info("--- PLANNING ---")
    model = get_model()

    wm = state.get("working_memory", empty_working_memory())
    wm_str = _format_working_memory(wm)

    iteration = state.get("iteration_count", 0)
    if iteration > 0:
        logger.info(f"Replanning (iteration {iteration})")

    codebase_ctx = get_codebase_context() or "No codebase profile available. Run 'init' to generate one."

    chain = PLAN_PROMPT | model.with_structured_output(PlanOutput)

    fallback_task = Task(
        goal="Search for relevant code related to the query",
        success_criteria="Found at least one relevant code snippet",
        abort_criteria="No results after 2 search attempts",
    )

    tools_summary = _build_tools_summary()

    try:
        result = chain.invoke({
            "input": state["input"],
            "working_memory": wm_str,
            "codebase_context": codebase_ctx,
            "available_tools_summary": tools_summary,
        })
        tasks = result.tasks
        if not tasks:
            tasks = [fallback_task]
        logger.info(f"Generated plan with {len(tasks)} tasks:")
        for i, t in enumerate(tasks):
            logger.info(f"  Task {i+1}: {t.goal}")
    except Exception as e:
        tasks = [fallback_task]
        logger.warning(f"Plan generation failed: {e}")

    task_dicts = [_task_to_dict(t) for t in tasks]

    # Save plan for observability
    _save_plan_log(state["input"], task_dicts, iteration, codebase_ctx)

    return {
        "plan": task_dicts,
        "current_step": 0,
        "iteration_count": iteration + 1,
    }


def setup_executor(state: AgentState) -> Dict[str, Any]:
    """Initialize messages for the executor LLM / ToolNode loop.

    Clears previous messages via RemoveMessage, then seeds with
    SystemMessage (task context) and HumanMessage (task instruction).
    """
    logger.info("--- SETUP EXECUTOR ---")

    task = _get_current_task(state)
    plan = state.get("plan", [])
    idx = state.get("current_step", 0)

    logger.info(f"Task {idx + 1}/{len(plan)}: {task['goal']}")

    wm = state.get("working_memory", empty_working_memory())
    wm_str = _format_working_memory(wm)

    # Clear old messages
    remove_msgs = [RemoveMessage(id=m.id) for m in state.get("messages", []) if m.id]

    # Build fresh messages from prompt template
    formatted = EXECUTOR_PROMPT.format_messages(
        task_goal=task["goal"],
        task_success_criteria=task["success_criteria"],
        task_abort_criteria=task["abort_criteria"],
        task_suggested_tools=", ".join(task.get("suggested_tools", [])) or "none specified",
        task_context_hint=task.get("context_hint") or "none",
        working_memory=wm_str,
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

    Collects structured artifacts from ToolMessage entries to update working_memory,
    and stores the AI summary in task_results.
    """
    logger.info("--- AGGREGATE ---")

    messages = state.get("messages", [])
    task = _get_current_task(state)
    current_step_idx = state.get("current_step", 0)

    wm = _deep_copy_working_memory(state.get("working_memory", empty_working_memory()))

    # Extract artifacts from ToolMessages and collect text parts
    text_parts = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # Extract structured artifact if present
            artifact = getattr(msg, "artifact", None)
            if artifact and isinstance(artifact, dict):
                new_entities = artifact.get("entities", [])
                new_rels = artifact.get("relationships", [])
                if new_entities:
                    wm["discovered_entities"] = _merge_entities(
                        wm["discovered_entities"], new_entities
                    )
                if new_rels:
                    wm["relationships"] = _merge_relationships(
                        wm["relationships"], new_rels
                    )
            if msg.content:
                text_parts.append(msg.content)
        elif isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            # Final AI summary — treat as an insight
            wm["insights"].append(msg.content)
            text_parts.append(msg.content)

    combined = "\n---\n".join(text_parts) if text_parts else "No results found"

    # Summarize findings via LLM
    if combined != "No results found":
        try:
            model = get_model()
            chain = AGGREGATE_PROMPT | model | StrOutputParser()
            summary = chain.invoke({
                "task_goal": task["goal"],
                "success_criteria": task["success_criteria"],
                "findings_text": combined,
            })
        except Exception as e:
            logger.warning(f"Aggregate LLM summarization failed: {e}")
            summary = combined
    else:
        summary = combined

    # Record task result
    wm["task_results"].append({
        "task": task["goal"],
        "step_index": current_step_idx,
        "summary": summary,
    })

    logger.info(
        f"  Working memory: {len(wm['discovered_entities'])} entities, "
        f"{len(wm['relationships'])} relationships, "
        f"{len(wm['insights'])} insights, "
        f"{len(wm['task_results'])} task results"
    )

    # Clear messages for next step
    remove_msgs = [RemoveMessage(id=m.id) for m in messages if m.id]

    return {
        "working_memory": wm,
        "current_step": current_step_idx + 1,
        "messages": remove_msgs,
    }


def refine_node(state: AgentState) -> Dict[str, Any]:
    """Logging-only pass-through before synthesis.

    The outer loop has been removed; this node now serves as an
    observability checkpoint between aggregation and synthesis.
    """
    logger.info("--- REFINERY ---")

    wm = state.get("working_memory", empty_working_memory())
    current = state.get("current_step", 0)
    total = len(state.get("plan", []))

    logger.info(
        f"  Completed {current}/{total} tasks | "
        f"{len(wm.get('discovered_entities', []))} entities | "
        f"{len(wm.get('task_results', []))} results"
    )

    return {}


def synthesize_node(state: AgentState) -> Dict[str, Any]:
    """Synthesize all findings into a final answer."""
    logger.info("--- SYNTHESIZING ---")
    model = get_model()

    wm = state.get("working_memory", empty_working_memory())
    wm_str = _format_working_memory_for_synthesis(wm)

    codebase_ctx = get_codebase_context() or "No codebase profile available. Run 'init' to generate one."

    chain = SYNTHESIZE_PROMPT | model | StrOutputParser()
    response = chain.invoke({
        "input": state["input"],
        "working_memory": wm_str,
        "codebase_context": codebase_ctx,
    })

    return {"response": response}
