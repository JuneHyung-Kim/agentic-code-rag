import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from config import config
from agent.state import AgentState, ExecutorStep
from tools.search_tool import SearchTool
from tools.structure import FileSystemTools
from tools.related import RelatedCodeTool

logger = logging.getLogger(__name__)

# -- Model Factory (Singleton) --
_model_instance = None


def get_model():
    """Get or create the LLM instance (singleton pattern)."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    if config.chat_provider == "gemini":
        _model_instance = ChatGoogleGenerativeAI(
            model=config.chat_model,
            google_api_key=config.gemini_api_key,
            temperature=0,
            convert_system_message_to_human=True
        )
    elif config.chat_provider == "ollama":
        _model_instance = ChatOllama(
            base_url=config.ollama_base_url,
            model=config.chat_model,
            temperature=0
        )
    else:
        raise ValueError(f"Unsupported chat provider: {config.chat_provider}. Must be 'gemini' or 'ollama'.")

    return _model_instance


def reset_model():
    """Reset the model singleton. Useful for testing or dynamic config changes."""
    global _model_instance
    _model_instance = None


# -- Tool Registry --
AVAILABLE_TOOLS = {
    "search_codebase": {
        "description": "Search the codebase using hybrid semantic + keyword search. "
                       "Use for finding code by concepts, function names, or keywords.",
        "parameters": {
            "query": "The search query string",
            "n_results": "(optional) Number of results to return, default 5"
        }
    },
    "read_file": {
        "description": "Read the contents of a specific file. Use when you know the exact file path.",
        "parameters": {
            "file_path": "The path to the file to read"
        }
    },
    "list_directory": {
        "description": "List files and subdirectories in a directory. Use to explore project structure.",
        "parameters": {
            "path": "The directory path to list (use '.' for root)"
        }
    },
    "get_callers": {
        "description": "Find functions that call the given function. Use to understand who uses a function.",
        "parameters": {
            "function_name": "The name of the function to find callers for"
        }
    },
    "get_callees": {
        "description": "Find functions that the given function calls. Use to understand dependencies.",
        "parameters": {
            "function_name": "The name of the function to find callees for"
        }
    },
    "finish": {
        "description": "Signal that you have gathered enough information for the current plan step. "
                       "Use when observations provide sufficient context.",
        "parameters": {
            "summary": "A brief summary of what was found"
        }
    }
}


def _format_tools_for_prompt() -> str:
    """Format available tools as a string for the LLM prompt."""
    lines = []
    for name, info in AVAILABLE_TOOLS.items():
        params = ", ".join([f"{k}: {v}" for k, v in info["parameters"].items()])
        lines.append(f"- {name}({params}): {info['description']}")
    return "\n".join(lines)


# -- Tool Instances (Singleton) --
_search_tool = None
_fs_tool = None
_related_tool = None


def _get_tools():
    """Get or create tool instances (singleton pattern)."""
    global _search_tool, _fs_tool, _related_tool
    if _search_tool is None:
        _search_tool = SearchTool()
    if _fs_tool is None:
        _fs_tool = FileSystemTools(config.project_root)
    if _related_tool is None:
        _related_tool = RelatedCodeTool()
    return _search_tool, _fs_tool, _related_tool


def reset_tools():
    """Reset tool singletons. Useful for testing or when project_root changes."""
    global _search_tool, _fs_tool, _related_tool
    _search_tool = None
    _fs_tool = None
    _related_tool = None


def _execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Execute a tool and return the result as a string."""
    search_tool, fs_tool, related_tool = _get_tools()

    try:
        if tool_name == "search_codebase":
            query = tool_input.get("query", "")
            n_results = tool_input.get("n_results", 5)
            return search_tool.search_codebase(query, n_results=n_results)

        elif tool_name == "read_file":
            file_path = tool_input.get("file_path", "")
            return fs_tool.read_file(file_path)

        elif tool_name == "list_directory":
            path = tool_input.get("path", ".")
            return fs_tool.list_dir(path)

        elif tool_name == "get_callers":
            function_name = tool_input.get("function_name", "")
            return related_tool.get_callers(function_name)

        elif tool_name == "get_callees":
            function_name = tool_input.get("function_name", "")
            return related_tool.get_callees(function_name)

        elif tool_name == "finish":
            summary = tool_input.get("summary", "Research complete.")
            return f"[FINISH] {summary}"

        else:
            return f"Unknown tool: {tool_name}. Available tools: {list(AVAILABLE_TOOLS.keys())}"

    except Exception as e:
        return f"Tool execution error ({tool_name}): {str(e)}"


# -- Nodes --

def plan_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the user input and findings to create or update the research plan.
    This acts as a Replanner if findings exist.
    """
    logger.info("--- PLANNING ---")
    model = get_model()

    findings_str = "\n".join([
        f"- {k}: {str(v)[:200]}..."
        for k, v in state.get("findings", {}).items()
    ])

    iteration = state.get("iteration_count", 0)
    if iteration > 0:
        logger.info(f"Replanning (iteration {iteration})")

    system_prompt = """You are an expert software architect acting as a planner.
Your goal is to create a step-by-step research plan to answer the user's request.
Consider the User Request and any Existing Findings.

Guidelines:
- If findings are partially sufficient, create a plan to find ONLY the MISSING information.
- Keep steps granular and actionable.
- Each step should be achievable with one or two tool calls.
- If you have enough information, return a single step: ["Synthesize the final answer"]
- Return ONLY a JSON array of strings, where each string is a step.

Example: ["Search for class SearchTool", "Read the file containing SearchTool", "Find usages of the search method"]"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "User Request: {input}\n\nExisting Findings:\n{findings}")
    ])

    chain = prompt | model | JsonOutputParser()

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
        "executor_steps": [],  # Clear executor history for new plan
        "iteration_count": iteration + 1
    }


def execute_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes plan steps using ReAct-style reasoning.
    LLM chooses tools, observes results, and decides next action.
    """
    logger.info("--- EXECUTING (ReAct) ---")
    model = get_model()

    current_step_idx = state.get("current_step", 0)
    plan = state.get("plan", [])

    if not plan or current_step_idx >= len(plan):
        logger.info("No more steps to execute")
        return {"current_step": current_step_idx}

    current_step_desc = plan[current_step_idx]
    logger.info(f"Step {current_step_idx + 1}/{len(plan)}: {current_step_desc}")

    # Build ReAct context
    findings_str = "\n".join([
        f"- {k}: {str(v)[:150]}..."
        for k, v in state.get("findings", {}).items()
    ])

    executor_steps: List[ExecutorStep] = list(state.get("executor_steps", []))
    max_react_steps = config.max_executor_steps

    react_system_prompt = f"""You are a code research assistant executing a specific task.
Your current task: {current_step_desc}

Available tools:
{_format_tools_for_prompt()}

Instructions:
1. Think about what information you need
2. Choose a tool to gather that information
3. After receiving observations, decide if you need more info or can finish

Return a JSON object with:
- "thought": Your reasoning about what to do next
- "action": The tool name to use
- "action_input": Object with the tool parameters

When you have gathered enough information for this step, use the "finish" tool.

Previous findings from earlier steps:
{findings_str}"""

    for react_step in range(max_react_steps):
        # Build history of this execution's steps
        history_str = ""
        if executor_steps:
            for i, step in enumerate(executor_steps):
                history_str += f"\n--- Step {i+1} ---\n"
                history_str += f"Thought: {step['thought']}\n"
                history_str += f"Action: {step['action']}({step['action_input']})\n"
                history_str += f"Observation: {step['observation'][:500]}...\n"

        user_prompt = f"Current task: {current_step_desc}"
        if history_str:
            user_prompt += f"\n\nPrevious actions in this task:{history_str}"
            user_prompt += "\n\nBased on the observations, what should you do next?"
        else:
            user_prompt += "\n\nWhat is your first action?"

        prompt = ChatPromptTemplate.from_messages([
            ("system", react_system_prompt),
            ("user", user_prompt)
        ])

        chain = prompt | model | JsonOutputParser()

        try:
            response = chain.invoke({})
            thought = response.get("thought", "")
            action = response.get("action", "finish")
            action_input = response.get("action_input", {})

            logger.info(f"  ReAct step {react_step + 1}: {action}")
            logger.debug(f"  Thought: {thought}")

            # Execute the tool
            observation = _execute_tool(action, action_input)

            # Record this step
            step_record: ExecutorStep = {
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation
            }
            executor_steps.append(step_record)

            # Check if LLM decided to finish
            if action == "finish":
                logger.info("  ReAct loop finished by LLM")
                break

        except Exception as e:
            logger.warning(f"  ReAct step failed: {e}")
            # Record the error and continue
            step_record: ExecutorStep = {
                "thought": f"Error: {e}",
                "action": "error",
                "action_input": {},
                "observation": str(e)
            }
            executor_steps.append(step_record)
            break

    # Summarize findings from this execution
    step_findings = []
    for step in executor_steps:
        if step["action"] != "error" and step["observation"]:
            step_findings.append(step["observation"])

    combined_findings = "\n---\n".join(step_findings) if step_findings else "No results found"
    finding_key = f"step_{current_step_idx}: {current_step_desc}"

    old_findings = dict(state.get("findings", {}))
    old_findings[finding_key] = combined_findings

    return {
        "findings": old_findings,
        "current_step": current_step_idx + 1,
        "executor_steps": executor_steps
    }


def refine_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates gathered findings and decides whether to continue or finish.
    Acts as the router between Planner and Synthesizer.
    """
    logger.info("--- REFINING ---")
    model = get_model()

    # Check iteration limit
    iteration = state.get("iteration_count", 0)
    if iteration >= config.max_iterations:
        logger.warning(f"Max iterations ({config.max_iterations}) reached, forcing FINISH")
        return {"loop_decision": "FINISH"}

    findings_str = "\n".join([
        f"{k}: {str(v)[:300]}..."
        for k, v in state.get("findings", {}).items()
    ])

    current = state.get("current_step", 0)
    plan = state.get("plan", [])
    total = len(plan)

    system_prompt = """You are a project manager overseeing a code research task.
Evaluate the gathered findings against the user's original request.

Guidelines:
- FINISH if the findings provide enough information to answer the user's question
- CONTINUE if critical information is still missing
- Consider: Do we have code snippets? Do we understand the structure? Can we explain the concept?

Return a JSON object with:
- "decision": Either "CONTINUE" or "FINISH"
- "reason": Brief explanation for your decision"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "User Request: {input}\n\nFindings:\n{findings}\n\nPlan Progress: {step}/{total} steps completed")
    ])

    chain = prompt | model | JsonOutputParser()

    try:
        decision_data = chain.invoke({
            "input": state["input"],
            "findings": findings_str,
            "step": current,
            "total": total
        })
        decision = decision_data.get("decision", "FINISH")
        reason = decision_data.get("reason", "")
        logger.info(f"Refinery decision: {decision} - {reason}")
    except Exception as e:
        logger.warning(f"Refinery failed: {e}")
        # Default: finish if plan complete, continue otherwise
        decision = "CONTINUE" if current < total else "FINISH"

    return {"loop_decision": decision}


def synthesize_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizes all findings into a final answer.
    """
    logger.info("--- SYNTHESIZING ---")
    model = get_model()

    findings_str = "\n".join([
        f"### {k}\n{v}\n"
        for k, v in state.get("findings", {}).items()
    ])

    system_prompt = """You are an expert software engineer providing a comprehensive answer.
Based on the research findings, answer the user's question.

Guidelines:
- Reference specific code snippets when relevant
- Structure your answer clearly with sections if needed
- If findings are insufficient, explain what's missing
- Use markdown formatting for code blocks and emphasis"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Question: {input}\n\nResearch Findings:\n{findings}")
    ])

    chain = prompt | model | StrOutputParser()

    response = chain.invoke({"input": state["input"], "findings": findings_str})

    return {"response": response}
