from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from config import config
from agent.state import AgentState
from tools.search_tool import SearchTool
from tools.structure import FileSystemTools
from tools.related import RelatedCodeTool

# -- Model Factory --
def get_model():
    if config.chat_provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config.chat_model,
            google_api_key=config.gemini_api_key,
            temperature=0,
            convert_system_message_to_human=True 
        )
    elif config.chat_provider == "ollama":
        return ChatOllama(
            base_url=config.ollama_base_url,
            model=config.chat_model,
            temperature=0
        )
    else:
        raise ValueError(f"Unsupported provider for LangGraph: {config.chat_provider}")

# -- Nodes --

def plan_node(state: AgentState):
    """
    Analyzes the user input and findings to create or update the research plan.
    This acts as a Replanner if findings exist.
    """
    print("--- PLANNING ---")
    model = get_model()
    
    findings_str = "\n".join([f"- {k}: {str(v)[:200]}..." for k, v in state.get("findings", {}).items()])
    
    system_prompt = (
        "You are an expert software architect acting as a planner.\n"
        "Your goal is to create a step-by-step research plan to answer the user's request.\n"
        "Consider the User Request and any Existing Findings.\n"
        "If the existing findings are partially sufficient, create a plan to find the MISSING information.\n"
        "Return ONLY a JSON list of strings, where each string is a step.\n"
        "Example: [\"Search for class X\", \"Check for usages of function Y inside file Z\"]\n"
        "Keep steps granular. If you think we have enough info, return an empty list or a step saying 'FINISH'."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "User Request: {input}\n\nExisting Findings:\n{findings}")
    ])
    
    chain = prompt | model | JsonOutputParser()
    
    try:
        plan = chain.invoke({"input": state["input"], "findings": findings_str})
        # Sanitize plan
        if isinstance(plan, list):
             # Remove "FINISH" steps if they are just placeholders, 
             # but keeping them empty works for the logic too.
             pass
    except Exception as e:
        # Fallback 
        plan = ["Search for relevant code related to the query"]
        print(f"Plan generation failed: {e}")

    # Reset current_step because we have a new plan
    return {"plan": plan, "current_step": 0}

def execute_node(state: AgentState):
    """
    Executes the next step in the plan.
    """
    print("--- EXECUTING ---")
    current_step_idx = state.get("current_step", 0)
    plan = state.get("plan", [])
    
    if not plan or current_step_idx >= len(plan):
        # No more steps to execute
        return {"current_step": current_step_idx}

    current_step_desc = plan[current_step_idx]
    print(f"Step {current_step_idx + 1}: {current_step_desc}")

    # Tool Selection Logic (Simple Heuristic for now)
    # In a real impl, we'd use an LLM or Router here.
    
    # Initialize tools
    search_tool = SearchTool()
    fs_tool = FileSystemTools(config.project_root)
    related_tool = RelatedCodeTool()
    
    step_lower = current_step_desc.lower()
    
    try:
        if "list" in step_lower and ("dir" in step_lower or "files" in step_lower):
            # precise path extraction or just root
            results = fs_tool.list_dir(".") 
        elif "read" in step_lower and "file" in step_lower:
             # Very naive extraction, TODO: use LLM for tool args
            results = "Please specify the file path clearly."
            # Fallback to search if specific file not clear
            results = search_tool.search_codebase(current_step_desc, n_results=1)
        elif "call" in step_lower or "usage" in step_lower or "related" in step_lower:
             # Again, naive. TODO: use related_tool properly
             # results = related_tool.get_related(...)
             results = search_tool.search_codebase(current_step_desc, n_results=3)
        else:
            results = search_tool.search_codebase(current_step_desc, n_results=3)
    except Exception as e:
        results = f"Tool execution failed: {e}"
    
    new_findings = {f"step_{current_step_idx}: {current_step_desc}": results}
    
    # Merge findings
    old_findings = state.get("findings", {})
    old_findings.update(new_findings)

    return {
        "findings": old_findings, 
        "current_step": current_step_idx + 1
    }

def refine_node(state: AgentState):
    """
    Refines the plan. Checks if we have enough info or need to loop back.
    Acts as the 'Router' between Planner and Synthesizer.
    """
    print("--- REFINING ---")
    model = get_model()
    
    findings_str = "\n".join([f"{k}: {str(v)[:300]}..." for k, v in state.get("findings", {}).items()])
    
    system_prompt = (
        "You are a project manager overseeing a code research task.\n"
        "Evaluate the gathered findings against the user's request.\n"
        "Return a JSON object with a single key 'decision' which is either 'CONTINUE' (need more info) or 'FINISH' (sufficient info).\n"
        "If the plan is empty or finished, and you still need info, say CONTINUE."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "User Request: {input}\n\nFindings:\n{findings}\n\nCurrent Plan Progress: {step}/{total}")
    ])
    
    chain = prompt | model | JsonOutputParser()
    
    current = state.get("current_step", 0)
    total = len(state.get("plan", []))
    
    try:
        decision_data = chain.invoke({
            "input": state["input"], 
            "findings": findings_str,
            "step": current,
            "total": total
        })
        decision = decision_data.get("decision", "FINISH")
    except Exception:
        # If ambiguous, and we finished the plan, we finish. If plan remains, we continue.
        decision = "CONTINUE" if current < total else "FINISH"

    # We store the decision in a temporary state key if needed, 
    # but strictly we can just return it to the condition.
    # We'll just return it as a finding or check it in the edge.
    # For now, let's append a special finding or just rely on the graph edge to call this node?
    # Wait, 'refine_node' is a node. It updates state.
    # We should add a 'loop_decision' to state?
    # Or we can just do the logic in the conditional edge function?
    # The user REQUESTED a 'Refinery' node. So let's make it a node that updates state.
    
    return {"loop_decision": decision}

def synthesize_node(state: AgentState):
    """
    Synthesizes all findings into a final answer.
    """
    print("--- SYNTHESIZING ---")
    model = get_model()
    
    findings_str = "\n".join([f"{k}: {v}" for k, v in state.get("findings", {}).items()])
    
    system_prompt = (
        "You are an expert software engineer. Answer the user's question based on the provided research findings.\n"
        "Reference the findings explicitly. If the findings are insufficient, state what is missing.\n"
        "Format your answer in Markdown."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Question: {input}\n\nFindings:\n{findings}")
    ])
    
    chain = prompt | model | StrOutputParser()
    
    response = chain.invoke({"input": state["input"], "findings": findings_str})
    
    return {"response": response}
