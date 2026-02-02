from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import plan_node, execute_node, synthesize_node, refine_node

def define_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("planner", plan_node)
    workflow.add_node("executor", execute_node)
    workflow.add_node("refinery", refine_node)
    workflow.add_node("synthesizer", synthesize_node)

    # 2. Add Edges
    workflow.set_entry_point("planner")
    
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "refinery")

    # 3. Conditional Edge for Loop
    def should_continue(state: AgentState):
        decision = state.get("loop_decision", "FINISH")
        
        if decision == "CONTINUE":
            return "planner"
        return "synthesizer"

    workflow.add_conditional_edges(
        "refinery",
        should_continue,
        {
            "planner": "planner",
            "synthesizer": "synthesizer"
        }
    )
    
    workflow.add_edge("synthesizer", END)

    # 4. Compile
    return workflow.compile()
