#!/usr/bin/env python3
"""Monitor agent execution in real-time using LangGraph streaming.

Requires API keys — makes real LLM calls.

Usage:
    python scripts/monitor_agent.py "How does the search engine work?"
    python scripts/monitor_agent.py "What is SearchTool?" --verbose
"""

import argparse
import json
import sys
import os
import time

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def format_value(value, max_len=200):
    """Format a value for display, truncating if needed."""
    s = str(value)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def monitor(query, verbose=False):
    """Run the agent with streaming and print state updates."""
    from agent.graph import define_graph

    app = define_graph()

    inputs = {
        "input": query,
        "chat_history": [],
        "plan": [],
        "current_step": 0,
        "findings": {},
        "messages": [],
        "executor_call_count": 0,
        "iteration_count": 0,
    }

    print(f"\n{'='*70}")
    print(f"  Query: {query}")
    print(f"{'='*70}\n")

    total_start = time.time()

    for event in app.stream(inputs, config={"recursion_limit": 150}, stream_mode="updates"):
        for node_name, state_update in event.items():
            node_start = time.time()

            print(f"--- {node_name} ---")

            # Show relevant state changes per node type
            if node_name == "planner":
                plan = state_update.get("plan", [])
                iteration = state_update.get("iteration_count", 0)
                print(f"  Iteration: {iteration}")
                print(f"  Plan ({len(plan)} steps):")
                for i, step in enumerate(plan):
                    print(f"    {i+1}. {step}")

            elif node_name == "setup_executor":
                count = state_update.get("executor_call_count", "?")
                msgs = state_update.get("messages", [])
                new_msgs = [m for m in msgs if not hasattr(m, "id") or not str(type(m).__name__).startswith("Remove")]
                print(f"  executor_call_count reset to: {count}")
                print(f"  Messages initialized: {len(new_msgs)} new")

            elif node_name == "executor_llm":
                msgs = state_update.get("messages", [])
                call_count = state_update.get("executor_call_count", "?")
                print(f"  executor_call_count: {call_count}")
                for m in msgs:
                    if hasattr(m, "tool_calls") and m.tool_calls:
                        for tc in m.tool_calls:
                            print(f"  Tool call: {tc['name']}({format_value(tc.get('args', {}), 80)})")
                    elif hasattr(m, "content") and m.content:
                        print(f"  AI response: {format_value(m.content, 150)}")

            elif node_name == "tool_node":
                msgs = state_update.get("messages", [])
                for m in msgs:
                    if hasattr(m, "content"):
                        print(f"  Tool result: {format_value(m.content, 150)}")

            elif node_name == "aggregate":
                findings = state_update.get("findings", {})
                step = state_update.get("current_step", "?")
                print(f"  current_step advanced to: {step}")
                print(f"  Findings keys: {list(findings.keys())}")

            elif node_name == "refinery":
                decision = state_update.get("loop_decision", "?")
                print(f"  Decision: {decision}")

            elif node_name == "synthesizer":
                response = state_update.get("response", "")
                print(f"  Response preview: {format_value(response, 300)}")

            if verbose:
                print(f"  [Full state update]: {json.dumps({k: format_value(v) for k, v in state_update.items() if k != 'messages'}, indent=2)}")

            elapsed = time.time() - node_start
            print(f"  ({elapsed:.2f}s)")
            print()

    total_elapsed = time.time() - total_start
    print(f"{'='*70}")
    print(f"  Total execution time: {total_elapsed:.2f}s")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Code RAG Agent execution in real-time"
    )
    parser.add_argument(
        "query",
        help="The query to send to the agent",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full state dumps at each node",
    )

    args = parser.parse_args()
    monitor(args.query, verbose=args.verbose)


if __name__ == "__main__":
    main()
