"""End-to-end agent tests — mocked full cycle + optional live LLM."""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import define_graph


def _make_model_mock(responses):
    """Create a mock model that returns different AIMessages on successive calls.

    Works with both LCEL chain __call__ and direct .invoke() patterns.
    """
    mock = MagicMock()
    idx = {"n": 0}

    def side_effect(*args, **kwargs):
        i = idx["n"]
        idx["n"] += 1
        if i < len(responses):
            return responses[i]
        return responses[-1]  # repeat last response if called extra times

    mock.side_effect = side_effect
    mock.invoke.side_effect = side_effect
    return mock


# ---------------------------------------------------------------------------
# Part A: Mocked E2E — full graph cycle with mock LLM
# ---------------------------------------------------------------------------

class TestMockedE2E:
    """Full graph cycle with both get_model and get_model_with_tools mocked."""

    @patch("agent.nodes.get_model_with_tools")
    @patch("agent.nodes.get_model")
    def test_full_graph_cycle(self, mock_get_model, mock_get_mwt):
        """Run planner → executor (no tool calls) → refine (FINISH) → synthesize."""

        # Base model: planner → refinery → synthesizer
        mock_get_model.return_value = _make_model_mock([
            # Planner
            AIMessage(content='["Search for main function"]'),
            # Refinery
            AIMessage(content='{"decision": "FINISH", "reason": "Sufficient info"}'),
            # Synthesizer
            AIMessage(content="The main function is the entry point of the program."),
        ])

        # Tools model: executor_llm (no tool calls → goes to aggregate)
        executor_mock = MagicMock()
        executor_mock.invoke.return_value = AIMessage(
            content="I found that the main function is in cli.py"
        )
        mock_get_mwt.return_value = executor_mock

        app = define_graph()
        inputs = {
            "input": "What is the main function?",
            "chat_history": [],
            "plan": [],
            "current_step": 0,
            "findings": {},
            "messages": [],
            "executor_call_count": 0,
            "iteration_count": 0,
        }

        final_state = app.invoke(inputs, config={"recursion_limit": 50})

        assert "response" in final_state
        assert len(final_state["response"]) > 0
        assert final_state["iteration_count"] >= 1
        assert len(final_state["findings"]) >= 1

    @patch("agent.nodes.get_model_with_tools")
    @patch("agent.nodes.get_model")
    def test_continue_then_finish(self, mock_get_model, mock_get_mwt):
        """Verify CONTINUE loops back to planner before finishing."""

        mock_get_model.return_value = _make_model_mock([
            # Planner iteration 1
            AIMessage(content='["Search for SearchEngine"]'),
            # Refinery iteration 1: CONTINUE
            AIMessage(content='{"decision": "CONTINUE", "reason": "Need more"}'),
            # Planner iteration 2
            AIMessage(content='["Read the search_engine.py file"]'),
            # Refinery iteration 2: FINISH
            AIMessage(content='{"decision": "FINISH", "reason": "Complete"}'),
            # Synthesizer
            AIMessage(content="SearchEngine uses hybrid search."),
        ])

        executor_mock = MagicMock()
        executor_mock.invoke.return_value = AIMessage(
            content="Found SearchEngine in retrieval/search_engine.py"
        )
        mock_get_mwt.return_value = executor_mock

        app = define_graph()
        inputs = {
            "input": "How does SearchEngine work?",
            "chat_history": [],
            "plan": [],
            "current_step": 0,
            "findings": {},
            "messages": [],
            "executor_call_count": 0,
            "iteration_count": 0,
        }

        final_state = app.invoke(inputs, config={"recursion_limit": 50})

        assert final_state["iteration_count"] >= 2
        assert len(final_state["findings"]) >= 2
        assert len(final_state["response"]) > 0


# ---------------------------------------------------------------------------
# Part B: Live LLM E2E — real API calls with cost protection
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestLiveE2E:
    """Real LLM tests — skipped by default, run with: pytest -m live -s"""

    def test_live_agent_query(self):
        """Run a real agent query with minimal iterations.

        Requires API keys and user confirmation.
        Run with: pytest tests/test_agent_e2e.py -m live -s -v
        """
        import sys
        from config import config

        # User confirmation gate
        print(
            f"\n{'='*60}\n"
            f"  This test will make real LLM API calls (~2-3 calls).\n"
            f"  Provider: {config.chat_provider}, Model: {config.chat_model}\n"
            f"{'='*60}"
        )

        if not sys.stdin.isatty():
            pytest.skip("Non-interactive session — skipping live test")

        answer = input("  Continue? [y/N]: ").strip().lower()
        if answer != "y":
            pytest.skip("User declined live test")

        # Minimize API calls
        original_max_iter = config.max_iterations
        original_max_exec = config.max_executor_steps
        try:
            config.max_iterations = 1
            config.max_executor_steps = 2

            app = define_graph()
            inputs = {
                "input": "What is the main function?",
                "chat_history": [],
                "plan": [],
                "current_step": 0,
                "findings": {},
                "messages": [],
                "executor_call_count": 0,
                "iteration_count": 0,
            }

            final_state = app.invoke(inputs, config={"recursion_limit": 50})

            assert "response" in final_state
            assert len(final_state["response"]) > 0
            assert final_state["plan"]
            assert final_state["iteration_count"] >= 1
            print(f"\n  Response preview: {final_state['response'][:200]}...")

        finally:
            config.max_iterations = original_max_iter
            config.max_executor_steps = original_max_exec
