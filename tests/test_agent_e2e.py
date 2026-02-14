"""End-to-end agent tests — mocked full cycle + optional live LLM."""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import define_graph
from agent.state import empty_working_memory
from agent.prompts import Task, PlanOutput, RefineDecision


def _make_structured_output_side_effect(outputs_by_schema):
    """Create a with_structured_output mock that returns different runnables per schema.

    Args:
        outputs_by_schema: dict mapping schema class -> output object(s).
            If the value is a list, successive calls return successive items.
    """
    # Track call index per schema
    call_idx = {}

    def with_structured_output(schema, **kwargs):
        mock_runnable = MagicMock()
        key = schema.__name__ if hasattr(schema, "__name__") else str(schema)

        if key not in call_idx:
            call_idx[key] = 0

        val = outputs_by_schema.get(schema)
        if isinstance(val, list):
            # Return successive items on each invoke
            def invoke_side_effect(*args, **kw):
                i = call_idx[key]
                call_idx[key] = i + 1
                return val[i] if i < len(val) else val[-1]

            mock_runnable.invoke.side_effect = invoke_side_effect
            mock_runnable.side_effect = invoke_side_effect
        else:
            mock_runnable.invoke.return_value = val
            mock_runnable.return_value = val

        return mock_runnable

    return with_structured_output


def _make_base_model_mock(structured_outputs, synth_response):
    """Create a mock for get_model() that handles both structured output and LCEL chains.

    - with_structured_output() -> dispatches by schema class
    - Direct __call__ / invoke -> returns synth_response (for synthesize_node's LCEL chain)
    """
    mock = MagicMock()
    mock.with_structured_output.side_effect = _make_structured_output_side_effect(
        structured_outputs
    )
    # For synthesize_node: SYNTHESIZE_PROMPT | model | StrOutputParser()
    mock.return_value = synth_response
    mock.invoke.return_value = synth_response
    return mock


def _make_inputs(user_input="What is the main function?"):
    """Create default graph inputs with new state shape."""
    return {
        "input": user_input,
        "plan": [],
        "current_step": 0,
        "working_memory": empty_working_memory(),
        "messages": [],
        "executor_call_count": 0,
        "iteration_count": 0,
    }


# ---------------------------------------------------------------------------
# Part A: Mocked E2E — full graph cycle with mock LLM
# ---------------------------------------------------------------------------

class TestMockedE2E:
    """Full graph cycle with both get_model and get_model_with_tools mocked."""

    @patch("agent.nodes._save_plan_log")
    @patch("agent.nodes.get_model_with_tools")
    @patch("agent.nodes.get_model")
    def test_single_task_cycle(self, mock_get_model, mock_get_mwt, _mock_save):
        """Run planner (1 task) -> executor (no tool calls) -> aggregate -> refinery -> synthesize."""

        mock_get_model.return_value = _make_base_model_mock(
            structured_outputs={
                PlanOutput: PlanOutput(tasks=[
                    Task(
                        goal="Search for main function",
                        success_criteria="Found the entry point",
                        abort_criteria="No results after 2 attempts",
                    ),
                ]),
            },
            synth_response=AIMessage(
                content="The main function is the entry point of the program."
            ),
        )

        # Tools model: executor_llm (no tool calls -> goes to aggregate)
        executor_mock = MagicMock()
        executor_mock.invoke.return_value = AIMessage(
            content="I found that the main function is in cli.py"
        )
        mock_get_mwt.return_value = executor_mock

        app = define_graph()
        final_state = app.invoke(_make_inputs(), config={"recursion_limit": 50})

        assert "response" in final_state
        assert len(final_state["response"]) > 0
        assert final_state["iteration_count"] >= 1
        # Working memory should have at least one task result
        wm = final_state["working_memory"]
        assert len(wm["task_results"]) >= 1

    @patch("agent.nodes._save_plan_log")
    @patch("agent.nodes.get_model_with_tools")
    @patch("agent.nodes.get_model")
    def test_multi_task_sequential_execution(self, mock_get_model, mock_get_mwt, _mock_save):
        """Verify planner generates 2 tasks and both are executed sequentially."""

        mock_get_model.return_value = _make_base_model_mock(
            structured_outputs={
                PlanOutput: PlanOutput(tasks=[
                    Task(
                        goal="Search for SearchEngine",
                        success_criteria="Found the class",
                        abort_criteria="Not found",
                    ),
                    Task(
                        goal="Read the search_engine.py file",
                        success_criteria="Understood hybrid search",
                        abort_criteria="File too large",
                    ),
                ]),
            },
            synth_response=AIMessage(content="SearchEngine uses hybrid search."),
        )

        # Executor returns different responses on successive calls
        executor_mock = MagicMock()
        call_count = {"n": 0}
        def executor_invoke(messages):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return AIMessage(content="Found SearchEngine in retrieval/search_engine.py")
            else:
                return AIMessage(content="SearchEngine combines vector and BM25 scores")
        executor_mock.invoke.side_effect = executor_invoke
        mock_get_mwt.return_value = executor_mock

        app = define_graph()
        final_state = app.invoke(
            _make_inputs("How does SearchEngine work?"),
            config={"recursion_limit": 50},
        )

        # Both tasks should have been executed
        wm = final_state["working_memory"]
        assert len(wm["task_results"]) == 2
        assert wm["task_results"][0]["task"] == "Search for SearchEngine"
        assert wm["task_results"][1]["task"] == "Read the search_engine.py file"
        assert len(final_state["response"]) > 0

    @patch("agent.nodes._save_plan_log")
    @patch("agent.nodes.get_model_with_tools")
    @patch("agent.nodes.get_model")
    def test_no_outer_loop(self, mock_get_model, mock_get_mwt, _mock_save):
        """Verify that the graph does NOT loop back to planner after refinery."""

        mock_get_model.return_value = _make_base_model_mock(
            structured_outputs={
                PlanOutput: PlanOutput(tasks=[
                    Task(
                        goal="Search for something",
                        success_criteria="Found it",
                        abort_criteria="Not found",
                    ),
                ]),
            },
            synth_response=AIMessage(content="Answer."),
        )

        executor_mock = MagicMock()
        executor_mock.invoke.return_value = AIMessage(content="Found it")
        mock_get_mwt.return_value = executor_mock

        app = define_graph()
        final_state = app.invoke(_make_inputs(), config={"recursion_limit": 50})

        # iteration_count should be exactly 1 (planner called once, no re-planning)
        assert final_state["iteration_count"] == 1


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
        original_max_exec = config.max_executor_steps
        try:
            config.max_executor_steps = 2

            app = define_graph()
            inputs = _make_inputs("What is the main function?")

            final_state = app.invoke(inputs, config={"recursion_limit": 50})

            assert "response" in final_state
            assert len(final_state["response"]) > 0
            assert final_state["plan"]
            assert final_state["iteration_count"] >= 1
            print(f"\n  Response preview: {final_state['response'][:200]}...")

        finally:
            config.max_executor_steps = original_max_exec
