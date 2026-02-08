"""Unit tests for agent nodes — uses mocked LLM, no real API calls."""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from agent.state import AgentState
from agent.nodes import (
    plan_node,
    setup_executor,
    executor_llm_node,
    route_executor,
    aggregate_node,
    refine_node,
    synthesize_node,
)
from agent.prompts import PlanOutput, RefineDecision


def _make_state(**overrides) -> dict:
    """Create a minimal AgentState dict with sensible defaults."""
    base = {
        "input": "How does the search engine work?",
        "chat_history": [],
        "plan": [],
        "current_step": 0,
        "findings": {},
        "response": "",
        "loop_decision": "CONTINUE",
        "messages": [],
        "executor_call_count": 0,
        "iteration_count": 0,
    }
    base.update(overrides)
    return base


def _mock_model_returning(ai_message):
    """Create a MagicMock that works as both callable and invoke-able."""
    mock = MagicMock()
    mock.return_value = ai_message
    mock.invoke.return_value = ai_message
    return mock


def _mock_model_with_structured_output(output_obj):
    """Create a mock model for with_structured_output() chains.

    The chain is: prompt | model.with_structured_output(Schema)
    So model.with_structured_output() must return a runnable that,
    when called or invoked, returns the output_obj.
    """
    mock_model = MagicMock()
    mock_runnable = MagicMock()
    mock_runnable.return_value = output_obj
    mock_runnable.invoke.return_value = output_obj
    mock_model.with_structured_output.return_value = mock_runnable
    return mock_model


# ---------------------------------------------------------------------------
# plan_node
# ---------------------------------------------------------------------------

class TestPlanNode:

    @patch("agent.nodes.get_model")
    def test_normal_plan_generation(self, mock_get_model):
        """plan_node should return a list of steps from LLM output."""
        plan_output = PlanOutput(steps=["Search for SearchEngine class", "Read the file"])
        mock_get_model.return_value = _mock_model_with_structured_output(plan_output)

        state = _make_state()
        result = plan_node(state)

        assert "plan" in result
        assert isinstance(result["plan"], list)
        assert len(result["plan"]) == 2
        assert result["current_step"] == 0
        assert result["iteration_count"] == 1

    @patch("agent.nodes.get_model")
    def test_fallback_plan_on_error(self, mock_get_model):
        """If LLM fails, plan_node should return fallback plan."""
        mock_model = MagicMock()
        mock_runnable = MagicMock()
        mock_runnable.invoke.side_effect = Exception("API error")
        mock_runnable.side_effect = Exception("API error")
        mock_model.with_structured_output.return_value = mock_runnable
        mock_get_model.return_value = mock_model

        state = _make_state()
        result = plan_node(state)

        assert result["plan"] == ["Search for relevant code related to the query"]
        assert result["current_step"] == 0

    @patch("agent.nodes.get_model")
    def test_iteration_count_increments(self, mock_get_model):
        """iteration_count should increment on each call."""
        plan_output = PlanOutput(steps=["step1"])
        mock_get_model.return_value = _mock_model_with_structured_output(plan_output)

        state = _make_state(iteration_count=2)
        result = plan_node(state)
        assert result["iteration_count"] == 3


# ---------------------------------------------------------------------------
# setup_executor
# ---------------------------------------------------------------------------

class TestSetupExecutor:

    def test_message_initialization(self):
        """setup_executor should create SystemMessage + HumanMessage."""
        state = _make_state(plan=["Search for code", "Read file"])
        result = setup_executor(state)

        new_msgs = [m for m in result["messages"] if not isinstance(m, RemoveMessage)]
        assert len(new_msgs) == 2
        assert isinstance(new_msgs[0], SystemMessage)
        assert isinstance(new_msgs[1], HumanMessage)

    def test_executor_call_count_reset(self):
        """setup_executor should reset executor_call_count to 0."""
        state = _make_state(executor_call_count=5, plan=["step1"])
        result = setup_executor(state)
        assert result["executor_call_count"] == 0

    def test_remove_messages_for_old_messages(self):
        """setup_executor should emit RemoveMessage for each existing message."""
        old_msgs = [
            HumanMessage(content="old1", id="msg1"),
            AIMessage(content="old2", id="msg2"),
        ]
        state = _make_state(messages=old_msgs, plan=["step1"])
        result = setup_executor(state)

        remove_msgs = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) == 2

    def test_no_plan_fallback(self):
        """setup_executor should handle empty plan gracefully."""
        state = _make_state(plan=[], current_step=0)
        result = setup_executor(state)
        new_msgs = [m for m in result["messages"] if not isinstance(m, RemoveMessage)]
        assert len(new_msgs) == 2
        assert "No more steps" in new_msgs[0].content


# ---------------------------------------------------------------------------
# executor_llm_node
# ---------------------------------------------------------------------------

class TestExecutorLLMNode:

    @patch("agent.nodes.get_model_with_tools")
    def test_returns_ai_message_with_tool_calls(self, mock_get_mwt):
        """executor_llm_node should pass through LLM response with tool_calls."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "search_codebase", "args": {"query": "search"}, "id": "tc1"}],
        )
        mock_model = MagicMock()
        mock_model.invoke.return_value = ai_msg
        mock_get_mwt.return_value = mock_model

        state = _make_state(messages=[HumanMessage(content="do something")])
        result = executor_llm_node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls

    @patch("agent.nodes.get_model_with_tools")
    def test_returns_ai_message_without_tool_calls(self, mock_get_mwt):
        """executor_llm_node should handle responses without tool_calls."""
        ai_msg = AIMessage(content="Here is what I found")
        mock_model = MagicMock()
        mock_model.invoke.return_value = ai_msg
        mock_get_mwt.return_value = mock_model

        state = _make_state(messages=[HumanMessage(content="do something")])
        result = executor_llm_node(state)

        assert len(result["messages"]) == 1
        assert not result["messages"][0].tool_calls

    @patch("agent.nodes.get_model_with_tools")
    def test_call_count_increments(self, mock_get_mwt):
        """executor_call_count should increment by 1."""
        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(content="done")
        mock_get_mwt.return_value = mock_model

        state = _make_state(executor_call_count=2)
        result = executor_llm_node(state)
        assert result["executor_call_count"] == 3


# ---------------------------------------------------------------------------
# route_executor (pure state logic — no mock needed)
# ---------------------------------------------------------------------------

class TestRouteExecutor:

    def test_routes_to_tools_when_tool_calls_exist(self):
        """Should return 'tools' if last message has tool_calls and count < max."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "search_codebase", "args": {"query": "x"}, "id": "tc1"}],
        )
        state = _make_state(messages=[ai_msg], executor_call_count=1)
        assert route_executor(state) == "tools"

    def test_routes_to_aggregate_when_no_tool_calls(self):
        """Should return 'aggregate' if last message has no tool_calls."""
        ai_msg = AIMessage(content="summary of findings")
        state = _make_state(messages=[ai_msg], executor_call_count=1)
        assert route_executor(state) == "aggregate"

    @patch("agent.nodes.config")
    def test_routes_to_aggregate_when_max_steps_reached(self, mock_config):
        """Should return 'aggregate' if executor_call_count >= max_executor_steps."""
        mock_config.max_executor_steps = 3
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "search_codebase", "args": {"query": "x"}, "id": "tc1"}],
        )
        state = _make_state(messages=[ai_msg], executor_call_count=3)
        assert route_executor(state) == "aggregate"

    def test_routes_to_aggregate_when_empty_messages(self):
        """Should return 'aggregate' if messages list is empty."""
        state = _make_state(messages=[])
        assert route_executor(state) == "aggregate"


# ---------------------------------------------------------------------------
# aggregate_node
# ---------------------------------------------------------------------------

class TestAggregateNode:

    def test_collect_findings_from_tool_messages(self):
        """aggregate_node should collect content from ToolMessages."""
        msgs = [
            SystemMessage(content="context", id="sys1"),
            HumanMessage(content="task", id="h1"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_codebase", "args": {"query": "x"}, "id": "tc1"}],
                id="ai1",
            ),
            ToolMessage(content="Found class Foo in file.py", tool_call_id="tc1", id="tm1"),
            AIMessage(content="Based on results, Foo is a utility class", id="ai2"),
        ]
        state = _make_state(
            messages=msgs,
            plan=["Search for Foo", "Read file"],
            current_step=0,
        )
        result = aggregate_node(state)

        assert "findings" in result
        assert len(result["findings"]) == 1
        key = list(result["findings"].keys())[0]
        assert "step_0" in key
        assert "Found class Foo" in result["findings"][key]
        assert result["current_step"] == 1

    def test_increment_current_step(self):
        """aggregate_node should advance current_step by 1."""
        state = _make_state(
            messages=[AIMessage(content="done", id="a1")],
            plan=["s1", "s2", "s3"],
            current_step=1,
        )
        result = aggregate_node(state)
        assert result["current_step"] == 2

    def test_clear_messages(self):
        """aggregate_node should emit RemoveMessage for all messages."""
        msgs = [
            HumanMessage(content="task", id="h1"),
            AIMessage(content="result", id="a1"),
        ]
        state = _make_state(messages=msgs, plan=["step1"])
        result = aggregate_node(state)

        remove_msgs = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) == 2


# ---------------------------------------------------------------------------
# refine_node
# ---------------------------------------------------------------------------

class TestRefineNode:

    @patch("agent.nodes.get_model")
    def test_continue_decision(self, mock_get_model):
        """refine_node should return CONTINUE when LLM decides so."""
        refine_output = RefineDecision(decision="CONTINUE", reason="Need more info")
        mock_get_model.return_value = _mock_model_with_structured_output(refine_output)

        state = _make_state(
            findings={"step_0: search": "some results"},
            plan=["s1", "s2"],
            current_step=1,
            iteration_count=1,
        )
        result = refine_node(state)
        assert result["loop_decision"] == "CONTINUE"

    @patch("agent.nodes.get_model")
    def test_finish_decision(self, mock_get_model):
        """refine_node should return FINISH when LLM decides so."""
        refine_output = RefineDecision(decision="FINISH", reason="Sufficient")
        mock_get_model.return_value = _mock_model_with_structured_output(refine_output)

        state = _make_state(
            findings={"step_0: search": "complete results"},
            plan=["s1"],
            current_step=1,
            iteration_count=1,
        )
        result = refine_node(state)
        assert result["loop_decision"] == "FINISH"

    def test_forced_finish_at_max_iterations(self):
        """refine_node should force FINISH when max_iterations reached."""
        from config import config

        original = config.max_iterations
        try:
            config.max_iterations = 3
            state = _make_state(iteration_count=3)
            result = refine_node(state)
            assert result["loop_decision"] == "FINISH"
        finally:
            config.max_iterations = original


# ---------------------------------------------------------------------------
# synthesize_node
# ---------------------------------------------------------------------------

class TestSynthesizeNode:

    @patch("agent.nodes.get_model")
    def test_generates_response(self, mock_get_model):
        """synthesize_node should produce a response string."""
        ai_msg = AIMessage(
            content="The search engine uses hybrid search combining vector and BM25."
        )
        mock_get_model.return_value = _mock_model_returning(ai_msg)

        state = _make_state(
            findings={"step_0: search": "Vector + BM25 hybrid search"},
        )
        result = synthesize_node(state)

        assert "response" in result
        assert len(result["response"]) > 0
        assert "hybrid search" in result["response"]
