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

from agent.state import AgentState, empty_working_memory
from agent.nodes import (
    plan_node,
    setup_executor,
    executor_llm_node,
    route_executor,
    aggregate_node,
    refine_node,
    synthesize_node,
)
from agent.prompts import Task, PlanOutput, RefineDecision


def _make_task(goal="Search for code", **overrides) -> dict:
    """Create a minimal task dict with sensible defaults."""
    base = {
        "goal": goal,
        "success_criteria": "Found relevant code",
        "abort_criteria": "No results after 2 attempts",
    }
    base.update(overrides)
    return base


def _make_state(**overrides) -> dict:
    """Create a minimal AgentState dict with sensible defaults."""
    base = {
        "input": "How does the search engine work?",
        "plan": [],
        "current_step": 0,
        "working_memory": empty_working_memory(),
        "response": "",
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

    @patch("agent.nodes._save_plan_log")
    @patch("agent.nodes.get_codebase_context", return_value="")
    @patch("agent.nodes.get_model")
    def test_normal_plan_generation(self, mock_get_model, _mock_ctx, _mock_save):
        """plan_node should return a list of task dicts from LLM output."""
        plan_output = PlanOutput(tasks=[
            Task(
                goal="Search for SearchEngine class",
                success_criteria="Found the class definition",
                abort_criteria="No results after 2 attempts",
                suggested_tools=["search"],
            ),
            Task(
                goal="Read the SearchEngine implementation",
                success_criteria="Understood the hybrid search logic",
                abort_criteria="File too large to parse",
            ),
        ])
        mock_get_model.return_value = _mock_model_with_structured_output(plan_output)

        state = _make_state()
        result = plan_node(state)

        assert "plan" in result
        assert isinstance(result["plan"], list)
        assert len(result["plan"]) == 2
        assert result["plan"][0]["goal"] == "Search for SearchEngine class"
        assert result["plan"][0]["success_criteria"] == "Found the class definition"
        assert "suggested_tools" in result["plan"][0]
        assert result["plan"][1]["goal"] == "Read the SearchEngine implementation"
        assert result["current_step"] == 0
        assert result["iteration_count"] == 1

    @patch("agent.nodes._save_plan_log")
    @patch("agent.nodes.get_codebase_context", return_value="")
    @patch("agent.nodes.get_model")
    def test_fallback_plan_on_error(self, mock_get_model, _mock_ctx, _mock_save):
        """If LLM fails, plan_node should return fallback task."""
        mock_model = MagicMock()
        mock_runnable = MagicMock()
        mock_runnable.invoke.side_effect = Exception("API error")
        mock_runnable.side_effect = Exception("API error")
        mock_model.with_structured_output.return_value = mock_runnable
        mock_get_model.return_value = mock_model

        state = _make_state()
        result = plan_node(state)

        assert len(result["plan"]) == 1
        assert result["plan"][0]["goal"] == "Search for relevant code related to the query"
        assert "success_criteria" in result["plan"][0]
        assert "abort_criteria" in result["plan"][0]
        assert result["current_step"] == 0

    @patch("agent.nodes._save_plan_log")
    @patch("agent.nodes.get_codebase_context", return_value="")
    @patch("agent.nodes.get_model")
    def test_iteration_count_increments(self, mock_get_model, _mock_ctx, _mock_save):
        """iteration_count should increment on each call."""
        plan_output = PlanOutput(tasks=[
            Task(
                goal="step1",
                success_criteria="done",
                abort_criteria="fail",
            ),
        ])
        mock_get_model.return_value = _mock_model_with_structured_output(plan_output)

        state = _make_state(iteration_count=2)
        result = plan_node(state)
        assert result["iteration_count"] == 3

    @patch("agent.nodes.get_codebase_context", return_value="some profile")
    @patch("agent.nodes.get_model")
    def test_plan_log_is_saved(self, mock_get_model, _mock_ctx, tmp_path):
        """plan_node should persist the plan to a JSON log file."""
        plan_output = PlanOutput(tasks=[
            Task(goal="find entry point", success_criteria="found", abort_criteria="not found"),
        ])
        mock_get_model.return_value = _mock_model_with_structured_output(plan_output)

        with patch("agent.nodes.PLAN_LOG_DIR", tmp_path):
            state = _make_state()
            plan_node(state)

        log_files = list(tmp_path.glob("plan_*.json"))
        assert len(log_files) == 1

        import json
        with open(log_files[0]) as f:
            log_data = json.load(f)
        assert log_data["query"] == "How does the search engine work?"
        assert len(log_data["tasks"]) == 1
        assert log_data["tasks"][0]["goal"] == "find entry point"
        assert log_data["iteration"] == 0


# ---------------------------------------------------------------------------
# setup_executor
# ---------------------------------------------------------------------------

class TestSetupExecutor:

    def test_message_initialization(self):
        """setup_executor should create SystemMessage + HumanMessage with task fields."""
        state = _make_state(plan=[
            _make_task("Search for relevant code"),
            _make_task("Read the file"),
        ])
        result = setup_executor(state)

        new_msgs = [m for m in result["messages"] if not isinstance(m, RemoveMessage)]
        assert len(new_msgs) == 2
        assert isinstance(new_msgs[0], SystemMessage)
        assert isinstance(new_msgs[1], HumanMessage)
        # Task goal should appear in the messages
        assert "Search for relevant code" in new_msgs[0].content

    def test_task_fields_in_system_message(self):
        """setup_executor should include success/abort criteria in the system message."""
        state = _make_state(plan=[
            _make_task(
                goal="Find swap functions",
                success_criteria="Located the swap entry point",
                abort_criteria="3 failed searches",
                suggested_tools=["search", "related_code"],
                context_hint="Check mm/swap.c",
            ),
        ])
        result = setup_executor(state)

        sys_msg = [m for m in result["messages"] if isinstance(m, SystemMessage)][0]
        assert "Find swap functions" in sys_msg.content
        assert "Located the swap entry point" in sys_msg.content
        assert "3 failed searches" in sys_msg.content
        assert "search, related_code" in sys_msg.content
        assert "Check mm/swap.c" in sys_msg.content

    def test_executor_call_count_reset(self):
        """setup_executor should reset executor_call_count to 0."""
        state = _make_state(executor_call_count=5, plan=[_make_task()])
        result = setup_executor(state)
        assert result["executor_call_count"] == 0

    def test_remove_messages_for_old_messages(self):
        """setup_executor should emit RemoveMessage for each existing message."""
        old_msgs = [
            HumanMessage(content="old1", id="msg1"),
            AIMessage(content="old2", id="msg2"),
        ]
        state = _make_state(messages=old_msgs, plan=[_make_task()])
        result = setup_executor(state)

        remove_msgs = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) == 2

    def test_no_plan_fallback(self):
        """setup_executor should handle empty plan gracefully."""
        state = _make_state(plan=[], current_step=0)
        result = setup_executor(state)
        new_msgs = [m for m in result["messages"] if not isinstance(m, RemoveMessage)]
        assert len(new_msgs) == 2
        assert "No more tasks" in new_msgs[0].content

    def test_working_memory_in_system_message(self):
        """setup_executor should include working memory summary in system message."""
        wm = empty_working_memory()
        wm["discovered_entities"] = [{"name": "SearchEngine", "type": "class", "location": "search.py"}]
        wm["insights"] = ["SearchEngine uses hybrid search"]
        state = _make_state(
            plan=[_make_task("Read search.py")],
            working_memory=wm,
        )
        result = setup_executor(state)

        sys_msg = [m for m in result["messages"] if isinstance(m, SystemMessage)][0]
        assert "SearchEngine" in sys_msg.content


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
        """aggregate_node should build working_memory from ToolMessage artifacts."""
        msgs = [
            SystemMessage(content="context", id="sys1"),
            HumanMessage(content="task", id="h1"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_codebase", "args": {"query": "x"}, "id": "tc1"}],
                id="ai1",
            ),
            ToolMessage(
                content="Found class Foo in file.py",
                tool_call_id="tc1",
                id="tm1",
                artifact={
                    "entities": [{"name": "file.py", "type": "file", "location": "file.py"}],
                    "relationships": [],
                },
            ),
            AIMessage(content="Based on results, Foo is a utility class", id="ai2"),
        ]
        state = _make_state(
            messages=msgs,
            plan=[
                _make_task("Search for Foo"),
                _make_task("Read file"),
            ],
            current_step=0,
        )
        result = aggregate_node(state)

        wm = result["working_memory"]
        assert len(wm["discovered_entities"]) == 1
        assert wm["discovered_entities"][0]["name"] == "file.py"
        assert len(wm["task_results"]) == 1
        assert wm["task_results"][0]["task"] == "Search for Foo"
        assert "Found class Foo" in wm["task_results"][0]["summary"]
        assert result["current_step"] == 1
        # AI summary should become an insight
        assert len(wm["insights"]) == 1
        assert "Foo is a utility class" in wm["insights"][0]

    def test_increment_current_step(self):
        """aggregate_node should advance current_step by 1."""
        state = _make_state(
            messages=[AIMessage(content="done", id="a1")],
            plan=[_make_task("s1"), _make_task("s2"), _make_task("s3")],
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
        state = _make_state(messages=msgs, plan=[_make_task()])
        result = aggregate_node(state)

        remove_msgs = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) == 2

    def test_dedup_entities(self):
        """aggregate_node should deduplicate entities by name across tool calls."""
        msgs = [
            ToolMessage(
                content="Found Foo",
                tool_call_id="tc1",
                id="tm1",
                artifact={
                    "entities": [{"name": "Foo", "type": "class", "location": "a.py"}],
                    "relationships": [],
                },
            ),
            ToolMessage(
                content="Found Foo again",
                tool_call_id="tc2",
                id="tm2",
                artifact={
                    "entities": [{"name": "Foo", "type": "class", "location": "a.py"}],
                    "relationships": [],
                },
            ),
        ]
        state = _make_state(messages=msgs, plan=[_make_task()])
        result = aggregate_node(state)
        assert len(result["working_memory"]["discovered_entities"]) == 1

    def test_merge_relationships(self):
        """aggregate_node should merge relationships from multiple tool calls."""
        msgs = [
            ToolMessage(
                content="callers of Foo",
                tool_call_id="tc1",
                id="tm1",
                artifact={
                    "entities": [{"name": "Bar", "type": "function", "location": "b.py"}],
                    "relationships": [{"source": "Bar", "type": "calls", "target": "Foo"}],
                },
            ),
            ToolMessage(
                content="callees of Foo",
                tool_call_id="tc2",
                id="tm2",
                artifact={
                    "entities": [{"name": "Baz", "type": "function", "location": "c.py"}],
                    "relationships": [{"source": "Foo", "type": "calls", "target": "Baz"}],
                },
            ),
        ]
        state = _make_state(messages=msgs, plan=[_make_task()])
        result = aggregate_node(state)
        wm = result["working_memory"]
        assert len(wm["discovered_entities"]) == 2
        assert len(wm["relationships"]) == 2

    def test_accumulate_across_tasks(self):
        """aggregate_node should preserve entities from previous tasks."""
        existing_wm = empty_working_memory()
        existing_wm["discovered_entities"] = [{"name": "OldEntity", "type": "class", "location": "old.py"}]
        existing_wm["task_results"] = [{"task": "prev task", "step_index": 0, "summary": "found old"}]

        msgs = [
            ToolMessage(
                content="Found NewEntity",
                tool_call_id="tc1",
                id="tm1",
                artifact={
                    "entities": [{"name": "NewEntity", "type": "function", "location": "new.py"}],
                    "relationships": [],
                },
            ),
        ]
        state = _make_state(
            messages=msgs,
            plan=[_make_task("prev"), _make_task("current")],
            current_step=1,
            working_memory=existing_wm,
        )
        result = aggregate_node(state)
        wm = result["working_memory"]
        assert len(wm["discovered_entities"]) == 2
        assert len(wm["task_results"]) == 2

    def test_no_artifact_graceful(self):
        """aggregate_node should handle ToolMessages without artifacts."""
        msgs = [
            ToolMessage(content="Some text result", tool_call_id="tc1", id="tm1"),
            AIMessage(content="Summary", id="ai1"),
        ]
        state = _make_state(messages=msgs, plan=[_make_task()])
        result = aggregate_node(state)
        wm = result["working_memory"]
        assert len(wm["task_results"]) == 1
        assert "Some text result" in wm["task_results"][0]["summary"]


# ---------------------------------------------------------------------------
# refine_node
# ---------------------------------------------------------------------------

class TestRefineNode:

    def test_returns_empty_dict(self):
        """refine_node should return an empty dict (logging-only shell)."""
        state = _make_state(
            working_memory=empty_working_memory(),
            plan=[_make_task("s1"), _make_task("s2")],
            current_step=2,
        )
        result = refine_node(state)
        assert result == {}

    def test_no_llm_calls(self):
        """refine_node should not call the LLM."""
        state = _make_state()
        with patch("agent.nodes.get_model") as mock_get_model:
            refine_node(state)
            mock_get_model.assert_not_called()


# ---------------------------------------------------------------------------
# synthesize_node
# ---------------------------------------------------------------------------

class TestSynthesizeNode:

    @patch("agent.nodes.get_codebase_context", return_value="")
    @patch("agent.nodes.get_model")
    def test_generates_response(self, mock_get_model, _mock_ctx):
        """synthesize_node should produce a response string."""
        ai_msg = AIMessage(
            content="The search engine uses hybrid search combining vector and BM25."
        )
        mock_get_model.return_value = _mock_model_returning(ai_msg)

        wm = empty_working_memory()
        wm["task_results"] = [
            {"task": "search", "step_index": 0, "summary": "Vector + BM25 hybrid search"},
        ]
        state = _make_state(working_memory=wm)
        result = synthesize_node(state)

        assert "response" in result
        assert len(result["response"]) > 0
        assert "hybrid search" in result["response"]

    @patch("agent.nodes.get_codebase_context", return_value="")
    @patch("agent.nodes.get_model")
    def test_uses_full_working_memory(self, mock_get_model, _mock_ctx):
        """synthesize_node should use _format_working_memory_for_synthesis."""
        ai_msg = AIMessage(content="Answer with entities")
        mock_get_model.return_value = _mock_model_returning(ai_msg)

        wm = empty_working_memory()
        wm["discovered_entities"] = [{"name": "Foo", "type": "class", "location": "foo.py"}]
        wm["relationships"] = [{"source": "Bar", "type": "calls", "target": "Foo"}]
        wm["insights"] = ["Foo is the main class"]
        wm["task_results"] = [{"task": "find Foo", "step_index": 0, "summary": "Found Foo class"}]

        state = _make_state(working_memory=wm)
        synthesize_node(state)

        # Verify the model was called (LCEL chains use __call__, not .invoke())
        mock_get_model.return_value.assert_called()
