"""Tests for agent graph topology — no LLM calls required."""

import pytest


EXPECTED_APP_NODES = {
    "planner",
    "setup_executor",
    "executor_llm",
    "tool_node",
    "aggregate",
    "refinery",
    "synthesizer",
}


class TestGraphTopology:
    """Verify the compiled graph has the correct structure."""

    def test_node_count(self, graph_topology):
        """Graph should have 7 app nodes + __start__ + __end__ = 9."""
        assert len(graph_topology.nodes) == 9

    def test_expected_nodes_present(self, graph_topology):
        """All 7 application node IDs must exist."""
        node_ids = set(graph_topology.nodes)
        assert EXPECTED_APP_NODES.issubset(node_ids)

    def test_start_and_end_present(self, graph_topology):
        """__start__ and __end__ sentinel nodes must exist."""
        node_ids = set(graph_topology.nodes)
        assert "__start__" in node_ids
        assert "__end__" in node_ids

    def test_entry_point(self, graph_topology):
        """__start__ should connect to planner."""
        edges = [(e.source, e.target) for e in graph_topology.edges]
        assert ("__start__", "planner") in edges

    def test_terminal_node(self, graph_topology):
        """synthesizer should connect to __end__."""
        edges = [(e.source, e.target) for e in graph_topology.edges]
        assert ("synthesizer", "__end__") in edges

    def test_linear_edges(self, graph_topology):
        """Verify the 4 direct (non-conditional) edges."""
        edges = [(e.source, e.target) for e in graph_topology.edges]
        expected_linear = [
            ("__start__", "planner"),
            ("planner", "setup_executor"),
            ("setup_executor", "executor_llm"),
            ("tool_node", "executor_llm"),
            ("aggregate", "refinery"),
            ("synthesizer", "__end__"),
        ]
        for src, tgt in expected_linear:
            assert (src, tgt) in edges, f"Missing edge: {src} → {tgt}"

    def test_conditional_edges_from_executor_llm(self, graph_topology):
        """executor_llm should have conditional edges to tool_node and aggregate."""
        targets = {e.target for e in graph_topology.edges if e.source == "executor_llm"}
        assert "tool_node" in targets
        assert "aggregate" in targets

    def test_conditional_edges_from_refinery(self, graph_topology):
        """refinery should have conditional edges to planner and synthesizer."""
        targets = {e.target for e in graph_topology.edges if e.source == "refinery"}
        assert "planner" in targets
        assert "synthesizer" in targets

    def test_executor_inner_loop(self, graph_topology):
        """executor_llm → tool_node and tool_node → executor_llm form a cycle."""
        edges = [(e.source, e.target) for e in graph_topology.edges]
        assert ("executor_llm", "tool_node") in edges
        assert ("tool_node", "executor_llm") in edges

    def test_mermaid_generation(self, graph_topology):
        """draw_mermaid() should produce a valid Mermaid string."""
        mermaid = graph_topology.draw_mermaid()
        assert isinstance(mermaid, str)
        assert len(mermaid) > 50
        assert "planner" in mermaid
        assert "synthesizer" in mermaid

    def test_graph_is_compilable(self, compiled_graph):
        """Compiled graph should have invoke and stream methods."""
        assert hasattr(compiled_graph, "invoke")
        assert hasattr(compiled_graph, "stream")
