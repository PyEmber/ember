"""Unit tests for XCSGraph functionality.

This module tests the functionality of the XCSGraph class, verifying:
    - Proper management of nodes and edges,
    - Accurate topological sorting,
    - Detection of cycles, and
    - Correct merging of graph instances with namespace handling.
"""

from typing import Any, Dict, List

import pytest

from ember.xcs.graph.xcs_graph import XCSGraph, merge_xcs_graphs


def dummy_operator(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Simulated operator that returns the provided inputs.

    Args:
        inputs (Dict[str, Any]): A dictionary of input values.

    Returns:
        Dict[str, Any]: The same dictionary of input values.
    """
    return inputs


def test_add_node_and_edge() -> None:
    """Verify that nodes and edges are correctly added to an XCSGraph.

    This test creates a graph with two nodes connected by an edge, then asserts that:
      - Both nodes are present in the graph's registry.
      - The outbound edge from the first node and inbound edge to the second node are set correctly.
    """
    graph: XCSGraph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="node1")
    graph.add_node(operator=dummy_operator, node_id="node2")
    graph.add_edge(from_id="node1", to_id="node2")
    assert "node1" in graph.nodes, "Expected 'node1' to be present in the graph."
    assert "node2" in graph.nodes, "Expected 'node2' to be present in the graph."
    assert graph.nodes["node1"].outbound_edges == [
        "node2"
    ], "Expected 'node1' to have an outbound edge to 'node2'."
    assert graph.nodes["node2"].inbound_edges == [
        "node1"
    ], "Expected 'node2' to have an inbound edge from 'node1'."


def test_duplicate_node_id_error() -> None:
    """Ensure that adding a duplicate node ID raises a ValueError.

    Attempts to add a node with an identifier that already exists in the graph should raise a ValueError,
    enforcing the uniqueness of node IDs.
    """
    graph: XCSGraph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="dup")
    with pytest.raises(ValueError, match="Node with ID 'dup' already exists."):
        graph.add_node(operator=dummy_operator, node_id="dup")


def test_topological_sort_linear() -> None:
    """Validate topological sorting on a linear graph.

    Constructs a simple linear graph (A -> B -> C) and ensures that the topological sort
    yields the order ['A', 'B', 'C'].

    Raises:
        AssertionError: If the sorted order does not match the expected sequence.
    """
    graph: XCSGraph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="A")
    graph.add_node(operator=dummy_operator, node_id="B")
    graph.add_node(operator=dummy_operator, node_id="C")
    graph.add_edge(from_id="A", to_id="B")
    graph.add_edge(from_id="B", to_id="C")
    order: List[str] = graph.topological_sort()
    assert order == [
        "A",
        "B",
        "C",
    ], "Topological sort should yield ['A', 'B', 'C'] for a linear graph."


def test_topological_sort_diamond() -> None:
    """Verify topological sorting on a diamond-shaped graph.

    Constructs a diamond-shaped graph:
          A
         / \
        B   C
         \ /
          D
    The test asserts that 'A' is sorted first, 'D' is sorted last, and 'B' and 'C' appear in between.

    Raises:
        AssertionError: If the sorted order does not comply with the diamond topology constraints.
    """
    graph: XCSGraph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="A")
    graph.add_node(operator=dummy_operator, node_id="B")
    graph.add_node(operator=dummy_operator, node_id="C")
    graph.add_node(operator=dummy_operator, node_id="D")
    graph.add_edge(from_id="A", to_id="B")
    graph.add_edge(from_id="A", to_id="C")
    graph.add_edge(from_id="B", to_id="D")
    graph.add_edge(from_id="C", to_id="D")
    order: List[str] = graph.topological_sort()
    assert order[0] == "A", "The first node should be 'A'."
    assert order[-1] == "D", "The last node should be 'D'."
    assert set(order[1:-1]) == {"B", "C"}, "Intermediate nodes should be 'B' and 'C'."


def test_cycle_detection() -> None:
    """Test that a cycle in the graph triggers a ValueError during topological sorting.

    Constructs a cyclic graph with two nodes forming a loop. The topological_sort
    method is expected to detect the cycle and raise a ValueError.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    graph: XCSGraph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="1")
    graph.add_node(operator=dummy_operator, node_id="2")
    graph.add_edge(from_id="1", to_id="2")
    graph.add_edge(from_id="2", to_id="1")
    with pytest.raises(ValueError, match="Graph contains a cycle"):
        graph.topological_sort()


def test_merge_xcs_graphs_namespace() -> None:
    """Test merging two graphs with namespace prefixing.

    Verifies that merging a base graph with an additional graph results in node IDs
    from the additional graph being correctly prefixed with the given namespace to avoid conflicts.
    """
    base_graph: XCSGraph = XCSGraph()
    additional_graph: XCSGraph = XCSGraph()
    base_graph.add_node(operator=dummy_operator, node_id="base1")
    additional_graph.add_node(operator=dummy_operator, node_id="add1")
    merged_graph: XCSGraph = merge_xcs_graphs(
        base=base_graph, additional=additional_graph, namespace="Test"
    )
    assert (
        "base1" in merged_graph.nodes
    ), "Merged graph must contain 'base1' from the base graph."
    assert (
        "Test_add1" in merged_graph.nodes
    ), "Merged graph must contain 'Test_add1' from the namespaced additional graph."


def test_merge_with_duplicates() -> None:
    """Ensure proper renaming when merging graphs with duplicate node IDs.

    When both the base and additional graphs contain a node with the identical ID,
    the additional node should be renamed with the provided namespace to guarantee uniqueness.

    Raises:
        AssertionError: If the merged graph does not include the correctly renamed node.
    """
    base_graph: XCSGraph = XCSGraph()
    additional_graph: XCSGraph = XCSGraph()
    base_graph.add_node(operator=dummy_operator, node_id="shared")
    additional_graph.add_node(operator=dummy_operator, node_id="shared")
    merged_graph: XCSGraph = merge_xcs_graphs(
        base=base_graph, additional=additional_graph, namespace="Ns"
    )
    assert (
        "shared" in merged_graph.nodes
    ), "Merged graph should contain 'shared' from the base graph."
    assert any(
        node_id.startswith("Ns_shared") for node_id in merged_graph.nodes
    ), "Merged graph must include a namespaced version of the duplicate node from the additional graph."
