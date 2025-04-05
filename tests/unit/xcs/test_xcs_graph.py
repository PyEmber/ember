# File: tests/test_xcs_graph.py
"""
Tests for XCSGraph: node creation, edge connectivity, topological sorting, cycle detection, and graph merging.

These tests now also check that merging graphs using namespace prefixes works as expected.
"""

import pytest

from ember.xcs.graph.xcs_graph import XCSGraph


def dummy_operator(*, inputs: dict) -> dict:
    return inputs


def test_add_node_and_edge() -> None:
    graph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="node1")
    graph.add_node(operator=dummy_operator, node_id="node2")
    graph.add_edge(from_id="node1", to_id="node2")
    assert "node1" in graph.nodes
    assert "node2" in graph.nodes
    assert "node1" in graph.nodes["node2"].inbound_edges
    assert "node2" in graph.nodes["node1"].outbound_edges


def test_duplicate_node_id_error() -> None:
    graph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="dup")
    with pytest.raises(ValueError):
        graph.add_node(operator=dummy_operator, node_id="dup")


def test_topological_sort() -> None:
    graph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="A")
    graph.add_node(operator=dummy_operator, node_id="B")
    graph.add_node(operator=dummy_operator, node_id="C")
    graph.add_edge(from_id="A", to_id="B")
    graph.add_edge(from_id="B", to_id="C")
    order = graph.topological_sort()
    assert order.index("A") < order.index("B") < order.index("C")


def test_cycle_detection() -> None:
    graph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="1")
    graph.add_node(operator=dummy_operator, node_id="2")
    graph.add_edge(from_id="1", to_id="2")
    # Create a cycle.
    graph.add_edge(from_id="2", to_id="1")
    with pytest.raises(ValueError):
        graph.topological_sort()


# (Optional) Test merging of two graphs with namespace prefixes.
def test_merge_xcs_graphs_namespace() -> None:
    from ember.xcs.graph.xcs_graph import merge_xcs_graphs

    base = XCSGraph()
    additional = XCSGraph()
    base.add_node(operator=dummy_operator, node_id="base1")
    additional.add_node(operator=dummy_operator, node_id="add1")
    merged = merge_xcs_graphs(base=base, additional=additional, namespace="Test")
    # Check that the additional node ID was namespaced.
    namespaced_node = "Test_add1"
    assert namespaced_node in merged.nodes
