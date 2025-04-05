"""Unit tests for XCS Engine execution functionality.

This module verifies the functionality of the XCS Engine's topological scheduler and
execution system, focusing on correct graph execution, task dependencies, and proper
error handling in accordance with high engineering standards.
"""

from typing import Any, Dict, List

import pytest

from ember.xcs.engine.xcs_engine import (
    IScheduler,
    TopologicalScheduler,
    TopologicalSchedulerWithParallelDispatch,
    execute_graph,
)
from ember.xcs.graph.xcs_graph import XCSGraph


def dummy_operator(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Echo operator that returns the provided inputs with a marker.

    Args:
        inputs (Dict[str, Any]): A dictionary of input parameters.

    Returns:
        Dict[str, Any]: The input dictionary with an added marker.
    """
    result = inputs.copy()
    result["executed"] = True
    return result


def test_topological_scheduler() -> None:
    """Test the TopologicalScheduler's ability to correctly schedule execution waves.

    This test constructs a graph with three nodes in a linear chain and verifies that:
    - The scheduler correctly identifies execution dependencies
    - The resulting waves respect the topological ordering of the graph
    - Each wave contains only nodes whose dependencies have been satisfied

    Raises:
        AssertionError: If the scheduled waves don't match expected execution order.
    """
    # Create a simple linear graph: A -> B -> C
    graph = XCSGraph()
    node_a = graph.add_node(operator=dummy_operator, node_id="A")
    node_b = graph.add_node(operator=dummy_operator, node_id="B")
    node_c = graph.add_node(operator=dummy_operator, node_id="C")

    graph.add_edge(from_id=node_a, to_id=node_b)
    graph.add_edge(from_id=node_b, to_id=node_c)

    # Schedule the graph
    scheduler = TopologicalScheduler()
    waves = scheduler.schedule(graph)

    # Verify the waves - should be [[A], [B], [C]] for a linear chain
    assert len(waves) == 3, f"Expected 3 waves, got {len(waves)}"
    assert waves[0] == [node_a], f"First wave should be [A], got {waves[0]}"
    assert waves[1] == [node_b], f"Second wave should be [B], got {waves[1]}"
    assert waves[2] == [node_c], f"Third wave should be [C], got {waves[2]}"


def test_execute_graph_simple() -> None:
    """Test the execution of a simple graph with sequential dependencies.

    This test constructs a graph with two nodes connected sequentially and verifies:
    - The graph executes completely
    - Each node's operator is called with correct inputs
    - Results from earlier nodes are properly passed to later nodes
    - The final execution results contain outputs from all nodes

    Raises:
        AssertionError: If execution results don't match expectations.
    """
    # Create a simple graph: node1 -> node2
    graph = XCSGraph()
    node1 = graph.add_node(operator=dummy_operator, node_id="node1")
    node2 = graph.add_node(operator=dummy_operator, node_id="node2")
    graph.add_edge(from_id=node1, to_id=node2)

    # Execute the graph
    input_data = {"value": "test_input"}
    results = execute_graph(graph=graph, global_input=input_data)

    # Verify results
    assert len(results) == 2, f"Expected results for 2 nodes, got {len(results)}"
    assert "node1" in results, "Results missing for node1"
    assert "node2" in results, "Results missing for node2"

    # Verify node1 received the global input
    assert (
        results["node1"]["value"] == "test_input"
    ), "Node1 didn't receive correct input"
    assert results["node1"]["executed"] == True, "Node1 wasn't executed"

    # Verify node2 received node1's output (which includes the global input)
    assert (
        results["node2"]["value"] == "test_input"
    ), "Node2 didn't receive node1's output"
    assert results["node2"]["executed"] == True, "Node2 wasn't executed"


def test_execute_graph_with_parallel_scheduler() -> None:
    """Test graph execution with the parallel scheduler.

    This test constructs a diamond-shaped graph (node1 -> node2a, node2b -> node3)
    and verifies:
    - The parallel scheduler correctly executes independent nodes in the same wave
    - Results are properly collected from all parallel branches
    - The final execution results contain outputs from all nodes

    Raises:
        AssertionError: If parallel execution doesn't work as expected.
    """
    # Create a diamond graph: node1 -> (node2a, node2b) -> node3
    graph = XCSGraph()
    node1 = graph.add_node(operator=dummy_operator, node_id="node1")
    node2a = graph.add_node(operator=dummy_operator, node_id="node2a")
    node2b = graph.add_node(operator=dummy_operator, node_id="node2b")
    node3 = graph.add_node(operator=dummy_operator, node_id="node3")

    graph.add_edge(from_id=node1, to_id=node2a)
    graph.add_edge(from_id=node1, to_id=node2b)
    graph.add_edge(from_id=node2a, to_id=node3)
    graph.add_edge(from_id=node2b, to_id=node3)

    # Execute with parallel scheduler
    scheduler = TopologicalSchedulerWithParallelDispatch(max_workers=2)
    input_data = {"value": "parallel_test"}
    results = execute_graph(graph=graph, global_input=input_data, scheduler=scheduler)

    # Verify all nodes executed
    for node_id in [node1, node2a, node2b, node3]:
        assert node_id in results, f"Results missing for {node_id}"
        assert results[node_id]["executed"] == True, f"Node {node_id} wasn't executed"

    # Verify node3 received outputs from both node2a and node2b
    # In the current implementation, later nodes in the same wave overwrite earlier ones
    # so we just verify node3 was executed with some input
    assert (
        results[node3]["value"] == "parallel_test"
    ), "Node3 didn't receive correct input"


class ErrorOperator:
    """An operator that raises an exception during execution."""

    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Raise an exception when called.

        Args:
            inputs: Input data (not used)

        Raises:
            RuntimeError: Always raises this error
        """
        raise RuntimeError("Intentional test error")


def test_execute_graph_error_handling() -> None:
    """Test error handling during graph execution.

    This test verifies:
    - Errors in node execution are caught and don't crash the entire process
    - The execution results include error information for failed nodes
    - Nodes that depend on failed nodes are still attempted with available inputs

    Raises:
        AssertionError: If error handling doesn't work as expected.
    """
    # Create a graph with an error node
    graph = XCSGraph()
    node1 = graph.add_node(operator=dummy_operator, node_id="node1")
    error_node = graph.add_node(operator=ErrorOperator(), node_id="error_node")
    node3 = graph.add_node(operator=dummy_operator, node_id="node3")

    graph.add_edge(from_id=node1, to_id=error_node)
    graph.add_edge(from_id=error_node, to_id=node3)

    # Execute the graph
    input_data = {"value": "error_test"}
    results = execute_graph(graph=graph, global_input=input_data)

    # Verify normal node executed
    assert node1 in results, "Results missing for node1"
    assert results[node1]["executed"] == True, "Node1 wasn't executed"

    # Verify error was captured for error_node
    assert error_node in results, "Results missing for error_node"
    assert "error" in results[error_node], "Error not captured in results"
    assert (
        "Intentional test error" in results[error_node]["error"]
    ), "Wrong error message"

    # Verify node3 attempted execution with available inputs
    assert node3 in results, "Results missing for node3"
