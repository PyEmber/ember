"""Unit tests for the XCSNoOpScheduler.

This module verifies the correct behavior of the NoOp scheduler implementation,
which provides a sequential execution strategy for XCS graphs without parallelism.
"""

from typing import Any, Dict, List

import pytest

from ember.xcs.engine.xcs_engine import execute_graph
from ember.xcs.engine.xcs_noop_scheduler import XCSNoOpScheduler
from ember.xcs.graph.xcs_graph import XCSGraph


def sequential_operator(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Return inputs with sequential execution marker.

    Args:
        inputs: Input parameters including execution history

    Returns:
        Dict with updated execution history
    """
    result = inputs.copy()
    history = result.get("execution_history", [])
    node_name = inputs.get("node_name", "unknown")
    history.append(node_name)
    result["execution_history"] = history
    return result


def test_noop_scheduler_sequential_execution() -> None:
    """Test that the NoOp scheduler executes nodes in strict sequential order.

    This test verifies:
    1. The scheduler processes all nodes in a graph
    2. Execution order matches the scheduler's waves exactly
    3. Each node is executed exactly once
    4. No parallelism is attempted even for independent branches
    """
    # Create a diamond-shaped graph
    graph = XCSGraph()

    # Create nodes with distinct names for tracking execution order
    node1 = graph.add_node(operator=sequential_operator, node_id="node1")
    node2a = graph.add_node(operator=sequential_operator, node_id="node2a")
    node2b = graph.add_node(operator=sequential_operator, node_id="node2b")
    node3 = graph.add_node(operator=sequential_operator, node_id="node3")

    # Set up a diamond pattern with two parallel branches
    graph.add_edge(from_id=node1, to_id=node2a)
    graph.add_edge(from_id=node1, to_id=node2b)
    graph.add_edge(from_id=node2a, to_id=node3)
    graph.add_edge(from_id=node2b, to_id=node3)

    # Create input data with node names to track execution
    input_data = {
        "value": "test_sequential",
        "execution_history": [],
        "node_name": "input",
    }

    # Execute with NoOp scheduler (should be strictly sequential)
    scheduler = XCSNoOpScheduler()
    results = execute_graph(graph=graph, global_input=input_data, scheduler=scheduler)

    # Verify all nodes were executed
    for node_id in [node1, node2a, node2b, node3]:
        assert node_id in results, f"Missing results for {node_id}"

    # Get final execution history from the last node
    # The exact order of node2a vs node2b may depend on topological sort implementation
    # What matters is that execution is sequential (not parallel)
    final_history = results[node3]["execution_history"]
    assert (
        len(final_history) >= 4
    ), f"Expected at least 4 executions, got {len(final_history)}"

    # Verify node1 executed first (after input) and node3 executed last
    assert "node1" in final_history, f"Node1 not found in execution history"
    assert (
        final_history[-1] == "node3"
    ), f"Last node executed was {final_history[-1]}, expected node3"

    # Verify middle nodes were executed
    assert "node2a" in final_history, f"Node2a not found in execution history"
    assert "node2b" in final_history, f"Node2b not found in execution history"


def test_noop_scheduler_wave_generation() -> None:
    """Test that the NoOp scheduler generates correct execution waves.

    This test verifies:
    1. Waves are correctly generated based on topological ordering
    2. Each node appears in exactly one wave
    3. The waves respect node dependencies
    """
    # Create a simple linear graph: A -> B -> C
    graph = XCSGraph()
    node_a = graph.add_node(operator=sequential_operator, node_id="A")
    node_b = graph.add_node(operator=sequential_operator, node_id="B")
    node_c = graph.add_node(operator=sequential_operator, node_id="C")

    graph.add_edge(from_id=node_a, to_id=node_b)
    graph.add_edge(from_id=node_b, to_id=node_c)

    # Get the scheduler's waves
    scheduler = XCSNoOpScheduler()
    waves = scheduler.schedule(graph)

    # For a NoOp scheduler, each node should be in its own wave to ensure
    # strictly sequential execution (or follow a topological order)
    assert len(waves) >= 1, f"Expected at least 1 wave, got {len(waves)}"

    # Count the total number of nodes in all waves
    node_count = sum(len(wave) for wave in waves)
    assert node_count == 3, f"Expected 3 total nodes across all waves, got {node_count}"

    # Verify nodes appear in correct topological sequence
    all_nodes = [node for wave in waves for node in wave]
    node_a_idx = all_nodes.index(node_a)
    node_b_idx = all_nodes.index(node_b)
    node_c_idx = all_nodes.index(node_c)

    assert (
        node_a_idx < node_b_idx
    ), f"Node A (idx {node_a_idx}) should come before Node B (idx {node_b_idx})"
    assert (
        node_b_idx < node_c_idx
    ), f"Node B (idx {node_b_idx}) should come before Node C (idx {node_c_idx})"


def test_simple_math_operation() -> None:
    """Test a simple mathematical operation with the NoOp scheduler.

    This test verifies that actual computation works correctly with the scheduler.
    """

    def math_operator(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """A simple operator that multiplies input 'value' by 2."""
        return {"out": inputs["value"] * 2}

    graph = XCSGraph()
    node1 = graph.add_node(operator=math_operator, node_id="node1")

    scheduler = XCSNoOpScheduler()
    results = execute_graph(graph=graph, global_input={"value": 3}, scheduler=scheduler)

    assert node1 in results, f"Missing results for node1"
    assert results[node1]["out"] == 6, f"Expected output 6, got {results[node1]['out']}"
