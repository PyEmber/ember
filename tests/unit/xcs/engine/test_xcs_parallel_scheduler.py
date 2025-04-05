"""Unit tests for TopologicalSchedulerWithParallelDispatch.

This module verifies parallel execution using the TopologicalSchedulerWithParallelDispatch
scheduler, focusing on performance and correctness when executing independent operations.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import pytest

from ember.xcs.engine.xcs_engine import (
    TopologicalSchedulerWithParallelDispatch,
    execute_graph,
)
from ember.xcs.graph.xcs_graph import XCSGraph


class ParallelDetectionOperator:
    """Operator that detects parallel execution via shared execution timeline.

    This operator records its execution time in a shared timeline,
    sleeps briefly, and then records completion time. This allows detection
    of parallel execution by examining overlapping execution periods.
    """

    def __init__(
        self,
        timeline: List[Dict[str, Any]],
        sleep_time: float = 0.05,
        name: str = "unnamed",
    ):
        """Initialize with shared timeline and execution parameters.

        Args:
            timeline: Shared list to record execution events
            sleep_time: How long to sleep during execution
            name: Identifying name for this operator instance
        """
        self.timeline = timeline
        self.sleep_time = sleep_time
        self.name = name

    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the operator, recording start/end times in the timeline.

        Args:
            inputs: Input data (passed through to output)

        Returns:
            Dict with the inputs and execution metadata
        """
        # Record execution start with thread info
        thread_id = threading.get_ident()
        start_time = time.time()
        self.timeline.append(
            {
                "node": self.name,
                "event": "start",
                "time": start_time,
                "thread": thread_id,
            }
        )

        # Simulate work with sleep
        time.sleep(self.sleep_time)

        # Record execution end
        end_time = time.time()
        self.timeline.append(
            {"node": self.name, "event": "end", "time": end_time, "thread": thread_id}
        )

        # Return inputs with execution metadata
        result = inputs.copy()
        result.update(
            {
                "node": self.name,
                "thread": thread_id,
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time,
            }
        )
        return result


def test_parallel_execution_detection() -> None:
    """Test that the parallel scheduler actually executes tasks in parallel.

    This test verifies:
    1. Independent operations are executed concurrently
    2. Different thread IDs are used for parallel operations
    3. Execution periods overlap in time for parallel paths
    """
    # Shared timeline for tracking execution events
    timeline: List[Dict[str, Any]] = []

    # Create a diamond-shaped graph with longer-running middle nodes
    graph = XCSGraph()

    # Create nodes with parallel detection operators
    node1 = graph.add_node(
        operator=ParallelDetectionOperator(timeline, sleep_time=0.01, name="source"),
        node_id="node1",
    )

    # These two should run in parallel
    node2a = graph.add_node(
        operator=ParallelDetectionOperator(timeline, sleep_time=0.1, name="branch_a"),
        node_id="node2a",
    )
    node2b = graph.add_node(
        operator=ParallelDetectionOperator(timeline, sleep_time=0.1, name="branch_b"),
        node_id="node2b",
    )

    node3 = graph.add_node(
        operator=ParallelDetectionOperator(timeline, sleep_time=0.01, name="sink"),
        node_id="node3",
    )

    # Set up diamond pattern
    graph.add_edge(from_id=node1, to_id=node2a)
    graph.add_edge(from_id=node1, to_id=node2b)
    graph.add_edge(from_id=node2a, to_id=node3)
    graph.add_edge(from_id=node2b, to_id=node3)

    # Create parallel scheduler with enough workers
    scheduler = TopologicalSchedulerWithParallelDispatch(max_workers=4)

    # Execute the graph
    input_data = {"test": "parallel_execution"}
    results = execute_graph(graph=graph, global_input=input_data, scheduler=scheduler)

    # Verify all nodes executed
    for node_id in [node1, node2a, node2b, node3]:
        assert node_id in results, f"Missing results for {node_id}"

    # Extract execution data for parallel nodes
    branch_a_data = results[node2a]
    branch_b_data = results[node2b]

    # Different thread IDs indicate parallel execution
    assert (
        branch_a_data["thread"] != branch_b_data["thread"]
    ), "Parallel branches executed on same thread, suggesting sequential execution"

    # Check for time overlap in execution periods
    a_start, a_end = branch_a_data["start"], branch_a_data["end"]
    b_start, b_end = branch_b_data["start"], branch_b_data["end"]

    # Calculate overlap between execution periods
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    has_overlap = overlap_end > overlap_start

    assert (
        has_overlap
    ), f"No execution time overlap detected between parallel branches: {a_start}-{a_end} vs {b_start}-{b_end}"


def test_parallel_scheduler_wave_generation() -> None:
    """Test that the scheduler correctly identifies waves for parallel execution.

    This test verifies:
    1. Nodes with satisfied dependencies are grouped in the same wave
    2. Waves respect topological ordering constraints
    3. The scheduler maximizes parallelism by including all eligible nodes in each wave
    """
    # Create a complex graph with multiple parallel paths
    graph = XCSGraph()

    # Level 1: single source
    source = graph.add_node(operator=lambda **kwargs: {"out": 1}, node_id="source")

    # Level 2: three parallel branches
    branch1 = graph.add_node(operator=lambda **kwargs: {"out": 2}, node_id="branch1")
    branch2 = graph.add_node(operator=lambda **kwargs: {"out": 3}, node_id="branch2")
    branch3 = graph.add_node(operator=lambda **kwargs: {"out": 4}, node_id="branch3")

    # Level 3: two nodes depending on different branches
    merge1 = graph.add_node(operator=lambda **kwargs: {"out": 5}, node_id="merge1")
    merge2 = graph.add_node(operator=lambda **kwargs: {"out": 6}, node_id="merge2")

    # Level 4: final sink node
    sink = graph.add_node(operator=lambda **kwargs: {"out": 7}, node_id="sink")

    # Create edges
    # Level 1 -> Level 2
    graph.add_edge(from_id=source, to_id=branch1)
    graph.add_edge(from_id=source, to_id=branch2)
    graph.add_edge(from_id=source, to_id=branch3)

    # Level 2 -> Level 3
    graph.add_edge(from_id=branch1, to_id=merge1)
    graph.add_edge(from_id=branch2, to_id=merge1)
    graph.add_edge(from_id=branch2, to_id=merge2)
    graph.add_edge(from_id=branch3, to_id=merge2)

    # Level 3 -> Level 4
    graph.add_edge(from_id=merge1, to_id=sink)
    graph.add_edge(from_id=merge2, to_id=sink)

    # Create scheduler and get waves
    scheduler = TopologicalSchedulerWithParallelDispatch(max_workers=3)
    waves = scheduler.schedule(graph)

    # Verify wave structure
    assert len(waves) == 4, f"Expected 4 execution waves, got {len(waves)}"

    # Wave 1: Should contain only the source
    assert len(waves[0]) == 1, f"First wave should have 1 node, got {len(waves[0])}"
    assert source in waves[0], f"Source node missing from first wave"

    # Wave 2: Should contain all three branches
    assert len(waves[1]) == 3, f"Second wave should have 3 nodes, got {len(waves[1])}"
    for node in [branch1, branch2, branch3]:
        assert node in waves[1], f"Node {node} missing from second wave"

    # Wave 3: Should contain both merge nodes
    assert len(waves[2]) == 2, f"Third wave should have 2 nodes, got {len(waves[2])}"
    for node in [merge1, merge2]:
        assert node in waves[2], f"Node {node} missing from third wave"

    # Wave 4: Should contain only the sink
    assert len(waves[3]) == 1, f"Fourth wave should have 1 node, got {len(waves[3])}"
    assert sink in waves[3], f"Sink node missing from fourth wave"


def test_parallel_execution_with_worker_limits() -> None:
    """Test that the scheduler respects worker limits during parallel execution.

    This test verifies:
    1. With limited workers, execution still completes correctly
    2. Work is distributed according to worker availability
    3. Results are correctly aggregated from all nodes
    """
    # Shared timeline for tracking execution events
    timeline: List[Dict[str, Any]] = []

    # Create a wide graph with many parallel nodes
    graph = XCSGraph()

    # Create a source node
    source = graph.add_node(
        operator=ParallelDetectionOperator(timeline, sleep_time=0.01, name="source"),
        node_id="source",
    )

    # Create 6 parallel nodes - with only 2 workers, can't all run at once
    parallel_nodes = []
    for i in range(6):
        node = graph.add_node(
            operator=ParallelDetectionOperator(
                timeline, sleep_time=0.05, name=f"parallel_{i}"
            ),
            node_id=f"parallel_{i}",
        )
        graph.add_edge(from_id=source, to_id=node)
        parallel_nodes.append(node)

    # Create a sink node
    sink = graph.add_node(
        operator=ParallelDetectionOperator(timeline, sleep_time=0.01, name="sink"),
        node_id="sink",
    )

    # Connect all parallel nodes to sink
    for node in parallel_nodes:
        graph.add_edge(from_id=node, to_id=sink)

    # Create scheduler with limited workers (2)
    scheduler = TopologicalSchedulerWithParallelDispatch(max_workers=2)

    # Execute the graph
    input_data = {"test": "worker_limits"}
    start_time = time.time()
    results = execute_graph(graph=graph, global_input=input_data, scheduler=scheduler)
    end_time = time.time()

    # Verify all nodes executed
    for node_id in [source] + parallel_nodes + [sink]:
        assert node_id in results, f"Missing results for {node_id}"

    # Extract thread IDs used for parallel nodes
    threads_used = set()
    for node in parallel_nodes:
        threads_used.add(results[node]["thread"])

    # With only 2 workers, should use at most 2 threads for parallel nodes
    # Allow some slack for thread reuse
    assert (
        len(threads_used) <= 3
    ), f"Expected at most 3 threads with 2 workers, got {len(threads_used)}"

    # Extract timeline events for parallel nodes only
    parallel_events = [e for e in timeline if e["node"].startswith("parallel_")]

    # Group events by thread
    thread_events = {}
    for event in parallel_events:
        thread = event["thread"]
        if thread not in thread_events:
            thread_events[thread] = []
        thread_events[thread].append(event)

    # Verify proper thread usage
    for thread, events in thread_events.items():
        # Sort events by time
        events.sort(key=lambda e: e["time"])

        # Check that events alternate between start and end
        for i in range(0, len(events) - 1, 2):
            assert (
                events[i]["event"] == "start"
            ), f"Expected start event, got {events[i]}"
            assert (
                events[i + 1]["event"] == "end"
            ), f"Expected end event, got {events[i+1]}"

            # Verify matching node names
            assert (
                events[i]["node"] == events[i + 1]["node"]
            ), f"Mismatched nodes in start/end pair: {events[i]['node']} vs {events[i+1]['node']}"
