"""Unit tests for TopologicalSchedulerWithParallelDispatch.

This module tests parallel execution using the TopologicalSchedulerWithParallelDispatch.
"""

from typing import Any, Dict

from ember.xcs.engine.xcs_engine import (
    TopologicalSchedulerWithParallelDispatch,
    compile_graph,
)
from ember.xcs.graph.xcs_graph import XCSGraph


def dummy_operator(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """A simple operator that multiplies input 'value' by 2."""
    return {"out": inputs["value"] * 2}


def test_parallel_scheduler() -> None:
    """Tests parallel execution with TopologicalSchedulerWithParallelDispatch."""
    graph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="node1")
    plan = compile_graph(graph=graph)
    scheduler = TopologicalSchedulerWithParallelDispatch(max_workers=2)
    results = scheduler.run_plan(plan=plan, global_input={"value": 3}, graph=graph)
    assert results["node1"] == {"out": 6}
