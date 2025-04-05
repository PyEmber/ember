from typing import Any, Dict, List

from ember.xcs.engine.xcs_engine import IScheduler, XCSPlan
from ember.xcs.graph.xcs_graph import XCSGraph


class XCSNoOpScheduler(IScheduler):
    """
    A single-thread (no concurrency) scheduler for XCS. It runs tasks sequentially.

    This scheduler creates a wave for each node, ensuring strictly sequential execution
    with no parallelism.
    """

    def schedule(self, graph: XCSGraph) -> List[List[str]]:
        """Schedule graph nodes in a strictly sequential manner.

        Each node is placed in its own execution wave to guarantee sequential
        execution with no parallelism.

        Args:
            graph: The computational graph to schedule

        Returns:
            A list of execution waves, where each wave contains a single node ID

        Raises:
            ValueError: If the graph contains cycles
        """
        # Get nodes in topological order
        topo_order = graph.topological_sort()

        # Create one wave per node for strictly sequential execution
        waves = [[node_id] for node_id in topo_order]
        return waves

    def run_plan(
        self, *, plan: XCSPlan, global_input: Dict[str, Any], graph: XCSGraph
    ) -> Dict[str, Any]:
        """Execute a compiled plan in strictly sequential order.

        Args:
            plan: The XCSPlan to execute
            global_input: Input data for the graph
            graph: The source graph (used for reference)

        Returns:
            A dictionary of results for each node
        """
        results: Dict[str, Any] = {}
        # Iterate over tasks by node_id; call the operator directly with the provided global input.
        for node_id, task in plan.tasks.items():
            result = task.operator(inputs=global_input)
            results[node_id] = result
        return results
