"""
XCS execution engine for computational graphs.

This module provides the execution engine for XCS computational graphs,
including schedulers for determining execution order and dispatchers for
executing the operations.
"""

import dataclasses
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from ember.xcs.graph.xcs_graph import XCSGraph, XCSNode

logger = logging.getLogger(__name__)


@dataclass
class XCSTask:
    """A computation task in an XCS plan.

    Attributes:
        operator: The operator function to execute
        inputs: Input node IDs for this task
        outputs: Output node IDs for this task
    """

    operator: Callable[..., Dict[str, Any]]
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


@dataclass
class XCSPlan:
    """A plan for executing a computation graph.

    Contains a mapping of node IDs to executable tasks.

    Attributes:
        tasks: Dictionary mapping node IDs to XCSTasks
        graph_id: Unique identifier for the source graph
    """

    tasks: Dict[str, XCSTask] = field(default_factory=dict)
    graph_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def add_task(self, node_id: str, task: XCSTask) -> None:
        """Add a task to the plan."""
        self.tasks[node_id] = task


class IScheduler(Protocol):
    """Protocol for graph scheduler implementations."""

    def schedule(self, graph: XCSGraph) -> List[List[str]]:
        """Schedule nodes for execution.

        Args:
            graph: The computational graph to schedule.

        Returns:
            A list of execution waves, where each wave is a list of node IDs
            that can be executed in parallel.
        """
        ...


class TopologicalScheduler:
    """Basic scheduler that creates execution waves based on topological sorting."""

    def schedule(self, graph: XCSGraph) -> List[List[str]]:
        """Schedule nodes into waves based on topological dependencies.

        Args:
            graph: The computational graph to schedule.

        Returns:
            A list of execution waves, where each wave is a list of node IDs
            that can be executed in parallel.
        """
        # Get topological order
        topo_order = graph.topological_sort()

        # Organize into waves based on dependencies
        waves: List[List[str]] = []
        completed_nodes: Set[str] = set()

        while topo_order:
            # Find all nodes whose dependencies are satisfied
            current_wave = []
            remaining = []

            for node_id in topo_order:
                node = graph.nodes[node_id]
                if all(dep in completed_nodes for dep in node.inbound_edges):
                    current_wave.append(node_id)
                else:
                    remaining.append(node_id)

            # Add the wave and update completed nodes
            waves.append(current_wave)
            completed_nodes.update(current_wave)
            topo_order = remaining

        return waves


class TopologicalSchedulerWithParallelDispatch(TopologicalScheduler):
    """Scheduler that groups nodes for parallel execution."""

    def __init__(self, max_workers: Optional[int] = None):
        """Initialize with optional worker count limit.

        Args:
            max_workers: Maximum number of worker threads to use for each wave.
        """
        self.max_workers = max_workers
        super().__init__()

    def run_plan(
        self, *, plan: XCSPlan, global_input: Dict[str, Any], graph: XCSGraph
    ) -> Dict[str, Any]:
        """Execute a compiled plan in parallel.

        Args:
            plan: The XCSPlan to execute
            global_input: Input data for the graph
            graph: The source graph (used for reference)

        Returns:
            A dictionary of results for each node
        """
        results: Dict[str, Dict[str, Any]] = {}

        # Get the waves by running schedule on the graph
        waves = self.schedule(graph)

        # Execute each wave in parallel
        for wave in waves:
            # Create a thread pool for parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}

                # Submit each node in the wave for execution
                for node_id in wave:
                    if node_id in plan.tasks:
                        task = plan.tasks[node_id]

                        # Collect inputs from predecessors
                        inputs = global_input.copy()
                        for pred_id in task.inputs:
                            if pred_id in results:
                                inputs.update(results[pred_id])

                        # Add node name to inputs for tracking
                        if "node_name" in global_input:
                            inputs["node_name"] = node_id

                        # Submit for execution
                        futures[executor.submit(task.operator, inputs=inputs)] = node_id

                # Collect results
                for future in as_completed(futures):
                    node_id = futures[future]
                    try:
                        results[node_id] = future.result()
                    except Exception as e:
                        logger.exception(f"Error executing node {node_id}: {e}")
                        results[node_id] = {"error": str(e)}

        return results


def execute_graph(
    graph: XCSGraph,
    global_input: Dict[str, Any],
    scheduler: Optional[IScheduler] = None,
) -> Dict[str, Dict[str, Any]]:
    """Execute a computational graph with the given input.

    Args:
        graph: The computational graph to execute.
        global_input: The input data for the graph.
        scheduler: Optional scheduler to determine execution order.

    Returns:
        A dictionary mapping node IDs to their execution results.
    """
    if scheduler is None:
        scheduler = TopologicalScheduler()

    # Schedule the nodes
    waves = scheduler.schedule(graph)

    # Execute the graph
    results: Dict[str, Dict[str, Any]] = {}

    for wave in waves:
        # Single-threaded execution for simple scheduler
        if isinstance(scheduler, TopologicalScheduler) and not isinstance(
            scheduler, TopologicalSchedulerWithParallelDispatch
        ):
            for node_id in wave:
                node = graph.nodes[node_id]

                # Collect inputs from predecessors or use global input for source nodes
                inputs = global_input.copy() if not node.inbound_edges else {}
                for pred_id in node.inbound_edges:
                    inputs.update(results[pred_id])

                # Add node_id to inputs for tracking in test functions
                if "node_name" in global_input:
                    inputs["node_name"] = node_id

                # Execute the node
                try:
                    node_result = node.operator(inputs=inputs)
                    results[node_id] = node_result
                except Exception as e:
                    logger.exception(f"Error executing node {node_id}: {e}")
                    results[node_id] = {"error": str(e)}

        # Parallel execution for parallel scheduler
        else:
            with ThreadPoolExecutor(
                max_workers=getattr(scheduler, "max_workers", None)
            ) as executor:
                futures = {}

                # Start all jobs in the wave
                for node_id in wave:
                    node = graph.nodes[node_id]

                    # Collect inputs from predecessors or use global input for source nodes
                    inputs = global_input.copy() if not node.inbound_edges else {}
                    for pred_id in node.inbound_edges:
                        inputs.update(results[pred_id])

                    # Add node_id to inputs for tracking in test functions
                    if "node_name" in global_input:
                        inputs["node_name"] = node_id

                    # Submit the job
                    futures[executor.submit(node.operator, inputs=inputs)] = node_id

                # Collect results
                for future in as_completed(futures):
                    node_id = futures[future]
                    try:
                        results[node_id] = future.result()
                    except Exception as e:
                        logger.exception(f"Error executing node {node_id}: {e}")
                        results[node_id] = {"error": str(e)}

    return results


def compile_graph(graph: XCSGraph) -> XCSPlan:
    """Compile an XCS graph into an execution plan.

    This function analyzes a graph and creates a plan that arranges
    tasks in the proper execution order with their dependencies.

    Args:
        graph: The XCS graph to compile

    Returns:
        An XCSPlan ready for execution

    Raises:
        ValueError: If the graph contains cycles or invalid nodes
    """
    # Ensure graph is valid with a topological sort
    topo_order = graph.topological_sort()

    # Create a new plan
    plan = XCSPlan()

    # Convert each node into a task with proper inputs and outputs
    for node_id in topo_order:
        node = graph.nodes[node_id]

        # Create a task for this node
        task = XCSTask(
            operator=node.operator,
            inputs=node.inbound_edges.copy(),
            outputs=node.outbound_edges.copy(),
        )

        # Add the task to the plan
        plan.add_task(node_id, task)

    return plan
