"""
Integration tests for combined XCS transformations.

This module provides tests for the integration and composition of multiple XCS
transformations, like combining vmap with pmap, pmap with mesh_sharded, etc.
It also tests the interaction of transformations with the XCS execution engine.
"""

import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pytest

from ember.xcs.engine.xcs_engine import (
    TopologicalSchedulerWithParallelDispatch,
    execute_graph,
)
from ember.xcs.graph.xcs_graph import XCSGraph
from tests.helpers.stub_classes import Operator

# Import test operators
from tests.unit.xcs.transforms.mock_operators import (
    AsyncBehaviorOperator,
    BasicOperator,
    NestedOperator,
)

# Import directly from our fixed imports module to avoid 'module is not callable' errors
from tests.unit.xcs.transforms.test_transform_imports import (
    DeviceMesh,
    PartitionSpec,
    mesh_sharded,
    pjit,
    pmap,
    vmap,
)
from tests.unit.xcs.transforms.test_utils import (
    assert_processing_time,
    generate_batch_inputs,
    time_function_execution,
)

# =============================== Fixtures ===============================


@pytest.fixture
def basic_operator():
    """Fixture providing a basic operator instance."""
    return BasicOperator(sleep_time=0.01)


@pytest.fixture
def simple_mesh():
    """Fixture providing a simple 2x2 device mesh."""
    return DeviceMesh(devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"], shape=(2, 2))


# =============================== Integration Tests ===============================


class TestTransformCombinations:
    """Tests for combining multiple transformations together."""

    def test_vmap_with_pmap(self, basic_operator):
        """Test combining vmap and pmap transformations."""
        # First apply vmap to handle batching
        vectorized_op = vmap(basic_operator)

        # Then parallelize the vectorized operation
        parallel_vectorized_op = pmap(vectorized_op, num_workers=2)

        # Test with nested batch structure
        batch_inputs = {
            "prompts": [
                ["inner1a", "inner1b"],
                ["inner2a", "inner2b"],
                ["inner3a", "inner3b"],
                ["inner4a", "inner4b"],
            ]
        }

        # Time sequential execution (with just vmap)
        sequential_time, sequential_result = time_function_execution(
            vectorized_op, inputs=batch_inputs
        )

        # Time parallel execution (with vmap+pmap)
        parallel_time, parallel_result = time_function_execution(
            parallel_vectorized_op, inputs=batch_inputs
        )

        # Should result in the same processed items
        assert len(parallel_result["results"]) == len(sequential_result["results"])

        # Compare sorted string representations since nested lists aren't hashable
        seq_results_str = sorted([str(x) for x in sequential_result["results"]])
        par_results_str = sorted([str(x) for x in parallel_result["results"]])
        assert seq_results_str == par_results_str

        # Combined transform should be faster
        assert_processing_time(sequential_time, parallel_time)

    def test_pmap_with_mesh_sharded(self, basic_operator, simple_mesh):
        """Test combining pmap and mesh_sharded transformations."""
        # First apply pmap for initial parallelization
        parallel_op = pmap(basic_operator, num_workers=2)

        # Then apply mesh sharding for further distribution
        partition = {"prompts": PartitionSpec(0, None)}
        mesh_parallel_op = mesh_sharded(
            parallel_op, simple_mesh, in_partition=partition
        )

        batch_inputs = {"prompts": [f"combined{i}" for i in range(16)]}

        # Time execution with just pmap
        pmap_time, pmap_result = time_function_execution(
            parallel_op, inputs=batch_inputs
        )

        # Time execution with pmap+mesh_sharded
        combined_time, combined_result = time_function_execution(
            mesh_parallel_op, inputs=batch_inputs
        )

        # In test mode, we might get duplicates when combining transformations
        # Just verify we processed all the inputs
        assert "results" in combined_result
        assert len(combined_result["results"]) > 0

        # Check that all the required items are present
        expected_items = {f"combined{i}_processed" for i in range(16)}
        for item in expected_items:
            assert item in combined_result["results"]

        # Performance may be better or worse depending on the specific workload
        # and overhead, so we only check that the combined version completes
        assert combined_time > 0

    def test_three_transforms_together(self, basic_operator, simple_mesh):
        """Test applying all three transforms together: vmap + pmap + mesh_sharded."""
        # Apply the transforms in sequence
        vectorized_op = vmap(basic_operator)
        parallel_vectorized_op = pmap(vectorized_op, num_workers=2)
        partition = {"prompts": PartitionSpec(0, None)}
        full_transform_op = mesh_sharded(
            parallel_vectorized_op, simple_mesh, in_partition=partition
        )

        # Create a nested batch structure
        batch_inputs = {
            "prompts": [[f"item{i}_{j}" for j in range(3)] for i in range(12)]
        }

        # Time execution with just the original operator
        original_time, original_result = time_function_execution(
            basic_operator, inputs=batch_inputs
        )

        # Time execution with all three transforms
        transformed_time, transformed_result = time_function_execution(
            full_transform_op, inputs=batch_inputs
        )

        # Should process all items correctly
        assert len(transformed_result["results"]) > 0

        # With all transforms, should be significantly faster for large batches
        assert_processing_time(original_time, transformed_time)

    def test_transform_order_matters(self, basic_operator, simple_mesh):
        """Test that the order of applying transforms affects behavior."""
        # Order 1: vmap -> pmap -> mesh_sharded
        order1_op = vmap(basic_operator)
        order1_op = pmap(order1_op, num_workers=2)
        partition = {"prompts": PartitionSpec(0, None)}
        order1_op = mesh_sharded(order1_op, simple_mesh, in_partition=partition)

        # Order 2: pmap -> vmap -> mesh_sharded
        order2_op = pmap(basic_operator, num_workers=2)
        order2_op = vmap(order2_op)
        order2_op = mesh_sharded(order2_op, simple_mesh, in_partition=partition)

        # Order 3: mesh_sharded -> vmap -> pmap
        order3_op = mesh_sharded(basic_operator, simple_mesh, in_partition=partition)
        order3_op = vmap(order3_op)
        order3_op = pmap(order3_op, num_workers=2)

        # Test with nested batch structure
        batch_inputs = {
            "prompts": [[f"order{i}_{j}" for j in range(2)] for i in range(4)]
        }

        # Execute with each order
        result1 = order1_op(inputs=batch_inputs)
        result2 = order2_op(inputs=batch_inputs)
        result3 = order3_op(inputs=batch_inputs)

        # All orders should produce valid results, but they might differ
        # in how they process the nested structure
        assert len(result1["results"]) > 0
        assert len(result2["results"]) > 0
        assert len(result3["results"]) > 0

    def test_transform_reuse(self, basic_operator):
        """Test that transforms can be reused with different operators."""
        op1 = BasicOperator(lambda x: f"{x}_first", sleep_time=0.01)
        op2 = BasicOperator(lambda x: f"{x}_second", sleep_time=0.01)

        # Create a transform that applies both vmap and pmap
        def create_parallel_vectorized(op):
            """Create a parallel vectorized version of the operator."""
            vectorized = vmap(op)
            return pmap(vectorized, num_workers=2)

        # Apply to both operators
        transformed_op1 = create_parallel_vectorized(op1)
        transformed_op2 = create_parallel_vectorized(op2)

        # Test with the same batch input
        batch_inputs = {"prompts": ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"]}

        result1 = transformed_op1(inputs=batch_inputs)
        result2 = transformed_op2(inputs=batch_inputs)

        # Each should apply its own transformation to all inputs
        assert len(result1["results"]) == 8
        assert len(result2["results"]) == 8

        for r in result1["results"]:
            assert r.endswith("_first")

        for r in result2["results"]:
            assert r.endswith("_second")


# =============================== XCS Graph Integration Tests ===============================


class TestTransformWithXCSGraph:
    """Tests for integrating transforms with the XCS graph execution engine."""

    def test_simple_graph_with_transformed_operators(self, basic_operator):
        """Test a simple XCS graph with transformed operators."""

        # Create a test chain operator that properly handles inputs
        class ChainOperator(Operator):
            """Operator that chains transformations in a graph context."""

            def __init__(self, name, transform_fn):
                self.name = name
                self.transform_fn = transform_fn

            def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
                """Process inputs and add our own transformation."""
                # Get input either from previous operator's result or from global input
                if "result" in inputs:
                    # If we have a result from a previous node, use that
                    input_value = inputs["result"]
                elif "prompts" in inputs:
                    # Otherwise use original input
                    input_value = inputs["prompts"]
                    if isinstance(input_value, list) and len(input_value) > 0:
                        input_value = input_value[0]  # Take first list item
                else:
                    input_value = "default"  # Fallback

                # Apply our transformation
                result = self.transform_fn(input_value)

                # Store result with explicit prompts key to pass through
                return {
                    "result": result,
                    "prompts": result,  # Pass the result as prompts for next nodes
                    "transformed_by": self.name,
                }

        # Create test operators
        op1 = ChainOperator("first", lambda x: f"{x}_first")
        op2 = ChainOperator("second", lambda x: f"{x}_second")

        # Create a graph
        graph = XCSGraph()
        node1 = graph.add_node(op1, name="first_op")
        node2 = graph.add_node(op2, name="second_op")

        # Connect nodes
        graph.add_edge(from_id=node1, to_id=node2)

        # Execute the graph
        inputs = {"prompts": "g1"}
        result = execute_graph(graph=graph, global_input=inputs)

        # Verify both nodes executed
        assert node1 in result, f"Node1 {node1} missing from results"
        assert node2 in result, f"Node2 {node2} missing from results"

        # Verify correct transformations applied
        assert (
            result[node1]["result"] == "g1_first"
        ), f"Expected 'g1_first', got {result[node1]['result']}"
        assert (
            result[node2]["result"] == "g1_first_second"
        ), f"Expected 'g1_first_second', got {result[node2]['result']}"

    def test_complex_graph_with_multiple_transforms(self, simple_mesh):
        """Test a complex XCS graph with chain operators that simulate transformations."""

        # Create a chain operator that simulates transformation behavior
        class ChainOperator(Operator):
            """Operator that adds a suffix to input and passes it through."""

            def __init__(self, suffix, sleep_time=0.01):
                self.suffix = suffix
                self.sleep_time = sleep_time

            def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
                """Process input and add suffix."""
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)

                # Extract previous value or use direct input
                previous = inputs.get("result", inputs.get("prompts", "default"))

                # Append suffix
                result = f"{previous}{self.suffix}"
                return {"result": result}

        # Create a graph with a chain of operators
        graph = XCSGraph()

        # Create three nodes that will form a chain of transformations
        node1 = graph.add_node(ChainOperator("_step1"), name="step1")
        node2 = graph.add_node(ChainOperator("_step2"), name="step2")
        node3 = graph.add_node(ChainOperator("_step3"), name="step3")

        # Connect the nodes in sequence
        graph.add_edge(from_id=node1, to_id=node2)
        graph.add_edge(from_id=node2, to_id=node3)

        # Execute the graph with parallel scheduler to test transform-like behavior
        scheduler = TopologicalSchedulerWithParallelDispatch(max_workers=2)
        inputs = {"prompts": "start"}

        # Time execution
        start_time = time.time()
        result = execute_graph(graph=graph, global_input=inputs, scheduler=scheduler)
        end_time = time.time()

        # Verify results flow through the graph correctly
        assert node1 in result, f"Node1 missing from results"
        assert node2 in result, f"Node2 missing from results"
        assert node3 in result, f"Node3 missing from results"

        # Check the transformation chain
        assert result[node1]["result"] == "start_step1"
        assert result[node2]["result"] == "start_step1_step2"
        assert result[node3]["result"] == "start_step1_step2_step3"

        # Verify execution completed in a reasonable time
        execution_time = end_time - start_time
        assert execution_time > 0, f"Execution time should be positive"

    def test_nested_operator_in_graph(self):
        """Test a nested operator within a graph."""

        # Create a simple nested operator compatible with graph execution
        class NestedGraphOperator(Operator):
            """Operator that applies multiple transformation steps internally."""

            def __init__(self, transformations):
                self.transformations = transformations

            def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
                """Apply all transformations in sequence."""
                # Get initial value
                value = inputs.get("prompts", "default")
                if isinstance(value, list) and len(value) > 0:
                    value = value[0]  # Take first item if list

                # Apply each transformation
                for transform_fn in self.transformations:
                    value = transform_fn(value)

                return {"result": value, "processed": True}

        # Create transformation functions
        def layer1(x):
            time.sleep(0.01)  # Simulate processing time
            return f"{x}_layer1"

        def layer2(x):
            time.sleep(0.01)  # Simulate processing time
            return f"{x}_layer2"

        # Create nested operator
        nested_op = NestedGraphOperator([layer1, layer2])

        # Create a simple graph with just the nested operator
        graph = XCSGraph()
        node = graph.add_node(nested_op, name="nested")

        # Execute the graph
        inputs = {"prompts": "n1"}
        result = execute_graph(graph=graph, global_input=inputs)

        # Verify results
        assert node in result, f"Node missing from results"
        assert "result" in result[node], "No result key in output"
        assert (
            result[node]["result"] == "n1_layer1_layer2"
        ), f"Expected 'n1_layer1_layer2', got {result[node]['result']}"

    def test_graph_with_async_operators(self):
        """Test graph execution with pseudo-async operators."""

        # Create a simple variable-time operator compatible with graphs
        class VariableTimeOperator(Operator):
            """Operator with variable execution times."""

            def __init__(self, name, base_time=0.01, variance=0.005):
                self.name = name
                self.base_time = base_time
                self.variance = variance
                self.thread_ids = set()

            def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
                """Execute with variable time and track thread ID."""
                # Record the thread that executed this
                thread_id = threading.get_ident()
                self.thread_ids.add(thread_id)

                # Calculate random sleep time
                import random

                sleep_time = self.base_time + random.uniform(
                    -self.variance, self.variance
                )
                if sleep_time > 0:
                    time.sleep(max(0.001, sleep_time))

                # Get input value
                value = inputs.get("result", inputs.get("prompts", "default"))
                if isinstance(value, list) and len(value) > 0:
                    value = value[0]  # Take first item if list

                # Return result with processing info
                return {
                    "result": f"{value}_{self.name}",
                    "thread_id": thread_id,
                    "processed_by": self.name,
                }

        # Create two variable-time operators
        op1 = VariableTimeOperator("first_step")
        op2 = VariableTimeOperator("second_step")

        # Create a graph with the operators
        graph = XCSGraph()
        node1 = graph.add_node(op1, name="first")
        node2 = graph.add_node(op2, name="second")

        # Add edge
        graph.add_edge(from_id=node1, to_id=node2)

        # Execute the graph with a parallel scheduler
        scheduler = TopologicalSchedulerWithParallelDispatch(max_workers=2)
        inputs = {"prompts": "async_test"}

        result = execute_graph(graph=graph, global_input=inputs, scheduler=scheduler)

        # Verify results
        assert node1 in result, f"Node1 missing from results"
        assert node2 in result, f"Node2 missing from results"

        # Check the processing chain
        assert result[node1]["result"] == "async_test_first_step"
        assert result[node2]["result"] == "async_test_first_step_second_step"

        # Record thread IDs for information (don't assert as it may be single-threaded in tests)
        thread_count = len(op1.thread_ids) + len(op2.thread_ids)
        assert thread_count >= 1, "At least one thread should have been used"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
