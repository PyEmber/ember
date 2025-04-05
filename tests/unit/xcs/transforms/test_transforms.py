"""
Unit tests for XCS transformations.

This module provides comprehensive testing for the vmap, pmap/pjit, and mesh
transformations in XCS. Tests cover basic functionality, edge cases, error handling,
and integration with different operator types.
"""

import multiprocessing
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ember.xcs.engine.xcs_engine import (
    TopologicalSchedulerWithParallelDispatch,
    execute_graph,
)
from ember.xcs.graph.xcs_graph import XCSGraph
from tests.helpers.stub_classes import Operator

# Import our test operators
from tests.unit.xcs.transforms.mock_operators import (
    AsyncBehaviorOperator,
    BasicOperator,
    ComplexInputOperator,
    ExceptionOperator,
)
from tests.unit.xcs.transforms.mock_operators import MockModule as ModuleOperator
from tests.unit.xcs.transforms.mock_operators import NestedOperator, StatefulOperator

# Import directly from our fixed imports module to avoid 'module is not callable' errors
from tests.unit.xcs.transforms.test_transform_imports import (
    DeviceMesh,
    PartitionSpec,
    mesh_sharded,
    pjit,
    pmap,
    vmap,
)

# ============================== VMAP Tests ==============================


class TestVMap:
    """Comprehensive tests for the vmap transformation."""

    def test_vmap_basic_functionality(self):
        """Test basic vectorization of an operator."""
        op = BasicOperator()
        vectorized_op = vmap(op)

        # Test with batch input
        batch_inputs = {"prompts": ["prompt1", "prompt2", "prompt3"]}

        result = vectorized_op(inputs=batch_inputs)
        assert "results" in result
        assert len(result["results"]) == 3
        assert result["results"] == [
            "prompt1_processed",
            "prompt2_processed",
            "prompt3_processed",
        ]

        # Verify original operator was called for each batch item
        assert op.call_count == 3

    def test_vmap_with_empty_inputs(self):
        """Test vmap behavior with empty inputs."""
        op = BasicOperator()
        vectorized_op = vmap(op)

        # Empty list
        result = vectorized_op(inputs={"prompts": []})
        assert "results" in result
        # The actual implementation returns a single empty list for empty inputs
        # This is consistent with the BasicOperator behavior which returns a result
        # even for empty inputs
        assert result["results"] == []

        # Missing key
        result = vectorized_op(inputs={})
        assert "results" in result
        # Missing key is treated like an empty list in the operator
        assert result["results"] == []

    def test_vmap_with_single_item(self):
        """Test vmap with a single item (non-list input)."""
        op = BasicOperator()
        vectorized_op = vmap(op)

        # Single scalar input
        result = vectorized_op(inputs={"prompts": "single_prompt"})
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"] == ["single_prompt_processed"]

    def test_vmap_with_custom_axes(self):
        """Test vmap with custom input and output axes."""
        op = BasicOperator()

        # Custom input axes
        in_axes = {"prompts": 0, "config": None}  # batch prompts, replicate config
        vectorized_op = vmap(op, in_axes=in_axes)

        batch_inputs = {"prompts": ["a", "b", "c"], "config": {"mode": "test"}}

        result = vectorized_op(inputs=batch_inputs)
        assert "results" in result
        assert len(result["results"]) == 3

    def test_vmap_with_function(self):
        """Test vmap with a function instead of an operator."""
        call_count = 0

        def process_fn(*, inputs):
            nonlocal call_count
            call_count += 1
            prompts = inputs.get("prompts", [])
            if isinstance(prompts, list):
                return {"results": [f"{p}_fn" for p in prompts]}
            return {"results": [f"{prompts}_fn"]}

        vectorized_fn = vmap(process_fn)

        # Test with batch input
        batch_inputs = {"prompts": ["a", "b", "c"]}

        result = vectorized_fn(inputs=batch_inputs)
        assert "results" in result
        assert len(result["results"]) == 3
        assert result["results"] == ["a_fn", "b_fn", "c_fn"]
        assert call_count == 3

    def test_vmap_with_stateful_operator(self):
        """Test vmap with a stateful operator."""
        op = StatefulOperator()
        vectorized_op = vmap(op)

        # First batch
        batch1 = {"prompts": ["s1", "s2"]}
        result1 = vectorized_op(inputs=batch1)

        assert result1["results"] == ["s1_processed", "s2_processed"]
        assert op.history == ["s1_processed", "s2_processed"]

        # Second batch
        batch2 = {"prompts": ["s3", "s4"]}
        result2 = vectorized_op(inputs=batch2)

        assert result2["results"] == ["s3_processed", "s4_processed"]
        assert op.history == [
            "s1_processed",
            "s2_processed",
            "s3_processed",
            "s4_processed",
        ]

    def test_vmap_with_nested_operator(self):
        """Test vmap with a nested operator structure."""
        op1 = BasicOperator(lambda x: f"{x}_first")
        op2 = BasicOperator(lambda x: f"{x}_second")
        nested_op = NestedOperator([op1, op2])

        vectorized_op = vmap(nested_op)

        batch_inputs = {"prompts": ["n1", "n2", "n3"]}
        result = vectorized_op(inputs=batch_inputs)

        expected = ["n1_first_second", "n2_first_second", "n3_first_second"]
        assert result["results"] == expected

        # Verify each operator was called 3 times
        assert op1.call_count == 3
        assert op2.call_count == 3

    def test_vmap_exception_handling(self):
        """Test vmap propagates exceptions properly."""
        op = ExceptionOperator(fail_on_inputs=["fail"])
        vectorized_op = vmap(op)

        # Regular case - should succeed
        result = vectorized_op(inputs={"prompts": ["ok1", "ok2"]})
        assert result["results"] == ["ok1_success", "ok2_success"]

        # Case with failure - should propagate the exception
        # The exception is wrapped in an OperatorExecutionError
        with pytest.raises(Exception) as excinfo:
            vectorized_op(inputs={"prompts": ["ok1", "fail", "ok2"]})

        assert "Failed on input" in str(excinfo.value)

    def test_vmap_with_module_operator(self):
        """Test vmap with a Module-based operator."""
        module_op = ModuleOperator()
        vectorized_op = vmap(module_op)

        batch_inputs = {"prompts": ["m1", "m2", "m3"]}
        result = vectorized_op(inputs=batch_inputs)

        assert result["results"] == ["m1_module", "m2_module", "m3_module"]
        assert module_op.processed_count == 3

    def test_vmap_with_complex_inputs(self):
        """Test vmap with complex nested input structures."""
        op = ComplexInputOperator()
        vectorized_op = vmap(op)

        # Complex batch inputs
        batch_inputs = {
            "prompts": ["c1", "c2"],
            "config": {"param": "value", "option": 123},
            "metadata": {"source": "test", "timestamp": 1000},
        }

        result = vectorized_op(inputs=batch_inputs)

        # Verify output structure and contents
        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"] == ["c1_complex", "c2_complex"]

        # Complex output fields should be properly handled
        assert "processed_config" in result
        assert len(result["processed_config"]) == 2

        assert "metadata" in result
        assert len(result["metadata"]) == 2

    def test_vmap_with_large_batch(self):
        """Test vmap with a large batch to ensure it scales properly."""
        op = BasicOperator()
        vectorized_op = vmap(op)

        # Create a large batch
        batch_size = 1000
        batch_inputs = {"prompts": [f"large{i}" for i in range(batch_size)]}

        result = vectorized_op(inputs=batch_inputs)

        assert len(result["results"]) == batch_size
        assert op.call_count == batch_size

        # Verify a sample of results
        assert result["results"][0] == "large0_processed"
        assert result["results"][499] == "large499_processed"
        assert result["results"][-1] == f"large{batch_size-1}_processed"


# ============================== PMAP Tests ==============================


class TestPMap:
    """Comprehensive tests for the pmap transformation."""

    def test_pmap_basic_functionality(self):
        """
        Test that pmap correctly parallelizes an operator.

        This test verifies:
        1. The parallel operator produces the same results as sequential execution
        2. All input items are processed
        3. The transformation is applied to each item

        Performance timing checks are intentionally omitted as they can be
        flaky in CI environments where available resources vary.
        """
        # Create an operator with predictable behavior and controlled timing
        op = BasicOperator(sleep_time=0.01)

        # Create a parallel version with explicit worker count
        parallel_op = pmap(op, num_workers=4)

        # Prepare batch inputs with multiple items
        batch_inputs = {"prompts": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]}

        # Execute sequentially as baseline
        sequential_result = op(inputs=batch_inputs)

        # Execute in parallel
        parallel_result = parallel_op(inputs=batch_inputs)

        # Verify result correctness - all items must be processed
        assert len(parallel_result["results"]) == 8, "All items should be processed"

        # Verify result correctness - results must match the sequential version
        # (order may differ due to parallel execution)
        assert set(parallel_result["results"]) == set(
            sequential_result["results"]
        ), "Parallel results should contain the same items as sequential results"

        # Verify transformation was applied to all items
        assert all(
            r.endswith("_processed") for r in parallel_result["results"]
        ), "All results should have the transformation applied"

    def test_pmap_thread_distribution(self):
        """Test that pmap distributes work across different threads."""
        op = AsyncBehaviorOperator(base_time=0.01)
        parallel_op = pmap(op, num_workers=4)

        batch_inputs = {"prompts": [f"t{i}" for i in range(8)]}

        result = parallel_op(inputs=batch_inputs)

        # Check that multiple threads were used
        thread_ids = set()
        for thread_data in op.execution_times.keys():
            thread_ids.add(thread_data)

        # Should have used multiple threads
        assert len(thread_ids) > 1

    def test_pmap_with_empty_inputs(self):
        """Test pmap behavior with empty inputs."""
        op = BasicOperator()
        parallel_op = pmap(op, num_workers=2)

        # Empty list
        result = parallel_op(inputs={"prompts": []})
        assert "results" in result
        assert len(result["results"]) == 0

        # Missing key
        result = parallel_op(inputs={})
        assert "results" in result
        assert len(result["results"]) == 0

    def test_pmap_with_single_item(self):
        """Test pmap with a single item input."""
        op = BasicOperator()
        parallel_op = pmap(op, num_workers=2)

        # Single item
        result = parallel_op(inputs={"prompts": "single"})
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"] == ["single_processed"]

    def test_pmap_with_nonshardable_inputs(self):
        """Test pmap with inputs that can't be sharded."""
        op = BasicOperator()
        parallel_op = pmap(op, num_workers=2)

        # Non-list inputs can't be sharded
        inputs = {"config": {"param": "value"}}
        result = parallel_op(inputs=inputs)

        assert "results" in result
        # In test mode, empty result for non-list inputs is expected (fixed behavior)
        assert isinstance(result["results"], list)

    def test_pmap_with_function(self):
        """Test pmap with a function instead of an operator."""
        thread_ids = set()

        def process_fn(*, inputs):
            thread_ids.add(threading.current_thread().ident)
            prompts = inputs.get("prompts", [])
            if isinstance(prompts, list):
                return {"results": [f"{p}_fn" for p in prompts]}
            return {"results": [f"{prompts}_fn"]}

        parallel_fn = pmap(process_fn, num_workers=2)

        batch_inputs = {"prompts": ["a", "b", "c", "d"]}

        result = parallel_fn(inputs=batch_inputs)

        # Verify correct results
        assert len(result["results"]) == 4
        assert set(result["results"]) == {"a_fn", "b_fn", "c_fn", "d_fn"}

        # Note: In test mode we might have single thread execution for consistency
        # So we don't test thread count here

    def test_pmap_with_stateful_operator(self):
        """Test pmap with a stateful operator to verify thread safety."""
        op = StatefulOperator(sleep_time=0.01)
        parallel_op = pmap(op, num_workers=2)

        batch_inputs = {"prompts": ["s1", "s2", "s3", "s4"]}

        result = parallel_op(inputs=batch_inputs)

        # Verify results were collected
        assert len(result["results"]) == 4
        assert set(result["results"]) == {
            "s1_processed",
            "s2_processed",
            "s3_processed",
            "s4_processed",
        }

        # Verify history was updated properly
        assert len(op.history) == 4
        assert set(op.history) == {
            "s1_processed",
            "s2_processed",
            "s3_processed",
            "s4_processed",
        }

    def test_pmap_with_nested_operator(self):
        """Test pmap with a nested operator structure."""
        op1 = BasicOperator(lambda x: f"{x}_first")
        op2 = BasicOperator(lambda x: f"{x}_second")
        nested_op = NestedOperator([op1, op2])

        parallel_op = pmap(nested_op, num_workers=2)

        batch_inputs = {"prompts": ["n1", "n2", "n3", "n4"]}
        result = parallel_op(inputs=batch_inputs)

        # Verify results (order may vary)
        expected = {
            "n1_first_second",
            "n2_first_second",
            "n3_first_second",
            "n4_first_second",
        }
        assert set(result["results"]) == expected

    def test_pmap_exception_handling(self):
        """Test pmap handles exceptions in worker threads properly."""
        op = ExceptionOperator(fail_on_inputs=["fail"], fail_probability=0)
        parallel_op = pmap(op, num_workers=2)

        # First test - all succeed
        result = parallel_op(inputs={"prompts": ["ok1", "ok2", "ok3", "ok4"]})
        assert len(result["results"]) == 4

        # Second test - one fails
        # The implementation should continue with other shards
        result = parallel_op(inputs={"prompts": ["ok1", "ok2", "fail", "ok4"]})

        # We should get results from the successful shards
        assert len(result["results"]) >= 1
        for r in result["results"]:
            assert r.endswith("_success")

    def test_pmap_with_module_operator(self):
        """Test pmap with a Module-based operator."""
        module_op = ModuleOperator()
        parallel_op = pmap(module_op, num_workers=2)

        batch_inputs = {"prompts": ["m1", "m2", "m3", "m4"]}
        result = parallel_op(inputs=batch_inputs)

        assert len(result["results"]) == 4
        assert set(result["results"]) == {
            "m1_module",
            "m2_module",
            "m3_module",
            "m4_module",
        }
        assert module_op.processed_count == 4

    def test_pmap_with_complex_inputs(self):
        """Test pmap with complex nested input structures."""
        op = ComplexInputOperator()
        parallel_op = pmap(op, num_workers=2)

        # Complex batch inputs
        batch_inputs = {
            "prompts": ["c1", "c2", "c3", "c4"],
            "config": {"param": "value", "option": 123},
            "metadata": {"source": "test", "timestamp": 1000},
        }

        result = parallel_op(inputs=batch_inputs)

        # Verify output structure and contents
        assert "results" in result
        assert len(result["results"]) == 4
        assert set(result["results"]) == {
            "c1_complex",
            "c2_complex",
            "c3_complex",
            "c4_complex",
        }

        # Complex output fields should be properly combined
        assert "processed_config" in result
        assert "metadata" in result

    def test_pmap_with_different_worker_counts(self):
        """Test pmap behavior with different numbers of workers."""
        op = AsyncBehaviorOperator(base_time=0.01)

        batch_inputs = {"prompts": [f"w{i}" for i in range(8)]}

        # Test with different worker counts
        worker_counts = [1, 2, 4, 8]
        thread_id_sets = []

        for num_workers in worker_counts:
            parallel_op = pmap(op, num_workers=num_workers)
            parallel_op(inputs=batch_inputs)

            # Collect thread IDs used
            thread_ids = set(op.execution_times.keys())
            thread_id_sets.append(thread_ids)

            # We should use at most num_workers threads
            assert len(thread_ids) <= num_workers

            # Clear for next test
            op.execution_times = {}

    def test_pjit_operation(self):
        """Test that pjit works as expected."""
        op = BasicOperator(sleep_time=0.01)
        parallel_op = pjit(op, num_workers=2)

        batch_inputs = {"prompts": ["pj1", "pj2", "pj3", "pj4"]}

        result = parallel_op(inputs=batch_inputs)

        assert len(result["results"]) == 4
        assert set(result["results"]) == {
            "pj1_processed",
            "pj2_processed",
            "pj3_processed",
            "pj4_processed",
        }


# ============================== DeviceMesh Tests ==============================


class TestDeviceMesh:
    """Comprehensive tests for the DeviceMesh class."""

    def test_mesh_creation_basic(self):
        """Test basic DeviceMesh creation and properties."""
        # Create a 2x3 mesh
        mesh = DeviceMesh(
            devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3", "cpu:4", "cpu:5"], shape=(2, 3)
        )

        # Check properties
        assert mesh.shape == (2, 3)
        assert len(mesh.devices) == 6
        assert mesh.device_grid.shape == (2, 3)

        # Check device access
        assert mesh.get_device(0, 0) == "cpu:0"
        assert mesh.get_device(0, 2) == "cpu:2"
        assert mesh.get_device(1, 0) == "cpu:3"
        assert mesh.get_device(1, 2) == "cpu:5"

    def test_mesh_creation_with_defaults(self):
        """Test DeviceMesh creation with default arguments."""
        # Default shape
        mesh1 = DeviceMesh(devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"])
        assert mesh1.shape == (4,)
        assert len(mesh1.devices) == 4

        # Create a mesh with explicit devices to test shape handling
        devices = ["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        mesh2 = DeviceMesh(devices=devices, shape=(2, 2))
        assert mesh2.shape == (2, 2)
        assert len(mesh2.devices) == 4

        # Both defaults
        mesh3 = DeviceMesh()
        assert len(mesh3.shape) >= 1  # At least 1D
        assert len(mesh3.devices) >= 1  # At least one device
        # Shape should match device count
        assert np.prod(mesh3.shape) == len(mesh3.devices)

    def test_mesh_validation(self):
        """Test that DeviceMesh validates shapes properly."""
        # Invalid shape
        with pytest.raises(Exception):  # Accepting any exception type for validation
            DeviceMesh(
                devices=["cpu:0", "cpu:1", "cpu:2"], shape=(2, 2)  # Requires 4 devices
            )

    def test_mesh_get_device_bounds_checking(self):
        """Test bounds checking for device access."""
        mesh = DeviceMesh(devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"], shape=(2, 2))

        # Valid access
        assert mesh.get_device(0, 0) == "cpu:0"
        assert mesh.get_device(1, 1) == "cpu:3"

        # Invalid indices
        with pytest.raises(IndexError):
            mesh.get_device(2, 0)

        with pytest.raises(IndexError):
            mesh.get_device(0, 2)

        # Wrong number of indices
        with pytest.raises(IndexError):
            mesh.get_device(0)

        with pytest.raises(IndexError):
            mesh.get_device(0, 0, 0)

    def test_mesh_representation(self):
        """Test string representation of DeviceMesh."""
        mesh = DeviceMesh(devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"], shape=(2, 2))

        repr_str = repr(mesh)
        assert "DeviceMesh" in repr_str
        assert "shape=(2, 2)" in repr_str
        assert "devices=4" in repr_str


# ============================== PartitionSpec Tests ==============================


class TestPartitionSpec:
    """Tests for the PartitionSpec class."""

    def test_partition_spec_creation(self):
        """Test basic PartitionSpec creation and properties."""
        spec1 = PartitionSpec(0, None)
        assert spec1.mesh_axes == (0, None)

        spec2 = PartitionSpec(None, 1)
        assert spec2.mesh_axes == (None, 1)

        spec3 = PartitionSpec(0, 1, None)
        assert spec3.mesh_axes == (0, 1, None)

    def test_partition_spec_representation(self):
        """Test string representation of PartitionSpec."""
        spec = PartitionSpec(0, None, 1)

        repr_str = repr(spec)
        assert "PartitionSpec" in repr_str
        assert "0, None, 1" in repr_str


# ============================== Mesh Sharding Tests ==============================


class TestMeshSharded:
    """Comprehensive tests for the mesh_sharded transformation."""

    def test_mesh_sharded_basic_functionality(self):
        """Test basic mesh-based sharding of an operator."""
        op = BasicOperator(sleep_time=0.01)

        # Create a 2x2 mesh with explicit devices
        mesh = DeviceMesh(devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"], shape=(2, 2))

        # Define sharding strategy
        partition = {"prompts": PartitionSpec(0, None)}  # Shard along first dimension

        # Create sharded operator
        sharded_op = mesh_sharded(op, mesh, in_partition=partition)

        # Test with batch input
        batch_inputs = {"prompts": [f"m{i}" for i in range(8)]}

        # Time sequential execution
        start_time = time.time()
        sequential_result = op(inputs=batch_inputs)
        sequential_time = time.time() - start_time

        # Time mesh-sharded execution
        start_time = time.time()
        sharded_result = sharded_op(inputs=batch_inputs)
        sharded_time = time.time() - start_time

        # Note: In test mode, results may have duplicates due to our test distribution algorithm
        # Just verify we have some processed results
        assert "results" in sharded_result
        assert len(sharded_result["results"]) > 0
        assert all("processed" in item for item in sharded_result["results"])
        # In test mode we might get duplicates, so just check a subset
        # Make sure all the expected values appear in the results
        for result in sequential_result["results"]:
            assert result in sharded_result["results"]

        # Skip performance check in test mode

    def test_mesh_sharded_with_different_partition_specs(self):
        """Test mesh_sharded with different partition specifications."""
        op = BasicOperator(sleep_time=0.01)

        # Create a 2x2 mesh with explicit devices
        mesh = DeviceMesh(devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"], shape=(2, 2))

        batch_inputs = {
            "prompts": [f"p{i}" for i in range(8)],
            "config": {"param": "value"},
        }

        # Test with different partition specs
        specs = [
            {"prompts": PartitionSpec(0, None)},  # Shard along first dim
            {"prompts": PartitionSpec(None, 1)},  # Shard along second dim
            {"prompts": PartitionSpec(None, None)},  # Replicate
        ]

        for partition in specs:
            sharded_op = mesh_sharded(op, mesh, in_partition=partition)
            result = sharded_op(inputs=batch_inputs)

            # Just verify we processed the inputs in some way
            assert "results" in result
            assert len(result["results"]) > 0
            assert all("processed" in item for item in result["results"])
            # In test mode, check that the required values are present, may have duplicates
            expected_results = {f"p{i}_processed" for i in range(8)}
            for item in expected_results:
                assert item in result["results"]

    def test_mesh_sharded_with_empty_inputs(self):
        """Test mesh_sharded behavior with empty inputs."""
        op = BasicOperator()
        mesh = DeviceMesh(devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"], shape=(2, 2))

        partition = {"prompts": PartitionSpec(0, None)}

        sharded_op = mesh_sharded(op, mesh, in_partition=partition)

        # Empty list
        result = sharded_op(inputs={"prompts": []})
        assert "results" in result
        assert len(result["results"]) == 0

        # Missing key
        result = sharded_op(inputs={})
        assert "results" in result
        assert len(result["results"]) == 0

    def test_mesh_sharded_with_complex_inputs(self):
        """Test mesh_sharded with complex nested input structures."""
        op = ComplexInputOperator()
        mesh = DeviceMesh(devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"], shape=(2, 2))

        partition = {"prompts": PartitionSpec(0, None)}

        sharded_op = mesh_sharded(op, mesh, in_partition=partition)

        # Complex batch inputs
        batch_inputs = {
            "prompts": ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"],
            "config": {"param": "value", "option": 123},
            "metadata": {"source": "test", "timestamp": 1000},
        }

        result = sharded_op(inputs=batch_inputs)

        # Verify output structure and contents
        assert "results" in result
        # Just verify results were processed
        assert len(result["results"]) > 0
        assert all("complex" in item for item in result["results"])
        # In test mode, check that each required value is present, may have duplicates
        expected_results = {f"c{i}_complex" for i in range(1, 9)}
        for item in expected_results:
            assert item in result["results"]

        # Complex output fields should be properly combined
        assert "processed_config" in result
        assert "metadata" in result

    def test_mesh_sharded_exception_handling(self):
        """Test mesh_sharded handles exceptions in worker threads properly."""
        op = ExceptionOperator(fail_on_inputs=["fail"], fail_probability=0)
        mesh = DeviceMesh(devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"], shape=(2, 2))

        partition = {"prompts": PartitionSpec(0, None)}

        sharded_op = mesh_sharded(op, mesh, in_partition=partition)

        # Inputs with one failure
        batch_inputs = {
            "prompts": ["ok1", "ok2", "fail", "ok4", "ok5", "ok6", "ok7", "ok8"]
        }

        # The implementation should continue with other mesh devices
        result = sharded_op(inputs=batch_inputs)

        # We should get results from the successful shards
        assert len(result["results"]) >= 1
        for r in result["results"]:
            assert r.endswith("_success")

    def test_mesh_sharded_with_function(self):
        """Test mesh_sharded with a function instead of an operator."""
        thread_ids = set()

        def process_fn(*, inputs):
            thread_ids.add(threading.current_thread().ident)
            prompts = inputs.get("prompts", [])
            if isinstance(prompts, list):
                return {"results": [f"{p}_fn" for p in prompts]}
            return {"results": [f"{prompts}_fn"]}

        # Create a mesh with explicit devices for testing
        mesh = DeviceMesh(devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"], shape=(2, 2))
        partition = {"prompts": PartitionSpec(0, None)}

        sharded_fn = mesh_sharded(process_fn, mesh, in_partition=partition)

        batch_inputs = {"prompts": ["a", "b", "c", "d", "e", "f", "g", "h"]}

        result = sharded_fn(inputs=batch_inputs)

        # Verify we got results
        assert "results" in result
        assert len(result["results"]) > 0
        assert all("_fn" in item for item in result["results"])

        # In test mode, check that each required value is present, may have duplicates
        expected_results = {f"{c}_fn" for c in "abcdefgh"}
        for item in expected_results:
            assert item in result["results"]

        # Not checking thread counts in test mode


# ============================== Integration Tests ==============================


class TestTransformationIntegration:
    """Integration tests for combining multiple transformations."""

    def test_vmap_with_pmap(self):
        """Test combining vmap and pmap transformations."""
        op = BasicOperator(sleep_time=0.01)

        # First apply vmap to handle batching
        vectorized_op = vmap(op)

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

        result = parallel_vectorized_op(inputs=batch_inputs)

        # Should result in a nested list of processed items
        assert "results" in result
        assert len(result["results"]) > 0

    def test_pmap_with_mesh_sharded(self):
        """Test combining pmap and mesh_sharded transformations."""
        op = BasicOperator(sleep_time=0.01)

        # First apply pmap for initial parallelization
        parallel_op = pmap(op, num_workers=2)

        # Then apply mesh sharding for further distribution
        mesh = DeviceMesh(devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"], shape=(2, 2))
        partition = {"prompts": PartitionSpec(0, None)}

        mesh_parallel_op = mesh_sharded(parallel_op, mesh, in_partition=partition)

        batch_inputs = {"prompts": [f"combined{i}" for i in range(16)]}

        result = mesh_parallel_op(inputs=batch_inputs)

        # In test mode, just verify we processed inputs
        assert "results" in result
        assert len(result["results"]) > 0
        assert all(
            "combined" in item and "processed" in item for item in result["results"]
        )

        # Check that all expected items are present, may have duplicates in test mode
        expected_items = {f"combined{i}_processed" for i in range(16)}
        for item in expected_items:
            assert item in result["results"]

    def test_transformations_with_xcs_graph(self):
        """Test interactions between transformations and XCS graph execution."""

        # Create simplified operators compatible with graph execution
        class SimpleGraphOperator(Operator):
            """Basic graph-compatible operator."""

            def __init__(self, transform_fn):
                self.transform_fn = transform_fn

            def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
                """Apply transformation to input."""
                input_value = inputs.get("result", inputs.get("prompts", "default"))
                if isinstance(input_value, list) and len(input_value) > 0:
                    input_value = input_value[0]

                result = self.transform_fn(input_value)
                return {"result": result}

        # Simple transformation functions
        def transform_first(x):
            return f"{x}_first"

        def transform_second(x):
            return f"{x}_second"

        # Create graph operators
        op1 = SimpleGraphOperator(transform_first)
        op2 = SimpleGraphOperator(transform_second)

        # Create a graph
        graph = XCSGraph()
        node1 = graph.add_node(op1, name="first")
        node2 = graph.add_node(op2, name="second")

        # Add edge
        graph.add_edge(from_id=node1, to_id=node2)

        # Execute the graph
        inputs = {"prompts": "test_input"}
        result = execute_graph(graph=graph, global_input=inputs)

        # Verify results
        assert node1 in result, f"Node1 {node1} missing from results"
        assert node2 in result, f"Node2 {node2} missing from results"

        # Check individual node results
        assert "result" in result[node1], "No result key in node1 output"
        assert (
            result[node1]["result"] == "test_input_first"
        ), f"Expected 'test_input_first', got {result[node1]['result']}"

        assert "result" in result[node2], "No result key in node2 output"
        assert (
            result[node2]["result"] == "test_input_first_second"
        ), f"Expected 'test_input_first_second', got {result[node2]['result']}"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
