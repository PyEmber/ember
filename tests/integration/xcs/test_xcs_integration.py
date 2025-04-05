"""XCS Integration tests.

Tests that verify the complete XCS tracing and execution pipeline
with various transforms and graph patterns. Uses minimal test doubles.
"""

from typing import Any, Callable, Dict, List

import numpy as np
import pytest

# Import minimal test doubles instead of actual implementations
from tests.helpers.xcs_minimal_doubles import (
    minimal_autograph,
    minimal_jit,
    minimal_pmap,
    minimal_vmap,
)

# Mark all tests as integration tests
pytestmark = [pytest.mark.integration]


def test_basic_autograph_integration():
    """Test that autograph can trace and execute a simple function."""

    @minimal_autograph
    def add_and_square(x, y):
        return (x + y) ** 2

    # Test regular execution
    result = add_and_square(3, 4)
    assert result == 49

    # Verify the graph structure
    graph = add_and_square.get_graph()
    assert graph is not None


def test_jit_integration():
    """Test that jit works with real computation."""

    @minimal_jit
    def complex_math(x, y, z):
        a = x * y
        b = a + z
        c = b**2
        return c - a

    # Test with different input types
    result1 = complex_math(2, 3, 4)
    # ((2*3)+4)^2 - (2*3) = 10^2 - 6 = 94
    assert result1 == 94

    # Test with numpy arrays
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    z = np.array([7, 8, 9])
    result2 = complex_math(x, y, z)

    # Verify element-wise operation
    expected = ((x * y) + z) ** 2 - (x * y)
    np.testing.assert_array_equal(result2, expected)


def test_vmap_integration():
    """Test vectorized mapping over a real function."""

    def square(x):
        """Square a number."""
        return x * x

    # Create a vectorized version
    vmap_square = minimal_vmap(square)

    # Test with list input
    inputs = [1, 2, 3, 4, 5]
    result = vmap_square(inputs)

    # Verify results
    expected = [1, 4, 9, 16, 25]
    assert result == expected

    # Test with numpy array
    np_inputs = np.array([1, 2, 3, 4, 5])
    np_result = vmap_square(np_inputs)

    # Verify results
    np_expected = np.array([1, 4, 9, 16, 25])
    np.testing.assert_array_equal(np_result, np_expected)


def test_pmap_integration():
    """Test parallel mapping over CPU cores."""

    def square(x):
        """Compute the square of a number."""
        return x * x

    # Create a parallel version
    pmap_square = minimal_pmap(square)

    # Generate test data
    data = list(range(10))

    # Execute the parallel function
    result = pmap_square(data)

    # Verify results
    expected = [x * x for x in data]
    assert result == expected


def test_complex_graph_execution():
    """Test execution of a complex graph with multiple operations."""

    @minimal_autograph
    def complex_workflow(data_list, scale_factor=2.0):
        # Map a function over the data
        squared = [x**2 for x in data_list]

        # Filter based on a condition
        filtered = [x for x in squared if x > 10]

        # Apply another transformation
        scaled = [x * scale_factor for x in filtered]

        # Reduce to a single result
        if scaled:
            return sum(scaled) / len(scaled)
        return 0

    # Test with sample data
    data = [1, 2, 3, 4, 5]
    result = complex_workflow(data)

    # Compute expected result
    squared = [x**2 for x in data]  # [1, 4, 9, 16, 25]
    filtered = [x for x in squared if x > 10]  # [16, 25]
    scaled = [x * 2.0 for x in filtered]  # [32.0, 50.0]
    expected = sum(scaled) / len(scaled)  # 82.0 / 2 = 41.0

    assert result == expected

    # Verify graph exists
    graph = complex_workflow.get_graph()
    assert graph is not None


def test_function_composition():
    """Test composition of functions with minimal test doubles."""

    # Define simple functions
    @minimal_jit
    def double(x):
        return x * 2

    @minimal_jit
    def add_10(x):
        return x + 10

    @minimal_jit
    def square(x):
        return x * x

    # Compose functions manually
    def composed(x):
        a = double(x)
        b = add_10(a)
        return square(b)

    # Test the composed function
    result = composed(5)
    # (5*2) + 10 = 20, then 20^2 = 400
    assert result == 400
