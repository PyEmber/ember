"""Example of using minimal test doubles instead of complex mocks.

This module demonstrates how to use minimal test doubles for testing XCS components
without duplicating the entire implementation. This follows the CLAUDE.md guidelines
of "avoid overmocking" and using "minimal test doubles".
"""

from typing import Any, Dict, List

from tests.helpers.operator_minimal_doubles import (
    MinimalOperator,
    MinimalTestModel,
    SimpleDeterministicOperator,
)

# Import minimal test doubles instead of complex mock implementations
from tests.helpers.xcs_minimal_doubles import minimal_autograph, minimal_jit


# Test using minimal JIT implementation
def test_minimal_jit():
    """Test that minimal JIT implementation works correctly."""

    @minimal_jit
    def double_value(x):
        return x * 2

    # Test basic functionality
    result = double_value(5)
    assert result == 10

    # Verify graph attribute exists for compatibility
    assert hasattr(double_value, "graph")
    assert hasattr(double_value, "get_graph")


# Test using minimal autograph implementation
def test_minimal_autograph():
    """Test that minimal autograph implementation works correctly."""

    @minimal_autograph
    def process_list(numbers):
        result = []
        for num in numbers:
            result.append(num * 2)
        return sum(result)

    # Test basic functionality
    result = process_list([1, 2, 3])
    assert result == 12  # (1*2 + 2*2 + 3*2)

    # Verify graph attribute exists for compatibility
    assert hasattr(process_list, "graph")
    assert hasattr(process_list, "get_graph")


# Test using minimal operator implementations
def test_minimal_operator():
    """Test that minimal operator implementation works correctly."""

    # Create a simple operator that doubles its input
    doubler = SimpleDeterministicOperator(transform_fn=lambda x: x * 2)

    # Create an input model
    input_model = MinimalTestModel(value=5)

    # Execute the operator
    result = doubler(inputs=input_model)

    # Verify result
    assert result.value == 10


# Test composition of operators with minimal implementations
def test_operator_composition():
    """Test composing operators using minimal implementations."""

    # Create operators that transform values
    doubler = SimpleDeterministicOperator(transform_fn=lambda x: x * 2)
    adder = SimpleDeterministicOperator(transform_fn=lambda x: x + 3)

    # Manual composition (without requiring complex Sequential implementation)
    def composed_operation(input_value):
        # Double then add
        intermediate = doubler(inputs=MinimalTestModel(value=input_value))
        final = adder(inputs=intermediate)
        return final.value

    # Test the composition
    result = composed_operation(5)
    assert result == 13  # (5 * 2) + 3

    # Verify the operations work in reverse order too
    def reverse_composed(input_value):
        # Add then double
        intermediate = adder(inputs=MinimalTestModel(value=input_value))
        final = doubler(inputs=intermediate)
        return final.value

    result = reverse_composed(5)
    assert result == 16  # (5 + 3) * 2


# Test with jit-decorated operator
def test_jit_decorated_operator():
    """Test using JIT with minimal operator implementations."""

    # Create a JIT-decorated operator
    @minimal_jit
    class JitOperator(MinimalOperator[MinimalTestModel, MinimalTestModel]):
        def forward(self, *, inputs: MinimalTestModel) -> MinimalTestModel:
            return MinimalTestModel(value=inputs.value * 3)

    # Create and use the operator
    tripler = JitOperator()
    result = tripler(inputs=MinimalTestModel(value=5))

    # Verify result
    assert result.value == 15
