"""Unit tests for the JIT tracer decorator functionality.

This module verifies that an operator decorated with the JIT decorator correctly caches its execution plan,
and that forced tracing bypasses caching, causing the operator's forward method to be executed on each call.
"""

from typing import Any, Dict, Type

import pytest
from pydantic import BaseModel

from ember.xcs.tracer.tracer_decorator import jit
from tests.helpers.stub_classes import Operator

# ----------------------------------------------------------------------------
# Dummy Models and Specification for Testing
# ----------------------------------------------------------------------------


class DummyInput(BaseModel):
    """Input model for testing the operators.

    Attributes:
        x (int): An integer value representing the input.
    """

    x: int


class DummyOutput(BaseModel):
    """Output model for testing the operators.

    Attributes:
        y (int): An integer value representing the output.
    """

    y: int


class DummySpecification:
    """A dummy specification providing minimal validation and prompt rendering.

    This class simulates the behavior of an operator specification, enforcing input validation,
    output validation, and prompt rendering using a simple dummy implementation.
    """

    def __init__(self) -> None:
        self.input_model: Type[BaseModel] = DummyInput

    def validate_inputs(self, *, inputs: Any) -> Any:
        """Validates the provided inputs.

        Args:
            inputs (Any): The inputs to validate.

        Returns:
            Any: The validated inputs (unchanged in this dummy implementation).
        """
        return inputs

    def validate_output(self, *, output: Any) -> Any:
        """Validates the operator output.

        Args:
            output (Any): The output to validate.

        Returns:
            Any: The validated output (unchanged in this dummy implementation).
        """
        return output

    def render_prompt(self, *, inputs: Dict[str, Any]) -> str:
        """Renders a prompt based on the provided inputs.

        Args:
            inputs (Dict[str, Any]): The inputs to render the prompt for.

        Returns:
            str: A dummy prompt string.
        """
        return "dummy prompt"


# ----------------------------------------------------------------------------
# Dummy Operator decorated with JIT
# ----------------------------------------------------------------------------


@jit()
class DummyOperator(Operator[DummyInput, DummyOutput]):
    """Dummy operator that increments an internal counter upon execution.

    This operator demonstrates caching of its execution plan. When invoked with the same input,
    the forward method is only executed once, with subsequent calls using the cached plan.
    """

    specification: DummySpecification = DummySpecification()

    def __init__(self) -> None:
        """Initializes the DummyOperator with a counter starting at zero."""
        self.counter: int = 0

    def forward(self, *, inputs: DummyInput) -> DummyOutput:
        """Executes the operator's logic by incrementing an internal counter.

        Args:
            inputs (DummyInput): The input data for the operator.

        Returns:
            DummyOutput: The output model containing the updated counter value.
        """
        self.counter += 1
        return DummyOutput(y=self.counter)


def test_jit_decorator_execution() -> None:
    """Tests that the JIT-decorated operator executes its forward method on each call."""
    operator_instance: DummyOperator = DummyOperator()

    # First call
    output_first: DummyOutput = operator_instance(inputs={"x": 5})
    assert output_first.y == 1, "Expected first counter value to be 1"

    # Second call with different input
    output_second: DummyOutput = operator_instance(inputs={"x": 6})
    assert output_second.y == 2, "Expected second counter value to be 2"

    # Third call with first input again
    output_third: DummyOutput = operator_instance(inputs={"x": 5})
    assert output_third.y == 3, "Expected third counter value to be 3"

    # Verify final counter state
    assert (
        operator_instance.counter == 3
    ), f"Expected counter to be 3, got {operator_instance.counter}"


# ----------------------------------------------------------------------------
# Dummy Operator with Forced Tracing Enabled
# ----------------------------------------------------------------------------


@jit(force_trace=True)
class ForceTraceOperator(DummyOperator):
    """Operator that is forced to create a trace record on every call."""

    pass


def test_jit_decorator_force_trace() -> None:
    """Tests that the JIT decorator with force_trace=True bypasses caching."""
    operator_instance: ForceTraceOperator = ForceTraceOperator()
    output_first: DummyOutput = operator_instance(inputs={"x": 10})
    output_second: DummyOutput = operator_instance(inputs={"x": 10})
    # With force_trace=True, the counter increments on each invocation.
    assert (
        operator_instance.counter == 2
    ), f"Expected counter to be 2, but got {operator_instance.counter}"
    assert (
        output_first != output_second
    ), "Expected distinct output due to forced trace."

    # Third call with same input should still increment the counter
    output_third: DummyOutput = operator_instance(inputs={"x": 10})
    assert (
        operator_instance.counter == 3
    ), f"Expected counter to be 3 with force_trace, but got {operator_instance.counter}"
