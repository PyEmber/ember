"""Unit tests for TracerContext functionality.

This module tests that TracerContext correctly patches operators, builds an IRGraph,
captures trace records, and restores the original operator methods.
"""

from typing import Any, Dict

import pytest
from pydantic import BaseModel

from ember.xcs.tracer.tracer_decorator import jit
from ember.xcs.tracer.xcs_tracing import TracerContext
from tests.helpers.stub_classes import Operator


class MockInput(BaseModel):
    """Input model for the mock operator."""

    value: int


@jit()
class MockOperator(Operator[MockInput, Dict[str, Any]]):
    """A mock operator that doubles the input value."""

    # For testing, we use a simplified specification.
    specification = type(
        "DummySpecification",
        (),
        {
            "input_model": MockInput,
            "validate_inputs": lambda self, *, inputs: inputs,
            "validate_output": lambda self, *, output: output,
            "render_prompt": lambda self, *, inputs: "dummy prompt",
        },
    )()

    def forward(self, *, inputs: Any) -> Dict[str, Any]:
        # Allow inputs to be passed as either a dict or a MockInput instance.
        if isinstance(inputs, dict):
            inputs = MockInput(**inputs)
        return {"result": inputs.value * 2}


def test_tracer_context_basic() -> None:
    """Tests basic tracing with TracerContext."""
    # Create a new instance of the operator
    operator = MockOperator()
    sample_input = {"value": 5}

    # Force trace mode to ensure we get trace records regardless of caching
    operator._force_trace = True

    with TracerContext() as tracer:
        result = operator(inputs=sample_input)

    # Verify we have at least one trace record
    assert len(tracer.records) >= 1, "Expected at least one trace record."

    # Verify the content of the first trace record
    first_record = tracer.records[0]
    assert first_record.outputs == {
        "result": 10
    }, f"Traced output {first_record.outputs} does not match expected {{'result': 10}}."


def test_tracer_context_patch_restore() -> None:
    """Tests that operator patching is no longer performed (or is preserved) in the new design."""
    operator = MockOperator()
    original_call = operator.__class__.__call__
    with TracerContext() as _:
        pass
    assert (
        operator.__class__.__call__ == original_call
    ), "Operator __call__ should remain unchanged in the new implementation."
