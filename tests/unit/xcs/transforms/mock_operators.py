"""
Mock operators for testing XCS transformations.

These simplified mock operators don't use actual validation or complex logic,
making them suitable for testing the transform functions.
"""

import dataclasses
import random
import threading
import time
from typing import Any, Dict, List, Optional, Protocol


# Define our own simplified base classes to avoid import errors
class Specification:
    """Simplified specification that doesn't perform validation."""

    def __init__(self, input_model=None, structured_output=None):
        self.input_model = input_model
        self.structured_output = structured_output

    def validate_inputs(self, *, inputs):
        """No-op input validation."""
        return inputs

    def validate_output(self, *, output):
        """No-op output validation."""
        return output


class Operator:
    """Base class for all operators."""

    specification = None

    def __init__(self):
        """Initialize the operator."""
        pass

    def __call__(self, *, inputs):
        """Call the operator's forward method."""
        return self.forward(inputs=inputs)

    def forward(self, *, inputs):
        """Process inputs to produce outputs. Must be implemented by subclasses."""
        raise NotImplementedError()


class EmberModule:
    """Simplified EmberModule stub."""

    pass


# Create a simple no-op specification for testing
class MockSpecification(Specification):
    """Mock specification class that doesn't perform validation."""

    def __init__(self):
        """Initialize with empty models."""
        pass

    def validate_inputs(self, *, inputs):
        """No-op input validation."""
        return inputs

    def validate_output(self, *, output):
        """No-op output validation."""
        return output


# Create a base testing operator class
class MockOperator(Operator):
    """Base mock operator for testing that satisfies Operator requirements."""

    # Use the mock specification to avoid validation
    specification = MockSpecification()

    def forward(self, *, inputs):
        """Default implementation that subclasses should override."""
        return inputs


class BasicOperator(MockOperator):
    """Simple operator that processes inputs with a customizable transformation."""

    def __init__(self, transform_fn=None, sleep_time=0):
        super().__init__()
        self.transform_fn = transform_fn or (lambda x: f"{x}_processed")
        self.sleep_time = sleep_time
        self.call_count = 0
        self.last_thread = None

    def forward(self, *, inputs):
        """Process inputs using the transform function."""
        self.call_count += 1
        self.last_thread = threading.current_thread().ident

        if self.sleep_time > 0:
            time.sleep(self.sleep_time)

        # Handle non-standard inputs (like config-only inputs)
        if "config" in inputs and "prompts" not in inputs:
            return {"results": ["config_processed"]}

        prompts = inputs.get("prompts", [])

        if isinstance(prompts, list):
            results = [self.transform_fn(prompt) for prompt in prompts]
        else:
            results = [self.transform_fn(prompts)]

        return {"results": results}

    def reset_call_count(self):
        """Reset the call counter for clean tests."""
        self.call_count = 0


class StatefulOperator(BasicOperator):
    """Operator that maintains state between calls."""

    def __init__(self, transform_fn=None, sleep_time=0):
        super().__init__(transform_fn, sleep_time)
        self.history = []

    def forward(self, *, inputs):
        """Process inputs and update internal state."""
        result = super().forward(inputs=inputs)
        self.history.extend(result["results"])
        return result


class NestedOperator(MockOperator):
    """Operator that composes multiple other operators."""

    def __init__(self, operators):
        super().__init__()
        self.operators = operators

    def forward(self, *, inputs):
        """Apply each operator in sequence."""
        prompts = inputs.get("prompts", [])

        if isinstance(prompts, list):
            # Process each prompt through the nested pipeline
            results = []
            for prompt in prompts:
                # Process through each operator
                current = prompt
                for op in self.operators:
                    # Wrap in the expected dict structure
                    result = op(inputs={"prompts": current})
                    # Extract the result (should be a 1-item list)
                    if "results" in result and result["results"]:
                        current = result["results"][0]
                    else:
                        current = current  # No change
                results.append(current)

            return {"results": results}
        else:
            # Single input
            current = prompts
            for op in self.operators:
                result = op(inputs={"prompts": current})
                if "results" in result and result["results"]:
                    current = result["results"][0]
                else:
                    current = current

            return {"results": [current]}


class ExceptionOperator(MockOperator):
    """Operator that raises exceptions under specified conditions."""

    def __init__(
        self, fail_on_inputs=None, exception_type=ValueError, fail_probability=0.0
    ):
        super().__init__()
        self.fail_on_inputs = fail_on_inputs or []
        self.exception_type = exception_type
        self.fail_probability = fail_probability

    def forward(self, *, inputs):
        """Process inputs, potentially raising an exception."""
        prompts = inputs.get("prompts", [])

        # Check for failure conditions
        if isinstance(prompts, list):
            if any(p in self.fail_on_inputs for p in prompts):
                raise self.exception_type(f"Failed on input: {prompts}")
        elif prompts in self.fail_on_inputs:
            raise self.exception_type(f"Failed on input: {prompts}")

        # Random failure
        if random.random() < self.fail_probability:
            raise self.exception_type("Random failure")

        # Success case
        if isinstance(prompts, list):
            results = [f"{p}_success" for p in prompts]
        else:
            results = [f"{prompts}_success"]

        return {"results": results}


class MockModule(MockOperator):
    """A simplified Module-like operator for testing."""

    def __init__(self):
        super().__init__()
        self.processed_count = 0
        self.name = "module_op"

    def forward(self, *, inputs):
        """Process a single input or batch of inputs."""
        prompts = inputs.get("prompts", [])

        # Increment the processed count (since we're mocking we can be mutable)
        if isinstance(prompts, list):
            self.processed_count += len(prompts)
            results = [f"{p}_module" for p in prompts]
        else:
            self.processed_count += 1
            results = [f"{prompts}_module"]

        return {"results": results}


class ComplexInputOperator(MockOperator):
    """Operator that handles complex nested input structures."""

    def forward(self, *, inputs):
        """Process complex nested inputs."""
        # Extract and process different input types
        prompts = inputs.get("prompts", [])
        config = inputs.get("config", {})
        metadata = inputs.get("metadata", None)

        # Process prompts
        if isinstance(prompts, list):
            processed_prompts = [f"{p}_complex" for p in prompts]
        else:
            processed_prompts = [f"{prompts}_complex"]

        # Process config
        processed_config = {}
        for k, v in config.items():
            if isinstance(v, str):
                processed_config[k] = f"{v}_cfg"
            else:
                processed_config[k] = v

        # Process metadata
        if metadata:
            processed_metadata = {
                "source": f"{metadata.get('source', '')}_meta",
                "timestamp": metadata.get("timestamp", 0) + 1,
            }
        else:
            processed_metadata = None

        return {
            "results": processed_prompts,
            "processed_config": processed_config,
            "metadata": processed_metadata,
        }


class AsyncBehaviorOperator(MockOperator):
    """Operator that simulates asynchronous behavior with varying execution times."""

    def __init__(self, base_time=0.01, variance=0.005):
        super().__init__()
        self.base_time = base_time
        self.variance = variance
        self.execution_times = {}

    def forward(self, *, inputs):
        """Process inputs with variable execution times."""
        prompts = inputs.get("prompts", [])
        thread_id = threading.current_thread().ident

        # Variable execution time
        if isinstance(prompts, list):
            results = []
            for i, prompt in enumerate(prompts):
                # Each prompt gets a different execution time
                exec_time = self.base_time + (random.random() * self.variance)
                time.sleep(exec_time)

                # Record execution time
                if thread_id not in self.execution_times:
                    self.execution_times[thread_id] = []
                self.execution_times[thread_id].append(exec_time)

                results.append(f"{prompt}_async_{exec_time:.6f}")
        else:
            exec_time = self.base_time + (random.random() * self.variance)
            time.sleep(exec_time)

            # Record execution time
            if thread_id not in self.execution_times:
                self.execution_times[thread_id] = []
            self.execution_times[thread_id].append(exec_time)

            results = [f"{prompts}_async_{exec_time:.6f}"]

        return {"results": results, "thread_id": thread_id}
