"""Container Operator Example with JIT.

This example demonstrates how to create a container operator that encapsulates
a complex pipeline, using JIT for tracing.

Note: In the current implementation, the JIT decorator enables tracing but
doesn't automatically build execution graphs. For automatic graph building
and parallel execution, you'd still need to use XCSGraph and execute_graph
(shown in the other examples).

To run:
    poetry run python src/ember/examples/container_operator_example.py
"""

import logging
import time
from typing import Any, ClassVar, Dict, List, Optional, Type

from ember.core import non

# ember imports
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel
from ember.xcs.tracer.tracer_decorator import jit


###############################################################################
# Custom Input/Output Models
###############################################################################
class QuestionAnsweringInput(EmberModel):
    """Input for question answering pipeline."""

    query: str


class QuestionAnsweringOutput(EmberModel):
    """Output for question answering pipeline."""

    answer: str
    confidence: float
    model_responses: List[str]


class QuestionAnsweringSpecification(Specification):
    """Specification for question answering pipeline."""

    input_model: Type[EmberModel] = QuestionAnsweringInput
    structured_output: Type[EmberModel] = QuestionAnsweringOutput


###############################################################################
# Container Operator with JIT
###############################################################################
@jit(sample_input={"query": "What is the capital of France?"})
class QuestionAnsweringPipeline(
    Operator[QuestionAnsweringInput, QuestionAnsweringOutput]
):
    """Container operator that encapsulates a complete question answering pipeline.

    This operator is decorated with @jit for tracing. The tracing doesn't
    automatically optimize execution in the current implementation, but lays
    the groundwork for future automatic optimization.

    The pipeline internally uses an ensemble of models followed by an aggregation step.
    """

    # Class-level specification declaration
    specification: ClassVar[Specification] = QuestionAnsweringSpecification()

    # Class-level field declarations
    model_name: str
    num_units: int
    temperature: float
    ensemble: non.UniformEnsemble
    aggregator: non.MostCommon

    def __init__(
        self, *, model_name: str, num_units: int = 3, temperature: float = 0.7
    ) -> None:
        """Initialize the pipeline.

        Args:
            model_name: The model to use
            num_units: Number of ensemble members
            temperature: Generation temperature
        """
        self.model_name = model_name
        self.num_units = num_units
        self.temperature = temperature

        # Create internal operators
        self.ensemble = non.UniformEnsemble(
            num_units=self.num_units,
            model_name=self.model_name,
            temperature=self.temperature,
        )
        self.aggregator = non.MostCommon()

    def forward(self, *, inputs: QuestionAnsweringInput) -> QuestionAnsweringOutput:
        """Execute the pipeline.

        Args:
            inputs: The input query

        Returns:
            Structured output with the final answer, confidence, and model responses
        """
        # Run the ensemble to get multiple responses
        ensemble_result = self.ensemble(inputs={"query": inputs.query})
        responses = ensemble_result["responses"]

        # Aggregate responses
        aggregated = self.aggregator(
            inputs={"query": inputs.query, "responses": responses}
        )

        # Extract confidence from aggregation
        confidence = aggregated.get("confidence", 0.0)

        # Return structured output model instance
        return QuestionAnsweringOutput(
            answer=aggregated["final_answer"],
            confidence=confidence,
            model_responses=responses,
        )


###############################################################################
# Main Demonstration
###############################################################################
def main() -> None:
    """Run demonstration of container operator with JIT."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n=== Container Operator with JIT ===\n")

    # Create the pipeline
    pipeline = QuestionAnsweringPipeline(model_name="openai:gpt-4o-mini", num_units=3)

    # Example queries
    queries = [
        "What is the capital of France?",
        "How many planets are in our solar system?",
        "Who wrote 'Pride and Prejudice'?",
    ]

    # Process all queries
    for query in queries:
        print(f"\nProcessing query: {query}")

        # Time the execution - using kwargs format for cleaner code
        start_time = time.perf_counter()
        result = pipeline(inputs={"query": query})
        end_time = time.perf_counter()

        # Display results
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Execution time: {end_time - start_time:.4f}s")

    print("\nNote: The first execution includes tracing overhead. In the current")
    print("      implementation, this tracing doesn't automatically optimize execution")
    print("      but lays the groundwork for future optimizations.")


if __name__ == "__main__":
    main()
