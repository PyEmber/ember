"""Simplified Container Operator Example.

This example demonstrates how to create a container operator that encapsulates
a simple processing pipeline, using JIT for tracing.

To run:
    uv run python src/ember/examples/container_simplified.py
"""

import logging
import time
from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel
from ember.xcs.tracer.tracer_decorator import jit

###############################################################################
# Custom Input/Output Models
###############################################################################


class ProcessingInput(EmberModel):
    """Input for text processing pipeline.

    Attributes:
        text: The input text to be processed.
    """

    text: str = Field(description="The input text to be processed")


class ProcessingOutput(EmberModel):
    """Output for text processing pipeline.

    Attributes:
        processed_text: The text after processing.
        word_count: The number of words in the processed text.
        processing_time: The time taken to process the text in seconds.
    """

    processed_text: str = Field(description="The text after processing")
    word_count: int = Field(description="Number of words in the processed text")
    processing_time: float = Field(
        description="Time taken to process the text in seconds"
    )


class ProcessingSpecification(Specification):
    """Specification for text processing pipeline."""

    input_model: Type[EmberModel] = ProcessingInput
    structured_output: Type[EmberModel] = ProcessingOutput


###############################################################################
# Component Operators
###############################################################################


@jit()
class TextNormalizer(Operator[ProcessingInput, ProcessingOutput]):
    """Operator that normalizes text (converts to lowercase, etc.)."""

    specification: ClassVar[Specification] = ProcessingSpecification()

    def forward(self, *, inputs: ProcessingInput) -> ProcessingOutput:
        """Normalize the input text.

        Args:
            inputs: Input containing the text to normalize.

        Returns:
            Normalized text with processing statistics.
        """
        start_time = time.time()
        text = inputs.text.lower().strip()
        time.sleep(0.1)  # Simulate processing time

        return ProcessingOutput(
            processed_text=text,
            word_count=len(text.split()),
            processing_time=time.time() - start_time,
        )


@jit()
class TextEnhancer(Operator[ProcessingOutput, ProcessingOutput]):
    """Operator that enhances text (adds formatting, etc.)."""

    specification: ClassVar[Specification] = ProcessingSpecification()

    def forward(self, *, inputs: ProcessingOutput) -> ProcessingOutput:
        """Enhance the input text.

        Args:
            inputs: Input containing already processed text to enhance.

        Returns:
            Enhanced text with updated processing statistics.
        """
        start_time = time.time()
        text = inputs.processed_text.capitalize()
        time.sleep(0.1)  # Simulate processing time

        return ProcessingOutput(
            processed_text=text + "!",
            word_count=inputs.word_count,
            processing_time=inputs.processing_time + (time.time() - start_time),
        )


###############################################################################
# Container Operator
###############################################################################


@jit()
class TextProcessor(Operator[ProcessingInput, ProcessingOutput]):
    """Container operator that applies normalization and enhancement."""

    specification: ClassVar[Specification] = ProcessingSpecification()

    # Define instance attributes with type hints
    normalizer: TextNormalizer
    enhancer: TextEnhancer

    def __init__(self) -> None:
        """Initialize with component operators."""
        self.normalizer = TextNormalizer()
        self.enhancer = TextEnhancer()

    def forward(self, *, inputs: ProcessingInput) -> ProcessingOutput:
        """Process the input text through the pipeline.

        Args:
            inputs: Input containing the text to process.

        Returns:
            Enhanced and normalized text with processing statistics.
        """
        # First normalize
        normalized = self.normalizer(inputs=inputs)

        # Then enhance
        enhanced = self.enhancer(inputs=normalized)

        return enhanced


def main() -> None:
    """Run a demonstration of the container operator."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n=== Container Operator Example ===\n")

    # Create the container operator
    processor = TextProcessor()

    # Process a text
    input_text = "Hello, world! This is a CONTAINER operator example."
    result = processor(inputs=ProcessingInput(text=input_text))

    print(f"Input: {input_text}")
    print(f"Output: {result.processed_text}")
    print(f"Word Count: {result.word_count}")
    print(f"Processing Time: {result.processing_time:.4f}s")

    # Show processing steps
    normalized_text = input_text.lower().strip()
    enhanced_text = normalized_text.capitalize() + "!"
    print("\nProcessing Steps:")
    print(f"1. Normalization: '{input_text}' -> '{normalized_text}'")
    print(f"2. Enhancement: '{normalized_text}' -> '{enhanced_text}'")

    # Process another text to demonstrate cached execution
    print("\n--- Second run (should use cached plan) ---")
    input_text2 = "ANOTHER example for DEMONSTRATION."
    start_time = time.time()
    result2 = processor(inputs=ProcessingInput(text=input_text2))
    elapsed = time.time() - start_time

    print(f"Input: {input_text2}")
    print(f"Output: {result2.processed_text}")
    print(f"Word Count: {result2.word_count}")
    print(f"Processing Time from Operator: {result2.processing_time:.4f}s")
    print(f"Total Wall Time: {elapsed:.4f}s")

    # Show processing steps for second run
    normalized_text2 = input_text2.lower().strip()
    enhanced_text2 = normalized_text2.capitalize() + "!"
    print("\nProcessing Steps:")
    print(f"1. Normalization: '{input_text2}' -> '{normalized_text2}'")
    print(f"2. Enhancement: '{normalized_text2}' -> '{enhanced_text2}'")
    print(f"Note: Second run is typically faster due to cached execution")


if __name__ == "__main__":
    main()
