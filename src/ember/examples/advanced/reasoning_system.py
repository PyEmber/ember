"""Advanced Reasoning System Example

This example demonstrates building a sophisticated reasoning system using Ember's
advanced features, including ensembles, verification, and synthesis of responses.

This is an advanced example that showcases:
1. Complex operator composition
2. Multiple LLM reasoning paths
3. Verification of reasoning steps
4. Synthesis of final response
5. Automated parallelization

To run:
    uv run python src/ember/examples/advanced/reasoning_system.py
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Type

# Ember API imports
from ember.api.xcs import execution_options, jit

# Keep non import for UniformEnsemble
from ember.core import non
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.operator.core.synthesis_judge import JudgeSynthesisOperator
from ember.core.registry.operator.core.verifier import VerifierOperator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel, Field

###############################################################################
# Input/Output Models
###############################################################################


class ReasoningInput(EmberModel):
    """Input for the advanced reasoning system."""

    query: str = Field(description="The reasoning query to process")
    context: Optional[str] = Field(
        default=None, description="Optional context to provide background knowledge"
    )


class ReasoningSteps(EmberModel):
    """Structured representation of reasoning steps."""

    steps: List[str] = Field(description="Individual reasoning steps")
    conclusion: str = Field(description="Conclusion based on reasoning")
    confidence: float = Field(description="Confidence score (0-1)", ge=0, le=1)


class VerificationResult(EmberModel):
    """Verification results for reasoning paths."""

    verified_steps: List[str] = Field(description="Verified reasoning steps")
    accuracy_score: float = Field(description="Accuracy score (0-1)", ge=0, le=1)
    coherence_score: float = Field(description="Coherence score (0-1)", ge=0, le=1)
    completeness_score: float = Field(
        description="Completeness score (0-1)", ge=0, le=1
    )


class ReasoningOutput(EmberModel):
    """Output from the advanced reasoning system."""

    query: str = Field(description="The original query")
    final_answer: str = Field(description="The synthesized final answer")
    confidence: float = Field(
        description="Overall confidence in the answer (0-1)", ge=0, le=1
    )
    reasoning_paths: List[ReasoningSteps] = Field(
        description="Multiple reasoning paths that were considered"
    )
    verification_results: List[VerificationResult] = Field(
        description="Verification results for each reasoning path"
    )


# Specification for the reasoning system
class ReasoningSpecification(Specification):
    """Specification for the advanced reasoning system."""

    input_model: Type[EmberModel] = ReasoningInput
    structured_output: Type[EmberModel] = ReasoningOutput


###############################################################################
# Reasoning Ensemble Operator
###############################################################################


@jit
class ReasoningEnsemble(Operator[ReasoningInput, Dict[str, Any]]):
    """Generates multiple reasoning paths for a query using different models."""

    # Class-level field declaration
    ensemble: non.UniformEnsemble

    def __init__(self, num_units: int = 3):
        """Initialize with configurable number of ensemble units."""
        self.ensemble = non.UniformEnsemble(
            num_units=num_units, model_name="openai:gpt-4o", temperature=0.7
        )

    def forward(self, *, inputs: ReasoningInput) -> Dict[str, Any]:
        """Generate multiple reasoning paths for the input query."""
        ensemble_inputs = {"query": inputs.query}

        if inputs.context:
            ensemble_inputs["context"] = inputs.context

        # This will be automatically parallelized
        ensemble_results = self.ensemble(inputs=ensemble_inputs)

        # Parse raw responses into structured reasoning steps
        reasoning_paths = []
        for response in ensemble_results.get("responses", []):
            # In a real implementation, this would use proper parsing
            steps = response.split("\n\n")[:3]  # Simplified parsing
            conclusion = response.split("\n\n")[-1] if "\n\n" in response else response

            reasoning_paths.append(
                ReasoningSteps(
                    steps=steps,
                    conclusion=conclusion,
                    confidence=0.8,  # Simplified confidence assignment
                )
            )

        return {"reasoning_paths": reasoning_paths, "query": inputs.query}


###############################################################################
# Verification Operator
###############################################################################


@jit
class ReasoningVerifier(Operator[Dict[str, Any], Dict[str, Any]]):
    """Verifies each reasoning path for accuracy, coherence, and completeness."""

    # Class-level field declaration
    verifier: VerifierOperator

    def __init__(self, model_name: str = "anthropic:claude-3-sonnet"):
        """Initialize with configurable model."""
        # Create LM module for the verifier
        lm_module = LMModule(
            config=LMModuleConfig(
                model_name=model_name,
                temperature=0.2,
            )
        )

        # Use the core VerifierOperator directly
        self.verifier = VerifierOperator(lm_module=lm_module)

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Verify each reasoning path in the inputs."""
        query = inputs.get("query", "")
        reasoning_paths = inputs.get("reasoning_paths", [])

        verification_results = []
        for path in reasoning_paths:
            # Use the verifier operator to check this reasoning path
            verification_input = {
                "query": query,
                "reasoning": path.conclusion,
                "steps": path.steps,
            }

            # This would be automatically parallelized across paths
            result = self.verifier(inputs=verification_input)

            # Parse verification results
            verification_results.append(
                VerificationResult(
                    verified_steps=path.steps,  # Simplified - would be filtered in real implementation
                    accuracy_score=result.get("accuracy_score", 0.0),
                    coherence_score=result.get("coherence_score", 0.0),
                    completeness_score=result.get("completeness_score", 0.0),
                )
            )

        return {
            "verification_results": verification_results,
            "reasoning_paths": reasoning_paths,
            "query": query,
        }


###############################################################################
# Synthesis Operator
###############################################################################


@jit
class ReasoningSynthesizer(Operator[Dict[str, Any], ReasoningOutput]):
    """Synthesizes a final answer from verified reasoning paths."""

    # Class-level field declaration
    synthesizer: JudgeSynthesisOperator

    def __init__(self, model_name: str = "anthropic:claude-3-opus"):
        """Initialize with configurable model."""
        # Create LM module for the synthesizer
        lm_module = LMModule(
            config=LMModuleConfig(
                model_name=model_name,
                temperature=0.2,
            )
        )

        # Use the core JudgeSynthesisOperator directly
        self.synthesizer = JudgeSynthesisOperator(lm_module=lm_module)

    def forward(self, *, inputs: Dict[str, Any]) -> ReasoningOutput:
        """Synthesize a final answer from multiple verified reasoning paths."""
        query = inputs.get("query", "")
        reasoning_paths = inputs.get("reasoning_paths", [])
        verification_results = inputs.get("verification_results", [])

        # Extract conclusions from reasoning paths
        conclusions = [path.conclusion for path in reasoning_paths]

        # Calculate verification scores
        verification_scores = []
        for result in verification_results:
            avg_score = (
                result.accuracy_score
                + result.coherence_score
                + result.completeness_score
            ) / 3.0
            verification_scores.append(avg_score)

        # Use the synthesizer to create the final answer
        synthesis_input = {
            "query": query,
            "conclusions": conclusions,
            "verification_scores": verification_scores,
        }

        synthesis_result = self.synthesizer(inputs=synthesis_input)

        # Calculate overall confidence as weighted average of verification scores
        # In a real implementation, this would be more sophisticated
        overall_confidence = (
            sum(verification_scores) / len(verification_scores)
            if verification_scores
            else 0.0
        )

        # Return structured output
        return ReasoningOutput(
            query=query,
            final_answer=synthesis_result.get("final_answer", "No answer generated"),
            confidence=overall_confidence,
            reasoning_paths=reasoning_paths,
            verification_results=verification_results,
        )


###############################################################################
# Complete Reasoning System
###############################################################################


@jit
class AdvancedReasoningSystem(Operator[ReasoningInput, ReasoningOutput]):
    """A sophisticated reasoning system with:
    1. Parallel LLM reasoning with different models
    2. Verification of reasoning steps
    3. Synthesis of final response
    """

    # Class-level specification declaration
    specification: ClassVar[Specification] = ReasoningSpecification()

    # Class-level field declarations with types
    reasoning_ensemble: ReasoningEnsemble
    verifier: ReasoningVerifier
    synthesizer: ReasoningSynthesizer

    def __init__(self, num_reasoning_paths: int = 3):
        """Initialize the advanced reasoning system."""
        # Initialize declared fields
        self.reasoning_ensemble = ReasoningEnsemble(num_units=num_reasoning_paths)
        self.verifier = ReasoningVerifier()
        self.synthesizer = ReasoningSynthesizer()

    def forward(self, *, inputs: ReasoningInput) -> ReasoningOutput:
        """Process the query through the complete reasoning pipeline."""
        # Step 1: Get multiple reasoning paths (executed in parallel)
        ensemble_result = self.reasoning_ensemble(inputs=inputs)

        # Step 2: Verify each reasoning path (also executed in parallel)
        verification_result = self.verifier(inputs=ensemble_result)

        # Step 3: Synthesize a final response from verified reasoning
        final_result = self.synthesizer(inputs=verification_result)

        return final_result


###############################################################################
# Example Usage
###############################################################################


def main() -> None:
    """Demonstrates the advanced reasoning system."""
    print("\n=== Advanced Reasoning System Example ===\n")

    # Create the reasoning system
    reasoning_system = AdvancedReasoningSystem(num_reasoning_paths=3)

    # Sample query
    query = "What are the potential economic implications of large-scale quantum computing adoption?"
    context = """
    Quantum computing leverages quantum mechanical phenomena to perform computations
    that would be infeasible on classical computers. Current estimates suggest that
    practical quantum computers could break widely-used cryptographic systems and
    enable significant advances in materials science, drug discovery, and optimization
    problems.
    """

    print(f"Query: {query}\n")
    print("Executing reasoning pipeline with maximum parallelization...\n")

    # Execute with parallelization enabled
    with execution_options(max_workers=4):
        result = reasoning_system(inputs={"query": query, "context": context})

    # Display results
    print(f"Final Answer: {result.final_answer}\n")
    print(f"Confidence: {result.confidence:.2f}\n")

    print(f"Number of reasoning paths: {len(result.reasoning_paths)}")
    print(f"Number of verification results: {len(result.verification_results)}\n")

    print("System Architecture:")
    print("1. ReasoningEnsemble: Generates multiple reasoning approaches in parallel")
    print(
        "2. ReasoningVerifier: Checks each reasoning path for accuracy, coherence, completeness"
    )
    print("3. ReasoningSynthesizer: Creates final response from verified reasoning\n")

    print("Key Benefits:")
    print("- Automatic parallelization of independent operations")
    print("- Structured, type-safe inputs and outputs")
    print("- Verification to reduce errors and hallucinations")
    print("- Synthesis to combine multiple perspectives\n")


if __name__ == "__main__":
    main()
