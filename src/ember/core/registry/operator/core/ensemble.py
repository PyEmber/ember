"""
Parallel Ensemble Pattern Implementation

This module implements the Ensemble pattern, a foundational distributed inference pattern
that enables parallel execution of multiple language model instances. This pattern addresses
several critical needs in Compound AI Systems applications:

1. Statistical Robustness: Multiple "independent" inferences reduce variance in outputs
2. Throughput Optimization: Concurrent execution maximizes utilization of available resources, or rate-limits
3. Entropy and diversity of perspectives: Different models or configurations can provide complementary insights

The pattern serves as a foundational building block for more complex NON patterns
and reliability-focused LLM applications.
"""

from __future__ import annotations

from typing import List

from ember.core.types import EmberModel

from ember.core.registry.operator.base.operator_base import Operator

from ember.core.registry.specification.specification import Specification
from ember.core.registry.model.model_module.lm import LMModule


class EnsembleOperatorInputs(EmberModel):
    """
    Structured input model for ensemble inference operations.
    
    This model defines the minimal input contract for ensemble operations,
    focusing on the core query text while allowing the specification to
    handle rendering details. The model is intentionally minimalist to
    enable maximum flexibility in ensemble configurations.
    
    Attributes:
        query: The primary text query to be sent to all language models
              in the ensemble. This text will be rendered according to
              the operator's specification template if one is defined.
    """
    query: str


class EnsembleOperatorOutputs(EmberModel):
    """
    Structured output model for ensemble inference results.
    
    This model provides a standardized container for the parallel responses
    generated by the ensemble. The ordered list preserves the relationship
    between responses and their source models, enabling downstream operators
    to apply model-specific weighting or processing if needed.
    
    Notable design considerations:
    - Maintains original response ordering for reproducibility
    - Preserves raw text outputs for maximum flexibility
    - Simple structure facilitates easy aggregation and analysis
    
    Attributes:
        responses: Ordered list of text responses from the language models
                 in the ensemble. Each element corresponds to the output
                 from one model instance, preserving the original order of
                 the LMModules provided to the operator.
    """
    responses: List[str]


class EnsembleOperator(Operator[EnsembleOperatorInputs, EnsembleOperatorOutputs]):
    """
    Executes the same query across multiple language models in parallel.
    
    Sends an identical prompt to each LM in the ensemble and collects all responses.
    This enables multiple independent samples from language models, which can
    be used for robustness, consensus, or diversity of outputs.
    
    The execution is implicitly parallel, with each model potentially running
    concurrently depending on the implementation.
    """

    specification: Specification = Specification(
        input_model=EnsembleOperatorInputs, structured_output=EnsembleOperatorOutputs
    )
    lm_modules: List[LMModule]

    def __init__(self, *, lm_modules: List[LMModule]) -> None:
        """
        Initializes the ensemble with a collection of language model modules.
        
        The constructor follows the Dependency Injection principle, accepting
        pre-configured LMModule instances rather than creating them internally.
        This approach provides maximum flexibility for ensemble configuration,
        enabling diverse model combinations and specialized configurations.
        
        The implementation preserves the order of the provided modules,
        ensuring deterministic execution and reproducible results.
        
        Args:
            lm_modules: Collection of language model modules to execute in parallel.
                      These modules must conform to the LMModule interface, but
                      can represent different model providers, model versions, or
                      configuration parameters.
        """
        self.lm_modules = lm_modules

    def forward(self, *, inputs: EnsembleOperatorInputs) -> EnsembleOperatorOutputs:
        """
        Executes the query across all language models.
        
        Args:
            inputs: Contains the query to send to all models.
                  
        Returns:
            Contains all model responses in an ordered list matching 
            the original lm_modules order.
        """
        rendered_prompt: str = self.specification.render_prompt(inputs=inputs)
        responses: List[str] = [lm(prompt=rendered_prompt) for lm in self.lm_modules]
        return {"responses": responses}
