<p align="center">
  <img src="assets/logo_ember_icon@2x.png" alt="Ember Logo" width="150"/>
</p>
<p align="center">
  <img src="assets/ember_workmark.svg" alt="Ember" width="350"/>
</p>

<p align="center">
<strong>Contributors</strong>
</p>

<p align="center">
This repository is in collaboration with the following early users, contributors, and reviewers:
</p>

<p align="center">
Jared Quincy Davis<sup>F,S</sup>, Marquita Ellis<sup>I</sup>, Diana Arroyo<sup>I</sup>, Pravein Govindan Kannan<sup>I</sup>, Paul Castro<sup>I</sup>, Siddharth Sharma<sup>F,S</sup>, Lingjiao Chen<sup>MS</sup>, Omar Khattab<sup>D,MT</sup>, Alan Zhu<sup>B</sup>, Parth Asawa<sup>B</sup>, Connor Chow<sup>B</sup>, Jason Lee<sup>B</sup>, Jay Adityanag Tipirneni<sup>B</sup>, Chad Ferguson<sup>B</sup>, Kathleen Ge<sup>B</sup>, Kunal Agrawal<sup>B</sup>, Rishab Bhatia<sup>B</sup>, Rohan Penmatcha<sup>B</sup>, Sai Kolasani<sup>B</sup>, Théo Jaffrelot Inizan<sup>B</sup>, Deepak Narayanan<sup>N</sup>, Long Fei<sup>F</sup>, Aparajit Raghavan<sup>F</sup>, Eyal Cidon<sup>F</sup>, Jacob Schein<sup>F</sup>, Prasanth Somasundar<sup>F</sup>, Boris Hanin<sup>F,P</sup>, James Zou<sup>S</sup>, Joey Gonzalez<sup>B</sup>, Peter Bailis<sup>G,S</sup>, Ion Stoica<sup>A,B,D</sup>, Matei Zaharia<sup>D,B</sup>
</p>

<p align="center">
<sup>F</sup> Foundry (MLFoundry), <sup>D</sup> Databricks, <sup>I</sup> IBM Research, <sup>S</sup> Stanford University, <sup>B</sup> UC Berkeley, <sup>MT</sup> MIT, <sup>N</sup> NVIDIA, <sup>MS</sup> Microsoft, <sup>A</sup> Anyscale, <sup>G</sup> Google, <sup>P</sup> Princeton
</p>

# <span style="color:#0366d6;">Ember</span>: A Compositional Framework for Compound AI Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Ember in a Nutshell

Aspirationally, Ember is to Networks of Networks (NONs) Compound AI Systems development what PyTorch 
and XLA are to Neural Networks (NN) development. It's a compositional framework with both eager 
execution affordances and graph execution optimization capabilities. It enables users to compose 
complex NONs, and supports automatic parallelization and optimization of these. 

Ember's vision is to enable development of **compound AI systems composed of, one day, millions-billions of inference calls** and beyond. Simple constructs--like **best-of-N graphs**, **verifier-prover structures**, and **ensembles with “voting-based” aggregation**--work surprisingly well in many regimes. 

This led us to believe that there is a rich architecture space for constructing and optimizing what we call “networks of networks” graphs, or **NONs**. This is analogous to how neural network architecture research uncovered many emergent properties of systems composed of simple artificial neurons. It would be frictionful to conduct NN research if we had to implement architectures from scratch via for-loops or implement bespoke libraries for vectorization and efficient execution. Similarly, it can be challenging at present to compose NON architectures of many calls, despite the **rapidly falling cost-per-token of intelligence**.

Ember's goal is to help unlock research and practice along this new frontier. 

## Documentation & Examples

- [Architecture Overview](ARCHITECTURE.md)
- [Quick Start Guide](QUICKSTART.md)
- [Model Registry Guide](docs/quickstart/model_registry.md)
- [Operators Guide](docs/quickstart/operators.md)
- [NON Patterns](docs/quickstart/non.md)
- [Data Processing](docs/quickstart/data.md)
- [Configuration](docs/quickstart/configuration.md)
- [Examples Directory](src/ember/examples)

## Simple Example: `horizontal` inference-time scaling with best-of-N

```python
from typing import ClassVar
from ember.api.operator import Operator, Specification
from ember.api.xcs import jit
from ember.api import non
from ember.api.models import EmberModel

# Define structured I/O types, in a TypedDict syntax
class QueryInput(EmberModel):
    """Input model containing the query to process.
    
    Attributes:
        query: The user's question or prompt text.
    """
    query: str
    
class ReasonedOutput(EmberModel):
    """Structured output model with answer and confidence score.
    
    Attributes:
        answer: The synthesized response text.
        confidence: A confidence score between 0.0 and 1.0.
    """
    answer: str
    confidence: float

class ReasonerSpecification(Specification):
    """Specification for the reasoning operator NON.
    
    Defines input/output models and prompt templates.
    """
    # Input/output type definitions
    input_model = QueryInput
    structured_output = ReasonedOutput
    

@jit  # Automatically optimize execution with JIT compilation
class EnsembleReasoner(Operator[QueryInput, ReasonedOutput]):
    """A multi-model ensemble reasoning Operator with a judge that reasons over and generates 
    a new response, informed by candidate responses.
    
    This compound system can help with reasoning robustness and reliability.
    
    Creates a pipeline that:
    1. Distributes a query to multiple model instances in parallel, generating `candidate responses`
    2. Creates an aggregate single best response, informed by the candidate responses
    3. Returns a structured output with answer and confidence
    
    Attributes:
        specification: The input/output specification for this operator.
        ensemble: A parallel ensemble of models generating multiple `candidate responses`.
        judge: A judge Operator that combines and evaluates responses.
    """
    # Input/output specification
    specification: Specification = ReasonerSpecification()

    # Sub-operators
    ensemble: non.UniformEnsemble
    judge: non.JudgeSynthesis
    
    def __init__(self, width: int = 3) -> None:
        """Initializing the ensemble reasoning pipeline.
        
        Args:
            width: Number of parallel model instances in the ensemble.
        """
        # Creating components for the reasoning pipeline
        self.ensemble = non.UniformEnsemble(
            num_units=width,
            model_name="openai:gpt-4o",
            temperature=0.7
        )
        
        self.judge = non.JudgeSynthesis(
            model_name="anthropic:claude-3-5-sonnet",
            temperature=0.2
        )
    
    def forward(self, *, inputs: QueryInput) -> ReasonedOutput:
        """Processing a query through the ensemble reasoning pipeline.
        
        Args:
            inputs: A QueryInput containing the user's question.
            
        Returns:
            A ReasonedOutput containing the synthesized answer and confidence score.
        """
        # Getting multiple reasoning paths (executed in parallel)
        ensemble_result = self.ensemble(query=inputs.query)
        
        # Synthesizing a final response 
        synthesis = self.judge(
            query=inputs.query,
            responses=ensemble_result["responses"]
        )
        
        # Returning structured output
        return ReasonedOutput(
            answer=synthesis["final_answer"],
            confidence=float(synthesis.get("confidence", 0.0))
        )

# Use it like any Python function
reasoner = EnsembleReasoner()
result = reasoner(query="What causes the northern lights?")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence:.2f}")
```

## Core Elements

1. **Composable Operators with Rigorous Specification**: Build reliable compound AI systems from 
   type-safe, reusable components with validated inputs and outputs
2. **Automatic Parallelization**: Independent operations are automatically executed concurrently 
   across a full computational graph
3. **XCS Optimization Framework**: "Accelerated Compound Systems" Just-in-time tracing and execution optimization. XCS is inspired by XLA, but intended more for accelerating compound systems vs. linear algebra operations, tuned for models and dicts, vs for vectors and numerical computation. 
4. **Multi-Provider Support**: Unified API across OpenAI, Anthropic, Claude, Gemini, and more 
   with standardized usage tracking

## Installation

Ember uses [uv](https://github.com/astral-sh/uv) as its recommended package manager for significantly faster installations and dependency resolution.

```bash
# First, install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or 
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
# or
pip install uv  # Any platform

# Quick install using uv (recommended)
uv pip install ember-ai

# Run examples directly with uv (no activation needed)
uv run python -c "import ember; print(ember.__version__)"

# Install from source for development
git clone https://github.com/pyember/ember.git
cd ember
uv pip install -e ".[dev]"

# Traditional pip installation (alternative, slower)
pip install ember-ai
```

For detailed installation instructions, troubleshooting, and environment management, see our [Installation Guide](INSTALLATION_GUIDE.md).

## Model Registry & Provider Integration

Access models from any provider through a unified interface:

```python
from ember import initialize_ember
from ember.api.models import ModelEnum

# Initialize with multiple providers
service = initialize_ember(usage_tracking=True)

# Access models from different providers with the same API
response = service(ModelEnum.gpt_4o, "What is quantum computing?")
print(response.data)

# Track usage across providers
usage = service.usage_service.get_total_usage()
print(f"Total cost: ${usage.cost:.4f}")
```

## NON Patterns & Ensembling

Build compound AI system architectures using the Network of Networks (NON) pattern with pre-built components:

```python
from ember.api import non

# 1. Create an ensemble of 5 model instances that run in parallel
# This improves statistical robustness through multiple independent samples
ensemble = non.UniformEnsemble(
    num_units=5, 
    model_name="openai:gpt-4o-mini",
    temperature=0.7
)

# 2. Create a judge to synthesize the ensemble responses
# This combines multiple perspectives into a coherent, reasoned answer
judge = non.JudgeSynthesis(
    model_name="anthropic:claude-3-5-sonnet",
    temperature=0.2
)

# 3. Create a verifier to independently check the final output
# This adds a quality control layer for factual accuracy and coherence
verifier = non.Verifier(
    model_name="anthropic:claude-3-5-haiku",
    temperature=0.0
)

# 4. Chain components together in a sequential pipeline
# Ember automatically optimizes the execution graph
pipeline = non.Sequential(operators=[ensemble, judge, verifier])

# Execute the entire pipeline with a single call
result = pipeline(query="What causes tsunamis?")
```

## Graph Optimization & Execution

Ember's XCS system provides JAX/XLA-inspired tracing, transformation, and automatic parallelization:

```python
from ember.api.xcs import jit, structural_jit, execution_options, vmap

# Basic JIT compilation for simple optimization
@jit
class SimplePipeline(Operator):
    # ... operator implementation ...

# Advanced structural JIT with parallel execution strategy
@structural_jit(execution_strategy="parallel")
class ComplexPipeline(Operator):
    def __init__(self):
        self.op1 = SubOperator1()
        self.op2 = SubOperator2()
        self.op3 = SubOperator3()
    
    def forward(self, *, inputs):
        # These operations will be automatically parallelized when the execution graph is built
        result1 = self.op1(inputs=inputs)
        result2 = self.op2(inputs=inputs)
        
        # Combine the parallel results
        combined = self.op3(inputs={"r1": result1, "r2": result2})
        return combined

# Configure execution parameters
with execution_options(max_workers=8):
    result = pipeline(query="Complex question...") 

# Vectorized mapping for batch processing
@vmap
def process_batch(inputs, model):
    return model(inputs)
```

## Data Handling & Evaluation

Ember provides a comprehensive data processing and evaluation framework with pre-built datasets and metrics:

```python
from ember.api.data import DatasetBuilder
from ember.api.eval import EvaluationPipeline, Evaluator

# Load a dataset with the builder pattern
dataset = (DatasetBuilder()
    .from_registry("mmlu")  # Use a registered dataset
    .subset("physics")      # Select a specific subset
    .split("test")          # Choose the test split
    .sample(100)            # Random sample of 100 items
    .transform(              # Apply transformations
        lambda x: {"query": f"Question: {x['question']}"} 
    )
    .build())

# Create a comprehensive evaluation pipeline
eval_pipeline = EvaluationPipeline([
    # Standard metrics
    Evaluator.from_registry("accuracy"),
    Evaluator.from_registry("response_quality"),
    
    # Custom evaluation metrics
    Evaluator.from_function(
        lambda prediction, reference: {
            "factual_accuracy": score_factual_content(prediction, reference)
        }
    )
])

# Evaluate a model or operator
results = eval_pipeline.evaluate(my_model, dataset)
print(f"Accuracy: {results['accuracy']:.2f}")
print(f"Response Quality: {results['response_quality']:.2f}")
print(f"Factual Accuracy: {results['factual_accuracy']:.2f}")
```

## License

Ember is released under the [MIT License](LICENSE).
