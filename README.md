# Ember: Compositional Framework for Compound AI Systems

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

Ember is a powerful, extensible Python framework for building and orchestrating Compound AI Systems and "Networks of Networks" (NONs). It provides a familiar experience to PyTorch/FLAX users while incorporating the efficiency benefits of JAX/XLA-like graph execution.

**Core Features:**
- 🔥 **Eager Execution by Default**: Intuitive development experience with immediate execution results
- 🚀 **Parallel Graph Scheduling**: Automatic optimization for concurrent operations
- 🧩 **Composable Operators**: PyTorch-like modular components for high reusability
- 🔌 **Extensible Registry System**: First-class support for custom models, operators, and evaluators
- ⚡ **Efficient Multi-Model Pipelines**: Optimized for complex LLM orchestration at scale
- 🧠 **Enhanced JIT System**: JAX-inspired tracing and graph optimization
- 📊 **Built-in Evaluation**: Comprehensive tools for measuring and analyzing model performance
- 🔄 **Powerful Data Handling**: Extensible data pipeline integration with popular datasets

## Quick Installation

```bash
# Install using pip
pip install ember-ai

# Or using Poetry
poetry add ember-ai
```

## A Simple Example: Multi-Model Ensemble

Below is a short snippet showcasing Ember's API for a multi-model ensemble:

```python
from ember.core.registry.model.model_module import LMModuleConfig
from ember.core.non import UniformEnsemble, JudgeSynthesis
from ember.xcs.graph import XCSGraph
from ember.xcs.engine import execute_graph

# Create a model ensemble and a judge
ensemble = UniformEnsemble(num_units=3, model_name="openai:gpt-4o", temperature=0.7)
judge = JudgeSynthesis(model_name="anthropic:claude-3.5-sonnet")

# Build an execution graph
graph = XCSGraph()
graph.add_node(operator=ensemble, node_id="ensemble")
graph.add_node(operator=judge, node_id="judge")
graph.add_edge(from_id="ensemble", to_id="judge")

# Execute with automatic parallelization
result = execute_graph(
    graph=graph,
    global_input={"query": "What is the capital of France?"},
    max_workers=3
)

print(f"Final answer: {result.final_answer}")
```

**What's happening?**
- We create a 3-instance ensemble using the same OpenAI model with variation from temperature
- A judge operator evaluates and synthesizes the ensemble's responses
- The graph executes with automatic parallelization of the ensemble members
- We get a single, high-quality response as the final output

## Core Components

### Model Registry

The Model Registry provides central management for model configurations and endpoints:

```python
from ember.core.registry.model import ModelRegistry, ModelInfo, ModelCost, RateLimit
from ember.core.registry.model.base.services import ModelService, UsageService

# Initialize registry with tracking
registry = ModelRegistry()
usage_service = UsageService()
model_service = ModelService(registry=registry, usage_service=usage_service)

# Register a model with metadata
registry.register_model(ModelInfo(
    id="anthropic:claude-3-sonnet",
    name="Claude 3 Sonnet",
    cost=ModelCost(
        input_cost_per_thousand=0.008,
        output_cost_per_thousand=0.024
    ),
    rate_limit=RateLimit(
        tokens_per_minute=100000,
        requests_per_minute=5000
    )
))

# Use the model through the service
response = model_service.invoke_model(
    model_id="anthropic:claude-3-sonnet",
    prompt="Explain quantum computing in simple terms."
)
```

### Operators

Operators encapsulate computation units, similar to PyTorch's `nn.Module`:

```python
from ember.core.registry.operator.base import Operator
from ember.core.registry.prompt_signature import Signature
from pydantic import BaseModel
from typing import Dict, Any

class SummarizerInput(BaseModel):
    text: str
    max_words: int = 100

class SummarizerOutput(BaseModel):
    summary: str
    word_count: int

class SummarizerSignature(Signature):
    input_model = SummarizerInput
    structured_output = SummarizerOutput
    prompt_template = """Summarize the following text in {max_words} words or less:
    
{text}
    
Summary:"""

class SummarizerOperator(Operator[SummarizerInput, SummarizerOutput]):
    signature = SummarizerSignature()
    
    def __init__(self, model_name: str = "openai:gpt-4o"):
        self.lm_module = LMModule(LMModuleConfig(model_name=model_name))
        
    def forward(self, *, inputs: SummarizerInput) -> SummarizerOutput:
        response = self.lm_module(self.signature.render_prompt(inputs=inputs))
        # Simple extraction logic for this example
        summary = response.strip()
        word_count = len(summary.split())
        return SummarizerOutput(summary=summary, word_count=word_count)
```

### Advanced Graph Execution

For complex workflows with multiple components, Ember enables explicit graph creation and parallel execution:

```python
from ember.xcs.graph import XCSGraph
from ember.xcs.engine import compile_graph, TopologicalSchedulerWithParallelDispatch

# Define operators (using operators from previous examples)
ensemble = UniformEnsemble(num_units=5, model_name="openai:gpt-4o")
summarizer = SummarizerOperator(model_name="anthropic:claude-3-sonnet")
judge = JudgeSynthesis(model_name="openai:o1-preview")

# Create execution graph
graph = XCSGraph()
graph.add_node(operator=ensemble, node_id="ensemble")
graph.add_node(operator=summarizer, node_id="summarizer")
graph.add_node(operator=judge, node_id="judge")
graph.add_edge(from_id="ensemble", to_id="judge")
graph.add_edge(from_id="summarizer", to_id="judge")

# Compile and execute with parallel scheduler
plan = compile_graph(graph)
scheduler = TopologicalSchedulerWithParallelDispatch(max_workers=5)
result = scheduler.run_plan(
    plan=plan, 
    global_input={"query": "Explain the benefits of quantum computing", "text": long_article},
    graph=graph
)
```

### Enhanced JIT API

Ember provides a JAX-inspired JIT (Just-In-Time) compilation system for optimal performance:

```python
from ember.xcs.tracer import jit

@jit
def complex_llm_pipeline(query: str, context: str):
    # Automatic graph construction and optimization
    ensemble = UniformEnsemble(num_units=3, model_name="openai:gpt-4o")
    responses = ensemble({"query": query})
    
    judge = JudgeSynthesis(model_name="anthropic:claude-3-sonnet")
    result = judge({"query": query, "responses": responses.responses})
    
    return result.final_answer

# Efficient execution with automatic parallelization
answer = complex_llm_pipeline(
    query="What is quantum entanglement?",
    context="I need a clear, concise explanation."
)
```

## Use Cases

Ember is well-suited for complex AI pipelines such as:

- **Multi-model ensembling** for improved answer quality and robustness
- **Verification and self-correction** pipelines with structured workflows
- **Multi-agent systems** with specialized roles and coordination
- **Tool-augmented systems** combining LLMs with specialized tools
- **Complex reasoning chains** with intermediate evaluations and refinement
- **Evaluation and benchmarking** of compound AI systems at scale

## Documentation

For comprehensive documentation, including tutorials, API reference, and advanced usage:

- [Ember Documentation](https://ember-ai.readthedocs.io/)
- [Architecture Overview](ARCHITECTURE.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Examples Directory](examples/)

### Quickstart Guides

- [Model Registry Quickstart](docs/quickstart/model_registry.md) - Get started with LLM integration
- [Operators Quickstart](docs/quickstart/operators.md) - Build reusable computation units
- [Prompt Signatures Quickstart](docs/quickstart/prompt_signatures.md) - Type-safe prompt engineering
- [NON Quickstart](docs/quickstart/non.md) - Networks of Networks patterns
- [Data Quickstart](docs/quickstart/data.md) - Working with datasets and evaluation

## Performance

Ember is designed for high-performance LLM orchestration:

- **Parallel execution** of independent operators
- **Minimal overhead** compared to direct API calls
- **Optimized scheduling** for complex dependency graphs
- **Efficient memory usage** with smart caching and pruning
- **Horizontal scaling** for distributed workloads

## Community

- [GitHub Issues](https://github.com/pyember/ember/issues): Bug reports and feature requests
- [GitHub Discussions](https://github.com/pyember/ember/discussions): Questions and community support
- [Discord](https://discord.gg/ember-ai): Join our community chat

## License

Ember is released under the [Apache 2.0 License](LICENSE).

---

Built with 🔥 by the Ember community