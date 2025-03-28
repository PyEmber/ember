# Ember NON (Networks of Networks) - Quickstart Guide

This guide introduces Ember's Networks of Networks (NON) module, which provides high-level abstractions for building complex AI systems through compositional patterns.

## 1. Understanding NON

The NON module provides ready-to-use operators that implement common compositional patterns:

- **Ensembles**: Run multiple models in parallel for robust results
- **Judges**: Evaluate and synthesize responses from multiple sources
- **Verifiers**: Check and potentially correct candidate answers
- **Sequential Pipelines**: Chain multiple operators together
- **Varied Model Combinations**: Blend different models and configurations

These abstractions make it easier to build reliable, high-performance AI systems.

## 2. Basic NON Operators

```python
from ember.core.non import (
    UniformEnsemble,
    JudgeSynthesis,
    MostCommon,
    Verifier,
    Sequential
)

# Create a simple ensemble of identical models
ensemble = UniformEnsemble(
    num_units=3,
    model_name="openai:gpt-4o",
    temperature=0.7,
    max_tokens=100
)

# Create a judge to synthesize ensemble outputs
judge = JudgeSynthesis(
    model_name="anthropic:claude-3-sonnet",
    temperature=0.1
)

# Create a simple voting mechanism
majority_vote = MostCommon()

# Create a verification operator
answer_verifier = Verifier(
    model_name="openai:gpt-4o",
    temperature=0.2
)
```

## 3. Building a Simple Ensemble + Judge Pipeline

```python
from ember.xcs.graph import XCSGraph
from ember.xcs.engine import execute_graph

# Define operators
ensemble = UniformEnsemble(num_units=3, model_name="openai:gpt-4o", temperature=0.7)
judge = JudgeSynthesis(model_name="anthropic:claude-3-sonnet")

# Create a graph
graph = XCSGraph()
graph.add_node(operator=ensemble, node_id="ensemble")
graph.add_node(operator=judge, node_id="judge")
graph.add_edge(from_id="ensemble", to_id="judge")

# Execute the graph
result = execute_graph(
    graph=graph,
    global_input={"query": "What are the three laws of thermodynamics?"},
    max_workers=3  # Parallelize ensemble execution
)

# Access the final answer
print(f"Final answer: {result.final_answer}")
```

## 4. Advanced NON Patterns

### Self-verification Pipeline

```python
from ember.core.non import UniformEnsemble, Verifier, Sequential

# Define a self-verification pipeline
def create_self_verification_pipeline(question):
    # Generate initial answer
    initial_model = UniformEnsemble(
        num_units=1,  # Just one model for initial answer
        model_name="openai:gpt-4o",
        temperature=0.3
    )
    
    # Verify the answer
    verifier = Verifier(
        model_name="anthropic:claude-3-sonnet",
        temperature=0.1
    )
    
    # Create sequential pipeline
    pipeline = Sequential(operators=[initial_model, verifier])
    
    # Execute the pipeline
    initial_response = initial_model(inputs={"query": question})
    candidate_answer = initial_response.responses[0]
    
    verification_result = verifier(
        inputs={
            "query": question,
            "candidate_answer": candidate_answer
        }
    )
    
    return {
        "original_answer": candidate_answer,
        "verdict": verification_result.verdict,
        "explanation": verification_result.explanation,
        "revised_answer": verification_result.revised_answer if verification_result.verdict == "incorrect" else candidate_answer
    }

# Use the pipeline
result = create_self_verification_pipeline("What is the capital of Australia?")
print(f"Final answer: {result['revised_answer']}")
print(f"Explanation: {result['explanation']}")
```

### Diverse Ensemble with Varied Models

```python
from ember.core.non import VariedEnsemble, JudgeSynthesis
from ember.core.registry.model.model_module import LMModuleConfig

# Create configurations for different models
model_configs = [
    LMModuleConfig(model_name="openai:gpt-4o", temperature=0.3),
    LMModuleConfig(model_name="anthropic:claude-3-haiku", temperature=0.4),
    LMModuleConfig(model_name="openai:gpt-4o-mini", temperature=0.5),
    LMModuleConfig(model_name="deepmind:gemini-1.5-pro", temperature=0.2)
]

# Create a varied ensemble
varied_ensemble = VariedEnsemble(model_configs=model_configs)

# Create a judge
judge = JudgeSynthesis(model_name="anthropic:claude-3-sonnet")

# Execute the ensemble
responses = varied_ensemble(inputs={"query": "What are the main challenges in quantum computing?"})

# Judge the responses
final_result = judge(
    inputs={
        "query": "What are the main challenges in quantum computing?",
        "responses": responses.responses
    }
)

print(f"Final synthesized answer: {final_result.final_answer}")
```

## 5. Auto-parallelization

One of the key advantages of NON operators is automatic parallelization:

```python
from ember.xcs.engine import execute_graph
from ember.xcs.graph import XCSGraph
from ember.core.non import UniformEnsemble, MostCommon

# Create a large ensemble
ensemble = UniformEnsemble(
    num_units=10,  # 10 parallel model calls
    model_name="openai:gpt-4o-mini",
    temperature=0.8
)

majority = MostCommon()

# Build graph
graph = XCSGraph()
graph.add_node(operator=ensemble, node_id="ensemble")
graph.add_node(operator=majority, node_id="majority")
graph.add_edge(from_id="ensemble", to_id="majority")

# Execute with parallelization
result = execute_graph(
    graph=graph,
    global_input={"query": "Is P=NP a solved problem?"},
    max_workers=10  # Control parallel execution
)

print(f"Most common answer: {result.final_answer}")
```

## 6. Using NON with JIT Compilation

```python
from ember.xcs.tracer import jit
from ember.core.non import UniformEnsemble, JudgeSynthesis

# Define a JIT-compiled function with NON operators
@jit
def robust_qa_pipeline(question):
    # This will be automatically traced and optimized
    ensemble = UniformEnsemble(
        num_units=3,
        model_name="openai:gpt-4o",
        temperature=0.7
    )
    
    judge = JudgeSynthesis(
        model_name="anthropic:claude-3-sonnet",
        temperature=0.1
    )
    
    # Execute the operators (will be parallelized)
    ensemble_results = ensemble(inputs={"query": question})
    final_result = judge(
        inputs={
            "query": question,
            "responses": ensemble_results.responses
        }
    )
    
    return final_result.final_answer

# Use the optimized pipeline
answer = robust_qa_pipeline("Explain the concept of neural networks in simple terms.")
print(f"Answer: {answer}")
```

## 7. Best Practices

1. **Start Simple**: Begin with UniformEnsemble and JudgeSynthesis before exploring advanced patterns
2. **Tune Temperature**: Use higher temperature for ensemble diversity, lower for verification
3. **Optimize Workers**: Set max_workers to match your ensemble size for optimal performance
4. **Mix Providers**: Use a mix of providers (OpenAI, Anthropic, etc.) for robust answers
5. **Monitor Costs**: Track API costs when using large ensembles of powerful models
6. **Use Different Judge Models**: For best results, use a different model for judging than for ensemble members

## Next Steps

Learn more about:
- [Operators](operators.md) - Building custom computation units
- [Prompt Specifications](specifications.md) - Type-safe prompt engineering
- [Model Registry](model_registry.md) - Managing LLM configurations
- [XCS Graphs](../advanced/xcs_graphs.md) - Advanced parallel execution
- [Enhanced JIT](../advanced/enhanced_jit.md) - Optimized tracing and execution