# Enhanced JIT API for Ember

## Overview

This document describes the design for an enhanced JIT (Just-In-Time) compilation API for Ember, providing a cleaner, more JAX-like user experience for building and executing complex operator DAGs.

> **Note**: This document describes the JIT compilation vision for Ember. The current implementation includes three complementary approaches:
>
> 1. **autograph** - A context manager for explicit graph building when maximum control is needed
> 2. **jit** - A decorator that traces execution and automatically builds graphs from observed behavior
> 3. **structural_jit** - An advanced decorator that analyzes operator composition without execution tracing
>
> Together, these provide a JAX-like experience where operators can be composed naturally with automatic parallelization. The system handles most common use cases, with ongoing development of advanced features like complete transforms integration and advanced optimization.

## Goals

1. Simplify the user experience when working with complex operator DAGs
2. Eliminate the need for manual graph construction in common cases
3. Enable transparent caching and reuse of execution plans
4. Allow flexible configuration of execution parameters
5. Match the elegance of JAX-like systems while preserving Ember's eager-mode compatible UX

## Implementation Components

### 1. Enhanced JIT Decorator

The core of the enhanced system is an improved `@jit` decorator that:

- Automatically traces operator execution
- Builds execution graphs from traces
- Caches graphs for reuse
- Supports sample inputs for initialization-time tracing

```python
@jit(sample_input={"query": "example"})
class MyOperator(Operator[InputType, OutputType]):
    # Implementation
    ...
```

### 2. Execution Options Context

A thread-local context manager that provides control over execution parameters:

```python
with execution_options(scheduler="parallel", max_workers=10):
    result = jit_op(inputs={"query": "example"})
```

### 3. Graph Builder

Internal utilities that automatically convert traces to execution graphs:

- Discovers dependencies between operators
- Creates efficient execution plans
- Handles both sequential and parallel execution
- Properly manages nested operator relationships
- Supports branching and merging execution patterns

## User Experience

### Basic Usage

```python
# Define a JIT-enabled operator
@jit()
class Ensemble(Operator):
    # ...implementation...
    pass

# Create and use it - no manual graph building required
ensemble = Ensemble(num_units=5, model_name="gpt-4o")
result = ensemble(inputs={"query": "What is machine learning?"})
```

### Composition Patterns

The enhanced API supports three composition patterns:

1. **Nested Pipeline Class**:
```python
@jit()
class Pipeline(Operator):
    def __init__(self):
        self.refiner = QuestionRefinement()
        self.ensemble = Ensemble()
        self.aggregator = MostCommon()
    
    def forward(self, inputs):
        refined = self.refiner(inputs)
        answers = self.ensemble(refined)
        return self.aggregator(answers)
```

2. **Functional Composition**:
```python
pipeline = compose(aggregator, compose(ensemble, refiner))
result = pipeline(inputs)
```

3. **Sequential Chaining**:
```python
def pipeline(inputs):
    refined = refiner(inputs)
    answers = ensemble(refined)
    return aggregator(answers)
```

### Execution Control

```python
# Default execution
result = pipeline(inputs)

# With custom execution options
with execution_options(scheduler="topo_sort_parallel_dispatch"):
    result = pipeline(inputs)
```

## Performance Benefits

The enhanced JIT system offers several performance advantages:

1. **Reduced overhead**: Only build graphs once, reuse for subsequent calls
2. **Automatic parallelism**: Intelligently schedule operations in parallel, based on topo sort of Operator DAG. 
3. **Optimized memory usage**: Minimize redundant data copying, and optionally cache calls

## Implementation Details

The core implementation accomplishes several key technical goals:

1. **Hierarchical Analysis**: The system builds a hierarchy map to understand parent-child relationships between operators, enabling proper handling of nested execution.

2. **Advanced Dependency Detection**: The dependency analysis algorithm identifies true data dependencies while respecting hierarchical relationships between operators.

3. **Execution Flow Modeling**: The system correctly models complex execution patterns including branching, merging, and nested operator invocations.

4. **Comprehensive Testing**: The implementation includes robust tests for a wide range of execution patterns, ensuring correct behavior in complex scenarios.

## Design Principles

The implementation adheres to several key design principles:

1. **SOLID**: SOLID adherence for modularity and extensibility. People should be able to add custom schedulers. 
2. **Minimalism**: Keep the API surface small and focused
3. **Composability**: Enable building gnarly, complex pipelines from simple components
4. **Pythonic**: Follow Python idioms and feel natural to Python developers with an ML research background
5. **Progressive disclosure**: Easy for beginners, whilst powerful for experts (tensegrity)

## Current Status

The implementation now provides:

1. **Complete tracing system**: Records detailed execution information
2. **Sophisticated dependency analysis**: Properly handles nested operators
3. **Advanced graph building**: Constructs execution graphs with correct dependencies
4. **Support for complex patterns**: Handles branching, merging, and nested execution

## Future Work

Future enhancements to consider:

1. **Optimized Graph Execution**: Use cached graphs for subsequent runs
2. **Smart Caching**: Implement intelligent caching of intermediate results
3. **Dynamic Graph Updates**: Support runtime graph modifications based on execution patterns
4. **Integration with profiling**: Add instrumentation for performance analysis
5. **Extended composition utilities**: More helpers for building complex pipelines