# XCS Quickstart Guide

This guide provides a quick introduction to using the XCS module for high-performance execution of computational graphs in Ember.

## Installation

XCS is included with Ember. Make sure you have the latest version:

```bash
pip install -U ember
```

## Basic Usage

## XCS JIT System

Ember provides three complementary approaches to optimizing operator execution:

### 1. JIT Compilation with @jit

The `jit` decorator uses execution tracing to automatically optimize operators:

```python
from ember.api.xcs import jit
from ember.api.operator import Operator

@jit
class SimpleOperator(Operator):
    def forward(self, *, inputs):
        return {"result": inputs["text"].upper()}

# Create and use the operator
op = SimpleOperator()
result = op(inputs={"text": "hello world"})
print(result)  # {"result": "HELLO WORLD"}
```

The `jit` decorator:
- Traces actual execution to identify dependencies
- Builds optimized execution graphs
- Caches compiled graphs for repeated use
- Works best for operators with dynamic execution patterns

### 2. Structural JIT for Complex Operator Compositions

For operators with known internal structure, the `structural_jit` decorator provides optimizations without requiring execution:

```python
from ember.api.xcs import structural_jit
from ember.api.operator import Operator

@structural_jit(execution_strategy="parallel")
class CompositeOperator(Operator):
    def __init__(self):
        self.op1 = FirstOperator()
        self.op2 = SecondOperator()
        
    def forward(self, *, inputs):
        intermediate = self.op1(inputs=inputs)
        result = self.op2(inputs=intermediate)
        return result
```

The `structural_jit` decorator:
- Analyzes operator structure directly
- Identifies potential parallelism through structural analysis
- Supports multiple execution strategies (auto, parallel, sequential)
- Works best for complex composite operators

### 3. Explicit Graph Building with autograph

For maximum control, you can build execution graphs explicitly:

```python
from ember.api.xcs import autograph, execute

# Define some operators
op1 = FirstOperator()
op2 = SecondOperator()

# Build a graph
with autograph() as graph:
    intermediate = op1(inputs={"query": "Example"})
    result = op2(inputs=intermediate)
    
# Execute the graph with optimized scheduling
results = execute(graph)
print(results)
```

This separates graph definition from execution, enabling optimization across operator boundaries.

### Parallelization Transforms

XCS provides transforms for vectorization and parallelization:

```python
from ember.api.xcs import vmap, pmap

# Vectorize a function to process batches
def process_item(item):
    return item * 2

batch_fn = vmap(process_item)
batch_results = batch_fn([1, 2, 3])  # [2, 4, 6]

# Parallelize a function across cores
def slow_computation(x):
    # Heavy computation here
    return processed_result

parallel_fn = pmap(slow_computation)
parallel_results = parallel_fn([data1, data2, data3, data4])
```

## Advanced Features

### Structural JIT

For complex operators that compose multiple sub-operators:

```python
from ember.api.xcs import structural_jit

@structural_jit
class CompositeOperator(Operator):
    def __init__(self):
        self.op1 = SubOperator1()
        self.op2 = SubOperator2()
        
    def forward(self, *, inputs):
        # Operations automatically parallelized based on structure
        intermediate = self.op1(inputs=inputs)
        result = self.op2(inputs=intermediate)
        return result
```

The `structural_jit` decorator analyzes operator structure and automatically parallelizes independent operations.

### Configuration Options

Fine-tune behavior with configuration options:

```python
from ember.api.xcs import jit, JITOptions, XCSExecutionOptions

# Configure JIT with precompilation
@jit(options=JITOptions(
    sample_input={"query": "example"},
    cache_size=100
))
class OptimizedOperator(Operator):
    def forward(self, *, inputs):
        return process_complex_input(inputs)

# Configure execution environment
with XCSExecutionOptions(
    scheduler="parallel",
    max_workers=4,
    timeout=30.0
):
    results = complex_operation(data)
```

## Common Patterns

### Sequential Pipeline

```python
@jit
class PipelineOperator(Operator):
    def __init__(self):
        self.op1 = Stage1Operator()
        self.op2 = Stage2Operator()
        self.op3 = Stage3Operator()
        
    def forward(self, *, inputs):
        stage1_result = self.op1(inputs=inputs)
        stage2_result = self.op2(inputs=stage1_result)
        stage3_result = self.op3(inputs=stage2_result)
        return stage3_result
```

### Parallel Branches

```python
@structural_jit
class ParallelBranchOperator(Operator):
    def __init__(self):
        self.branch1 = Branch1Operator()
        self.branch2 = Branch2Operator()
        self.merger = MergeOperator()
        
    def forward(self, *, inputs):
        # These two operations run in parallel
        branch1_result = self.branch1(inputs=inputs)
        branch2_result = self.branch2(inputs=inputs)
        
        # Merge the results
        final_result = self.merger(
            inputs={"branch1": branch1_result, "branch2": branch2_result}
        )
        return final_result
```

### Batch Processing

```python
def process_item(item):
    # Process a single item
    return processed_item

# Create a batched version
batch_processor = vmap(process_item)

# Process a batch at once
results = batch_processor([item1, item2, item3, item4])
```

## Next Steps

- See the [API Reference](API_REFERENCE.md) for detailed function documentation
- Read the [JIT Overview](JIT_OVERVIEW.md) for a comprehensive explanation of the different JIT approaches
- Check the [Architecture Overview](ARCHITECTURE.md) for system design details
- Read the [Performance Guide](PERFORMANCE_GUIDE.md) for optimization tips
- Explore the [Transforms Documentation](TRANSFORMS.md) for advanced parallelization