# Ember XCS: High-Performance Execution Framework

The Ember XCS (Accelerated Compound Systems) module provides a high-performance execution framework for computational graphs, enabling intelligent scheduling, just-in-time tracing, and advanced parallel execution transformations.

## Core Features

- **Just-In-Time Compilation**: Automatically optimize operator execution paths
- **Intelligent Graph Building**: Create and execute computational graphs
- **Parallelization Transforms**: Vectorize and parallelize operations
- **Structural Analysis**: Analyze operator structures for optimized execution 
- **Distributed Computing**: Support for mesh-based distribution

## Architecture

XCS is designed with a clean, modular architecture consisting of these key components:

- **Tracer**: JIT compilation and structural analysis
- **Engine**: Efficient execution scheduling and runtime management
- **Graph**: Computational graph representation and manipulation
- **Transforms**: Function transformations for vectorization and parallelization
- **Utils**: Common utility functions and tree manipulation tools

The system is built on functional programming principles and follows SOLID design patterns throughout its implementation.

For a detailed explanation of the JIT system architecture and the relationship between autograph, jit, and structural_jit, see [JIT Overview](JIT_OVERVIEW.md).

## Usage Examples

### Just-In-Time Compilation

```python
from ember.api.xcs import jit

@jit
class MyOperator(Operator):
    def forward(self, *, inputs):
        # Complex computation
        return {"result": processed_data}
```

### Automatic Graph Building

```python
from ember.api.xcs import autograph, execute

with autograph() as graph:
    # Operations are traced, not executed
    result1 = op1(inputs={"query": "Example"})
    result2 = op2(inputs=result1)
    
# Execute the graph with optimized scheduling
results = execute(graph)
```

### Function Transforms

```python
from ember.api.xcs import vmap, pmap

# Vectorize a function to process batches
batch_fn = vmap(single_item_function)
batch_results = batch_fn([item1, item2, item3])

# Parallelize across multiple cores/devices
parallel_fn = pmap(heavy_computation)
parallel_results = parallel_fn(large_dataset)
```

### Structural JIT

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

### Advanced Configuration

```python
from ember.api.xcs import XCSExecutionOptions, JITOptions

# Configure JIT compilation with precompilation
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

## Simplified Imports

XCS provides a clean, simplified import structure through the `ember.api.xcs` module. All key functionality is available with short, intuitive imports:

```python
from ember.api.xcs import jit, vmap, pmap, autograph, execute, structural_jit
```

For more details, see [Simplified Imports](SIMPLIFIED_IMPORTS.md).

## Additional Resources

- Check the [examples directory](../../src/ember/examples/xcs) for practical demonstrations
- See the source code for implementation details
- Refer to the project README for overall framework information