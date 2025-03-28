# XCS Transforms

XCS provides powerful transformation primitives that enable vectorization, parallelization, and distributed execution of computations. This document provides a detailed overview of the available transforms and their usage.

## Core Transform Concepts

Transforms in XCS follow these core principles:

1. **Function Transformation**: Transforms take functions as input and return transformed functions
2. **Semantic Preservation**: Transformed functions preserve the semantics of the original function
3. **Composability**: Transforms can be composed to create more complex transformations
4. **Explicit Configuration**: Transforms provide clear configuration options

## Available Transforms

### Vectorized Mapping (vmap)

The `vmap` transform enables automatic vectorization of functions to process batched inputs:

```python
from ember.api.xcs import vmap

def process_item(x):
    return x * 2

# Create a vectorized version
batch_process = vmap(process_item)

# Process multiple items at once
results = batch_process([1, 2, 3])  # [2, 4, 6]
```

#### Advanced vmap Features

**Handling Multiple Arguments**:

```python
def process_pairs(x, y):
    return x + y

# Vectorize over both arguments
vectorized = vmap(process_pairs)
result = vectorized([1, 2, 3], [10, 20, 30])  # [11, 22, 33]
```

**Specifying Axes**:

```python
# Vectorize only the first argument
vectorized = vmap(process_pairs, in_axes=(0, None))
result = vectorized([1, 2, 3], 10)  # [11, 12, 13]
```

**Nested Batching**:

```python
# Vectorize in two dimensions
double_vectorized = vmap(vmap(process_item))
result = double_vectorized([[1, 2], [3, 4]])  # [[2, 4], [6, 8]]
```

### Parallel Mapping (pmap)

The `pmap` transform enables parallel execution across multiple cores:

```python
from ember.api.xcs import pmap
import time

def slow_computation(x):
    time.sleep(1)  # Simulate work
    return x * 2

# Create a parallelized version
parallel_process = pmap(slow_computation)

# Process items in parallel (much faster than sequential)
results = parallel_process([1, 2, 3, 4])  # [2, 4, 6, 8]
```

#### Advanced pmap Features

**Controlling Worker Count**:

```python
# Specify number of worker threads
parallel_process = pmap(slow_computation, num_workers=4)
```

**Error Handling**:

```python
from ember.api.xcs import TransformOptions

# Configure error handling
options = TransformOptions(propagate_errors=True)
parallel_process = pmap(risky_function, options=options)
```

**Timeouts**:

```python
# Add timeout to prevent hanging
options = TransformOptions(timeout=5.0)
parallel_process = pmap(slow_computation, options=options)
```

### Structural JIT (structural_jit)

The `structural_jit` decorator analyzes operator structure and constructs an optimized execution graph:

```python
from typing import Dict, Any, ClassVar
from ember.xcs.tracer.structural_jit import structural_jit
from ember.core.registry.specification import Specification

# Define a composite operator with proper field declarations
@structural_jit
class CompositeOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    # Class-level specification
    specification: ClassVar[Specification] = CompositeSpecification()
    
    # Field declarations
    op1: SubOperator1
    op2: SubOperator2
    
    def __init__(self, *, config_param: str) -> None:
        # Initialize sub-operators
        self.op1 = SubOperator1(param=config_param)
        self.op2 = SubOperator2(param=config_param)
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Multi-step computation that gets optimized
        intermediate = self.op1(inputs=inputs)
        return self.op2(inputs=intermediate)
```

#### Advanced structural_jit Features

**Execution Strategy Control**:

```python
# Control the execution strategy
@structural_jit(execution_strategy="parallel", max_workers=8)
class ParallelPipeline(Operator[Dict[str, Any], Dict[str, Any]]):
    # Implementation...
```

**Caching Control**:

```python
# Control graph caching behavior
@structural_jit(cache_graph=True)
class CachedPipeline(Operator[Dict[str, Any], Dict[str, Any]]):
    # Implementation...
```

**Runtime Control**:

```python
# Create an optimized operator
pipeline = OptimizedPipeline()

# Disable JIT for debugging
pipeline.disable_jit()

# Re-enable JIT
pipeline.enable_jit()

# Clear graph cache to force recompilation
pipeline.clear_graph_cache()
```

### Mesh Sharding (mesh_sharded)

The `mesh_sharded` transform enables distributed execution across a device mesh:

```python
from ember.api.xcs import mesh_sharded, DeviceMesh, PartitionSpec

# Create a 2D device mesh
mesh = DeviceMesh(shape=(2, 2))

# Specify how to partition the data
pspec = PartitionSpec(0, 1)  # Partition along both dimensions

# Create a sharded function
sharded_fn = mesh_sharded(heavy_computation, mesh=mesh, partition_spec=pspec)

# Execute with data distributed across the mesh
result = sharded_fn(large_data)
```

#### Advanced Mesh Features

**Custom Device Specification**:

```python
# Specify devices explicitly
mesh = DeviceMesh(
    devices=["gpu:0", "gpu:1", "cpu:0", "cpu:1"],
    shape=(2, 2)
)
```

**Partial Sharding**:

```python
# Only shard along first dimension
pspec = PartitionSpec(0, None)
```

**Nested Meshes**:

```python
# Create hierarchical meshes
outer_mesh = DeviceMesh(shape=(2,))
inner_mesh = DeviceMesh(shape=(2,))

# Compose sharding transformations
outer_sharded = mesh_sharded(inner_sharded_fn, mesh=outer_mesh, partition_spec=PartitionSpec(0))
```

## Combining Transforms

Transforms can be combined to create more powerful transformations. This section provides detailed examples of common integration patterns.

### vmap + pmap (Batch Processing with Parallelism)

```python
from typing import Dict, Any, ClassVar
from ember.api.xcs import vmap, pmap
from ember.core.registry.specification import Specification

# Define an operator with proper field declarations
class TextProcessor(Operator[Dict[str, Any], Dict[str, Any]]):
    specification: ClassVar[Specification] = TextProcessorSpecification()
    model_name: str
    
    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = inputs.get("text", "")
        return {"processed": f"Processed with {self.model_name}: {text[:20]}..."}

# Create operator instance
processor = TextProcessor(model_name="nlp-model")

# First vectorize to handle batches
vectorized_processor = vmap(processor)

# Then parallelize to distribute across workers
distributed_processor = pmap(vectorized_processor, num_workers=4)

# Process a large batch
large_batch = {
    "text": [f"Document {i}" for i in range(100)],
    "options": {"format": "plain"}  # Non-batched parameter
}

# This will:
# 1. Divide the batch into chunks (pmap)
# 2. Process each chunk as a batch (vmap)
# 3. Combine all results
results = distributed_processor(inputs=large_batch)
```

### structural_jit + vmap (Optimized Batch Processing)

```python
from typing import Dict, Any, ClassVar
from ember.api.xcs import vmap
from ember.xcs.tracer.structural_jit import structural_jit
from ember.core.registry.specification import Specification

# Define component operators
class FeatureExtractor(Operator[Dict[str, Any], Dict[str, Any]]):
    specification: ClassVar[Specification] = ExtractorSpecification()
    extractor_type: str
    
    def __init__(self, *, extractor_type: str) -> None:
        self.extractor_type = extractor_type
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = inputs.get("text", "")
        return {"features": f"Features({self.extractor_type}): {text[:20]}..."}

class Classifier(Operator[Dict[str, Any], Dict[str, Any]]):
    specification: ClassVar[Specification] = ClassifierSpecification()
    model_name: str
    threshold: float
    
    def __init__(self, *, model_name: str, threshold: float = 0.5) -> None:
        self.model_name = model_name
        self.threshold = threshold
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        features = inputs.get("features", "")
        return {"classification": f"Classified({self.model_name}, {self.threshold}): {features}"}

# Define complex pipeline with optimization
@structural_jit(execution_strategy="auto", max_workers=8)
class ClassificationPipeline(Operator[Dict[str, Any], Dict[str, Any]]):
    specification: ClassVar[Specification] = PipelineSpecification()
    extractor: FeatureExtractor
    classifier: Classifier
    
    def __init__(self, *, model_name: str) -> None:
        self.extractor = FeatureExtractor(extractor_type="advanced")
        self.classifier = Classifier(model_name=model_name)
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # This execution flow gets optimized by structural_jit
        features = self.extractor(inputs=inputs)
        return self.classifier(inputs=features)

# Create optimized pipeline
pipeline = ClassificationPipeline(model_name="bert-classifier")

# Add batch processing capability
vectorized_pipeline = vmap(pipeline)

# Process a batch through the optimized pipeline
batch_inputs = {
    "text": [
        "This product is excellent!",
        "Service was terrible.",
        "Average experience, could be better."
    ]
}

# This will:
# 1. Use the optimized execution graph from structural_jit
# 2. Apply that optimized execution to each item in the batch
results = vectorized_pipeline(inputs=batch_inputs)
```

### structural_jit + pmap (Distributed Complex Processing)

```python
from typing import Dict, Any, ClassVar
from ember.xcs.tracer.structural_jit import structural_jit
from ember.api.xcs import pmap
from ember.core.registry.specification import Specification

# Define a complex operator with optimization
@structural_jit(execution_strategy="parallel")
class ProcessingPipeline(Operator[Dict[str, Any], Dict[str, Any]]):
    specification: ClassVar[Specification] = ProcessingSpecification()
    stage1: Stage1Operator
    stage2: Stage2Operator
    stage3: Stage3Operator
    
    def __init__(self, *, config: str) -> None:
        self.stage1 = Stage1Operator(config=config)
        self.stage2 = Stage2Operator(config=config)
        self.stage3 = Stage3Operator(config=config)
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Multi-stage execution optimized by structural_jit
        intermediate1 = self.stage1(inputs=inputs)
        intermediate2 = self.stage2(inputs=intermediate1)
        return self.stage3(inputs=intermediate2)

# Create pipeline
pipeline = ProcessingPipeline(config="high-throughput")

# Parallelize the entire pipeline
parallel_pipeline = pmap(pipeline, num_workers=4)

# Process multiple items through distributed pipelines
batch_inputs = {
    "data": [f"Data{i}" for i in range(16)],
    "config": {"quality": "high"}  # Shared config
}

# This will:
# 1. Distribute items across workers (pmap)
# 2. Each worker processes items using the optimized graph (structural_jit)
results = parallel_pipeline(inputs=batch_inputs)
```

### Integration with Execution Options

Use execution options to control transform behavior at runtime:

```python
from ember.api.xcs import vmap, pmap
from ember.xcs.engine.execution_options import execution_options
from ember.xcs.tracer.structural_jit import structural_jit

# Define optimized operator
@structural_jit
class ComplexOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    # Implementation...

# Apply transforms
op = ComplexOperator()
vectorized_op = vmap(op)

# Fine-tune execution with options
with execution_options(
    use_parallel=True,         # Enable parallel execution
    max_workers=8,             # Set worker count
    enable_caching=True,       # Cache intermediate results
    timeout_seconds=30.0,      # Set timeout
    device_strategy="auto"     # Auto device selection
):
    results = vectorized_op(inputs=batch_inputs)
```

### Transformation Order

The order of transforms matters and affects performance characteristics:

- `pmap(vmap(f))`: Each worker processes a batch (often more efficient)
  - Lower overhead as each worker handles multiple items
  - Better for memory-bound operations
  - Good for large batches with moderate per-item processing

- `vmap(pmap(f))`: Each element processed in parallel (higher parallelism)
  - Higher overhead from more worker coordination
  - Better for compute-bound operations
  - Good for small batches with intensive per-item processing

### Integration with JIT System

Transforms can be combined with Ember's JIT system for additional optimizations:

```python
from ember.api.xcs import jit, vmap

# JIT-compiled vectorized function
@jit
def process_item(x):
    return expensive_computation(x)

# Vectorized version with JIT optimization
batch_process = vmap(process_item)
```

The relationship between transforms and the different JIT approaches (jit, structural_jit, autograph) is described in more detail in [JIT Overview](JIT_OVERVIEW.md).

## Performance Optimization Patterns

### Memory-Efficient Processing for Large Datasets

```python
from ember.api.xcs import vmap
from ember.xcs.engine.execution_options import execution_options

def process_in_chunks(large_dataset, chunk_size=64):
    vectorized_op = vmap(base_operator)
    all_results = []
    
    # Process in controlled-size chunks to manage memory
    for i in range(0, len(large_dataset), chunk_size):
        chunk = large_dataset[i:i+chunk_size]
        
        # Control parallelism and caching based on chunk
        with execution_options(max_workers=4, enable_caching=(i == 0)):
            # First chunk can be cached as a template
            chunk_results = vectorized_op(inputs={"data": chunk})
            all_results.append(chunk_results)
    
    # Combine all results
    return combine_results(all_results)
```

### JIT Warm-up for Latency-Sensitive Applications

```python
from ember.xcs.tracer.structural_jit import structural_jit

@structural_jit
class CriticalPipeline(Operator[Dict[str, Any], Dict[str, Any]]):
    # Implementation...

# Application startup: warm up the JIT compilation
pipeline = CriticalPipeline()
_ = pipeline(inputs=warm_up_sample)  # Compilation happens here

# Later, during time-critical operation
def process_critical_request(request_data):
    # Uses cached compilation, minimal latency
    return pipeline(inputs=request_data)
```

## Transform Implementation Details

### vmap Implementation

The `vmap` transform works by:

1. Splitting the input batch into individual elements
2. Applying the original function to each element
3. Combining the results into a batched output

For optimization, it:
- Uses vectorized operations when available
- Employs batch processing primitives
- Handles nested data structures correctly

### pmap Implementation

The `pmap` transform works by:

1. Creating a thread pool of worker threads
2. Dividing work among workers
3. Executing the function on each worker
4. Combining results in the original order

It automatically handles:
- Thread creation and management
- Work distribution
- Result aggregation
- Error propagation

### structural_jit Implementation

The `structural_jit` transform works by:

1. Analyzing the operator's structure through attributes and composition
2. Building a directed graph of operations
3. Detecting execution dependencies
4. Creating an optimized execution plan based on the graph
5. Caching the plan for repeated execution

It provides:
- Dependency-aware execution
- Adaptive parallelization
- Operation reordering when safe
- Graph caching for performance

### mesh_sharded Implementation

The `mesh_sharded` transform works by:

1. Partitioning input data according to the partition spec
2. Mapping partitions to devices in the mesh
3. Executing the function on each device with its partition
4. Gathering and combining the results

## Best Practices

### Transform Selection Guide

| Scenario | Recommended Transform | Configuration |
|----------|------------------------|---------------|
| Batch vectorization | `vmap` | Use `in_axes` to control batching dimensions |
| Parallel execution | `pmap` | Set `num_workers` based on CPU cores |
| Complex operator optimization | `structural_jit` | Choose appropriate `execution_strategy` |
| Distributing computation | `mesh_sharded` | Configure `PartitionSpec` for data distribution |
| Large dataset processing | `vmap` + `pmap` | Apply `pmap(vmap(f))` with chunking |
| Memory-constrained environments | Chunked processing | Process in smaller batches with controlled workers |

### General Recommendations

1. **Start Simple**: Begin with individual transforms before combining them.

2. **Benchmark Configurations**: Test different transform combinations and parameters.

3. **Match to Hardware**: 
   - For CPU-bound tasks: Use `pmap` with `num_workers` set to physical core count
   - For memory-bound tasks: Use controlled batch sizes with `vmap`
   - For complex pipelines: Use `structural_jit` with appropriate execution strategy

4. **Consider Overhead**: 
   - Transforming very small operations may not be worth the overhead
   - For simple operations, batch processing may be more efficient than parallelism

5. **Control Resources**:
   - Use execution options to limit worker count for shared systems
   - Set appropriate timeouts for production environments
   - Enable caching selectively for repetitive operations

6. **Debug Transformed Code**:
   - Use `disable_structural_jit()` to temporarily disable JIT for debugging
   - Enable `trace_execution=True` to get execution information
   - Fall back to sequential execution to isolate issues

7. **Layer Transforms Appropriately**:
   - Optimize with `structural_jit` first for complex operations
   - Apply `vmap` at the appropriate level (whole pipeline vs. components)
   - Use `pmap` at the outermost level when possible to minimize thread creation

### Implementation Details and Performance Tuning

For more information on performance optimization with transforms, see [Performance Guide](PERFORMANCE_GUIDE.md).

For detailed execution control options, see [Execution Options](EXECUTION_OPTIONS.md).

For information about the JIT system that powers structural_jit, see [JIT Overview](JIT_OVERVIEW.md).