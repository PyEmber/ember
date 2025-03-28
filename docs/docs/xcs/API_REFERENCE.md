# XCS API Reference

This document provides detailed API references for the XCS module.

## Core Functions

### jit

```python
from ember.api.xcs import jit

@jit(options=None)
class MyOperator(Operator):
    def forward(self, *, inputs):
        return processed_inputs
```

The `jit` decorator enables just-in-time compilation for operators. It traces execution and compiles optimized execution plans by analyzing the actual runtime behavior of the operator.

**Parameters:**

- `options` (Optional[JITOptions]): Configuration options for JIT compilation
- `sample_input` (Dict[str, Any]): Sample input for pre-compilation during initialization
- `force_trace` (bool): When True, always trace execution even for cached graphs
- `recursive` (bool): Whether to trace and compile nested operator calls

**Options:**

- `cache_size` (int): Maximum number of cached execution plans (default: 128)
- `sample_input` (Dict[str, Any]): Sample input for precompilation
- `trace_level` (str): Tracing detail level ("minimal", "standard", "verbose")
- `fallback` (bool): Whether to fall back to original function on errors

**When to use:**
- For most operator optimization needs
- When execution patterns can vary based on inputs
- When you need to optimize based on actual runtime behavior

### structural_jit

```python
from ember.api.xcs import structural_jit

@structural_jit(execution_strategy="parallel")
class CompositeOperator(Operator):
    def __init__(self):
        self.op1 = SubOperator1()
        self.op2 = SubOperator2()
        
    def forward(self, *, inputs):
        intermediate = self.op1(inputs=inputs)
        result = self.op2(inputs=intermediate)
        return result
```

The `structural_jit` decorator analyzes operator structure directly to optimize execution without requiring execution traces. It examines the composition relationships between operators to identify optimization opportunities.

**Parameters:**

- `execution_strategy` (str): Execution strategy ("auto", "parallel", "sequential")
- `parallel_threshold` (int): Minimum number of nodes to trigger parallel execution in auto mode
- `max_workers` (Optional[int]): Maximum number of worker threads for parallel execution
- `cache_graph` (bool): Whether to cache and reuse the compiled graph

**When to use:**
- For complex composite operators with many subcomponents
- When operator structure is known and static
- For maximum optimization of operator composition
- To parallelize independent operations in composite operators

### autograph

```python
from ember.api.xcs import autograph, execute

with autograph() as graph:
    result1 = op1(inputs={"query": "Example"})
    result2 = op2(inputs=result1)
    
results = execute(graph)
```

The `autograph` context manager provides explicit control over graph construction by recording operator calls to build a computational graph. Unlike the decorators, this approach requires manual graph building and execution.

**Returns:**

- `XCSGraph`: A computational graph representing the recorded operations

**When to use:**
- When you need explicit control over graph construction
- For debugging execution paths
- When you want to construct a graph once and execute it multiple times
- To create execution graphs for visualization or analysis

> **Note:** For a comprehensive comparison and detailed explanation of the relationship between these approaches, see [JIT_OVERVIEW.md](JIT_OVERVIEW.md).

### execute

```python
from ember.api.xcs import execute

results = execute(graph, inputs={"query": "Example"})
```

Executes a computational graph with the specified inputs.

**Parameters:**

- `graph` (XCSGraph): The graph to execute
- `inputs` (Dict[str, Any]): Input values for the graph
- `options` (Optional[XCSExecutionOptions]): Execution configuration

**Returns:**

- `Dict[str, Any]`: The results of graph execution

## Transforms

### vmap

```python
from ember.api.xcs import vmap

# Basic usage
batch_fn = vmap(single_item_function)
results = batch_fn([item1, item2, item3])

# With axis specification
batch_fn = vmap(function, in_axes=(0, None))
results = batch_fn(batch_inputs, constant_arg)
```

Vectorizes a function to process batched inputs in parallel.

**Parameters:**

- `func` (Callable): The function to vectorize
- `in_axes` (Tuple[Optional[int], ...]): Input axes to vectorize (None for constants)
- `out_axes` (Optional[int]): Output axis for batched results (default: 0)
- `options` (Optional[TransformOptions]): Additional configuration

**Returns:**

- `Callable`: A vectorized version of the input function

### pmap

```python
from ember.api.xcs import pmap

parallel_fn = pmap(heavy_computation)
results = parallel_fn(large_dataset)
```

Parallelizes a function to execute across multiple cores or devices.

**Parameters:**

- `func` (Callable): The function to parallelize
- `num_workers` (Optional[int]): Number of worker threads (default: auto)
- `options` (Optional[TransformOptions]): Additional configuration

**Returns:**

- `Callable`: A parallelized version of the input function

### mesh_sharded

```python
from ember.api.xcs import mesh_sharded, DeviceMesh, PartitionSpec

# Create a device mesh
mesh = DeviceMesh(shape=(2, 2))

# Define the partition spec
pspec = PartitionSpec(0, 1)

# Create a sharded function
sharded_fn = mesh_sharded(function, mesh=mesh, partition_spec=pspec)
results = sharded_fn(inputs)
```

Executes a function with inputs sharded across a device mesh.

**Parameters:**

- `func` (Callable): The function to shard
- `mesh` (DeviceMesh): The device mesh to use
- `partition_spec` (PartitionSpec): How to partition inputs across the mesh
- `options` (Optional[TransformOptions]): Additional configuration

**Returns:**

- `Callable`: A sharded version of the input function

## Configuration Types

### XCSExecutionOptions

```python
from ember.api.xcs import XCSExecutionOptions

with XCSExecutionOptions(
    scheduler="parallel",
    max_workers=4,
    timeout=30.0
):
    results = complex_operation(data)
```

Configuration options for XCS execution.

**Parameters:**

- `scheduler` (str): Scheduler to use ("auto", "parallel", "sequential")
- `max_workers` (Optional[int]): Maximum number of worker threads
- `timeout` (Optional[float]): Execution timeout in seconds
- `trace_execution` (bool): Whether to trace execution for debugging
- `fail_fast` (bool): Whether to stop on first error

### JITOptions

```python
from ember.api.xcs import JITOptions

options = JITOptions(
    sample_input={"query": "example"},
    cache_size=100
)
```

Configuration options for JIT compilation.

**Parameters:**

- `cache_size` (int): Maximum number of cached execution plans
- `sample_input` (Dict[str, Any]): Sample input for precompilation
- `trace_level` (str): Tracing detail level
- `fallback` (bool): Whether to fall back to original function on errors

### TransformOptions

```python
from ember.api.xcs import TransformOptions

options = TransformOptions(
    propagate_errors=True,
    timeout=10.0
)
```

Configuration options for transformations.

**Parameters:**

- `propagate_errors` (bool): Whether to propagate errors or catch them
- `timeout` (Optional[float]): Execution timeout in seconds

## Utility Types

### DeviceMesh

```python
from ember.api.xcs import DeviceMesh

# Create a 2x2 mesh
mesh = DeviceMesh(shape=(2, 2))

# Create a mesh with specific devices
mesh = DeviceMesh(devices=["gpu:0", "gpu:1", "gpu:2", "gpu:3"], shape=(2, 2))
```

Represents a logical grid of devices for distributed computation.

**Parameters:**

- `devices` (Optional[List[str]]): List of device identifiers
- `shape` (Optional[Tuple[int, ...]]): Logical shape of the mesh

### PartitionSpec

```python
from ember.api.xcs import PartitionSpec

# Partition along first dimension
pspec = PartitionSpec(0)

# Partition along first and second dimensions
pspec = PartitionSpec(0, 1)
```

Specifies how data should be partitioned across a device mesh.

**Parameters:**

- `*axes` (int): Dimensions to partition along