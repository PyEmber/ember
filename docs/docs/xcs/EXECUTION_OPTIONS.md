# Execution Options Guide

## Overview

The execution options system in Ember XCS allows you to control how operations are executed, including parallelism settings, device selection, and scheduling strategies. This guide explains how to use execution options effectively in your applications.

## Basic Usage

Execution options can be applied in two ways:

1. As a temporary context using the `execution_options` context manager
2. As a global setting using `set_execution_options`

### Context Manager (Recommended)

Using the context manager is the recommended approach for most use cases, as it ensures options are only applied within a specific scope and automatically restored afterward:

```python
from ember.xcs.engine.execution_options import execution_options

# Run with parallel execution and 4 workers
with execution_options(use_parallel=True, max_workers=4):
    result = vectorized_op(inputs={"prompt": prompt, "seed": seeds})
    
# Run with sequential execution
with execution_options(use_parallel=False):
    result = vectorized_op(inputs={"prompt": prompt, "seed": seeds})
```

### Global Settings

For cases where you want to set execution options for an entire application, you can use the `set_execution_options` function:

```python
from ember.xcs.engine.execution_options import set_execution_options

# Set global execution options
set_execution_options(use_parallel=True, max_workers=8)
```

You can retrieve the current execution options with `get_execution_options()`.

## Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_parallel` | `bool` | `True` | Whether to use parallel execution where possible |
| `max_workers` | `Optional[int]` | `None` | Maximum number of worker threads for parallel execution |
| `device_strategy` | `str` | `"auto"` | Strategy for device selection ('auto', 'cpu', 'gpu', etc.) |
| `enable_caching` | `bool` | `False` | Whether to cache intermediate results |
| `trace_execution` | `bool` | `False` | Whether to trace execution for debugging |
| `timeout_seconds` | `Optional[float]` | `None` | Maximum execution time in seconds before timeout |
| `scheduler` | `Optional[str]` | `None` | Legacy parameter for backward compatibility |

## Integration with XCS Transforms

### vmap

Execution options are particularly useful when combined with vmap for controlling how batched operations are processed:

```python
from typing import Dict, Any, ClassVar
from ember.xcs.transforms.vmap import vmap
from ember.xcs.engine.execution_options import execution_options
from ember.core.registry.specification import Specification

# Define a single-item operator with proper field declarations
class TextProcessor(Operator[Dict[str, Any], Dict[str, Any]]):
    # Class-level specification
    specification: ClassVar[Specification] = TextProcessorSpecification()
    
    # Field declarations
    model_name: str
    max_length: int
    
    def __init__(self, *, model_name: str, max_length: int = 100) -> None:
        self.model_name = model_name
        self.max_length = max_length
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = inputs.get("text", "")
        return {"processed": f"Processed: {text[:self.max_length]}"}

# Create vectorized version of the operator
processor = TextProcessor(model_name="text-processor")
vectorized_processor = vmap(processor)

# Process a batch with parallel execution
with execution_options(use_parallel=True, max_workers=4):
    batch_result = vectorized_processor(inputs={
        "text": ["Sample text 1", "Sample text 2", "Sample text 3"],
        "options": {"format": "plain"}  # Non-batched parameter applied to all items
    })
```

### pmap

For pmap, execution options control the underlying execution behavior:

```python
from typing import Dict, Any, ClassVar
from ember.xcs.transforms.pmap import pmap
from ember.xcs.engine.execution_options import execution_options
from ember.core.registry.specification import Specification

# Define an operator with proper field declarations
class DocumentAnalyzer(Operator[Dict[str, Any], Dict[str, Any]]):
    # Class-level specification
    specification: ClassVar[Specification] = AnalyzerSpecification()
    
    # Field declarations
    analyzer_type: str
    depth: int
    
    def __init__(self, *, analyzer_type: str = "basic", depth: int = 3) -> None:
        self.analyzer_type = analyzer_type
        self.depth = depth
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        # Analyze documents
        return {"analysis": [f"Analysis of {doc}" for doc in documents]}

# Create parallelized version with specific worker count
analyzer = DocumentAnalyzer(analyzer_type="comprehensive")
parallelized_analyzer = pmap(analyzer, num_workers=4)

# execution_options can still affect other aspects of execution
with execution_options(timeout_seconds=10.0, enable_caching=True):
    result = parallelized_analyzer(inputs={"documents": ["doc1.txt", "doc2.txt", "doc3.txt"]})
```

### structural_jit

The structural JIT decorator analyzes operator composition and optimizes execution using the XCS graph system:

```python
from typing import Dict, Any, ClassVar
from ember.xcs.tracer.structural_jit import structural_jit
from ember.xcs.engine.execution_options import execution_options
from ember.core.registry.specification import Specification

# Define sub-operators with proper field declarations
class ExtractorOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    # Class-level specification
    specification: ClassVar[Specification] = ExtractorSpecification()
    
    # Field declarations
    extractor_type: str
    
    def __init__(self, *, extractor_type: str = "entities") -> None:
        self.extractor_type = extractor_type
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = inputs.get("text", "")
        return {"extracted": f"Extracted {self.extractor_type} from: {text}"}

class ClassifierOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    # Class-level specification
    specification: ClassVar[Specification] = ClassifierSpecification()
    
    # Field declarations
    model_name: str
    threshold: float
    
    def __init__(self, *, model_name: str, threshold: float = 0.5) -> None:
        self.model_name = model_name
        self.threshold = threshold
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        extracted = inputs.get("extracted", "")
        return {"classification": f"Classified with {self.model_name}: {extracted}"}

# Define a JIT-optimized composite operator
@structural_jit
class NLPPipeline(Operator[Dict[str, Any], Dict[str, Any]]):
    # Class-level specification
    specification: ClassVar[Specification] = PipelineSpecification()
    
    # Field declarations for sub-operators
    extractor: ExtractorOperator
    classifier: ClassifierOperator
    
    def __init__(self, *, model_name: str = "default") -> None:
        # Create sub-operators
        self.extractor = ExtractorOperator(extractor_type="entities")
        self.classifier = ClassifierOperator(model_name=model_name)
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Connect operators in the pipeline
        extracted = self.extractor(inputs=inputs)
        return self.classifier(inputs=extracted)

# Execution options affect the graph execution strategy
pipeline = NLPPipeline(model_name="advanced-classifier")
with execution_options(use_parallel=True, max_workers=8):
    result = pipeline(inputs={"text": "Sample document text for analysis"})
```

## Combining Transforms

Ember XCS transforms can be combined for powerful computation patterns. Here are common integration patterns:

### vmap + structural_jit

Combine batch processing with graph optimization:

```python
from typing import Dict, Any, ClassVar
from ember.xcs.transforms.vmap import vmap
from ember.xcs.tracer.structural_jit import structural_jit
from ember.xcs.engine.execution_options import execution_options
from ember.core.registry.specification import Specification

# Define simple operators with proper field declarations
class FeatureExtractor(Operator[Dict[str, Any], Dict[str, Any]]):
    # Class-level specification
    specification: ClassVar[Specification] = FeatureExtractorSpecification()
    
    # Field declarations
    feature_type: str
    
    def __init__(self, *, feature_type: str = "basic") -> None:
        self.feature_type = feature_type
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = inputs.get("text", "")
        return {"features": f"Features({self.feature_type}): {text[:20]}..."}

class SentimentAnalyzer(Operator[Dict[str, Any], Dict[str, Any]]):
    # Class-level specification
    specification: ClassVar[Specification] = SentimentAnalyzerSpecification()
    
    # Field declarations
    model_name: str
    
    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        features = inputs.get("features", "")
        return {"sentiment": f"Sentiment({self.model_name}): {features}"}

# Define a JIT-optimized composite operator
@structural_jit
class TextAnalysisPipeline(Operator[Dict[str, Any], Dict[str, Any]]):
    # Class-level specification
    specification: ClassVar[Specification] = AnalysisPipelineSpecification()
    
    # Field declarations for sub-operators
    extractor: FeatureExtractor
    analyzer: SentimentAnalyzer
    
    def __init__(self, *, feature_type: str = "advanced", model_name: str = "sentiment-v2") -> None:
        # Create sub-operators
        self.extractor = FeatureExtractor(feature_type=feature_type)
        self.analyzer = SentimentAnalyzer(model_name=model_name)
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Connect operators in the pipeline
        features = self.extractor(inputs=inputs)
        return self.analyzer(inputs=features)

# Create a pipeline instance
pipeline = TextAnalysisPipeline(feature_type="comprehensive", model_name="sentiment-v3")

# Create a vectorized version of the JIT-optimized pipeline
vectorized_pipeline = vmap(pipeline)

# Process a batch with execution options controlling both vectorization and JIT
with execution_options(use_parallel=True, max_workers=8, enable_caching=True):
    results = vectorized_pipeline(inputs={
        "text": [
            "This product exceeded my expectations!",
            "The service was terrible and I want a refund.",
            "Average experience, neither good nor bad."
        ],
        "include_details": True  # Non-batched parameter applied to all items
    })
```

### pmap + vmap (Nested Parallelism)

Parallelize across devices and batch within each device:

```python
from typing import Dict, Any, ClassVar
from ember.xcs.transforms.vmap import vmap
from ember.xcs.transforms.pmap import pmap
from ember.xcs.engine.execution_options import execution_options
from ember.core.registry.specification import Specification

# Define an operator with proper field declarations
class ImageProcessor(Operator[Dict[str, Any], Dict[str, Any]]):
    # Class-level specification
    specification: ClassVar[Specification] = ImageProcessorSpecification()
    
    # Field declarations
    model_name: str
    resolution: str
    
    def __init__(self, *, model_name: str, resolution: str = "high") -> None:
        self.model_name = model_name
        self.resolution = resolution
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Process a single image
        image_data = inputs.get("image", "")
        return {
            "processed": f"Processed({self.model_name}, {self.resolution}): {image_data[:10]}...",
            "metadata": {"format": "png", "size": len(image_data)}
        }

# Create a processor instance
processor = ImageProcessor(model_name="image-model-v2", resolution="medium")

# First apply vmap to handle batches within each worker
vectorized_processor = vmap(processor)

# Then apply pmap to distribute batches across workers
distributed_processor = pmap(vectorized_processor, num_workers=4)

# Structure data as a large batch of images
large_batch = {
    "image": [f"image_data_{i}" for i in range(100)],
    "options": {"format": "json"}  # Non-batched parameter for all images
}

# Execute with additional options
with execution_options(enable_caching=True, timeout_seconds=30.0, device_strategy="auto"):
    # pmap will divide the batch into chunks, and vmap will process each chunk as a batch
    results = distributed_processor(inputs=large_batch)
```

### structural_jit + pmap

Optimize complex operators and distribute them across workers:

```python
from typing import Dict, Any, ClassVar
from ember.xcs.tracer.structural_jit import structural_jit
from ember.xcs.transforms.pmap import pmap
from ember.xcs.engine.execution_options import execution_options
from ember.core.registry.specification import Specification

# Define sub-operators with proper field declarations
class DataPreprocessor(Operator[Dict[str, Any], Dict[str, Any]]):
    # Class-level specification
    specification: ClassVar[Specification] = PreprocessorSpecification()
    
    # Field declarations
    preprocessor_type: str
    
    def __init__(self, *, preprocessor_type: str = "standard") -> None:
        self.preprocessor_type = preprocessor_type
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        data = inputs.get("data", "")
        return {"preprocessed": f"Preprocessed({self.preprocessor_type}): {data[:10]}..."}

class ModelInferencer(Operator[Dict[str, Any], Dict[str, Any]]):
    # Class-level specification
    specification: ClassVar[Specification] = InferencerSpecification()
    
    # Field declarations
    model_name: str
    precision: str
    
    def __init__(self, *, model_name: str, precision: str = "float32") -> None:
        self.model_name = model_name
        self.precision = precision
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        preprocessed = inputs.get("preprocessed", "")
        return {"prediction": f"Prediction({self.model_name}, {self.precision}): {preprocessed}"}

class PostProcessor(Operator[Dict[str, Any], Dict[str, Any]]):
    # Class-level specification
    specification: ClassVar[Specification] = PostProcessorSpecification()
    
    # Field declarations
    formatter: str
    
    def __init__(self, *, formatter: str = "json") -> None:
        self.formatter = formatter
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prediction = inputs.get("prediction", "")
        return {"result": f"Result({self.formatter}): {prediction}"}

# Define a complex JIT-optimized operator with multiple stages
@structural_jit
class InferencePipeline(Operator[Dict[str, Any], Dict[str, Any]]):
    # Class-level specification
    specification: ClassVar[Specification] = InferencePipelineSpecification()
    
    # Field declarations for sub-operators
    preprocessor: DataPreprocessor
    inferencer: ModelInferencer
    postprocessor: PostProcessor
    
    def __init__(self, *, model_name: str, preprocessor_type: str = "advanced") -> None:
        # Create sub-operators
        self.preprocessor = DataPreprocessor(preprocessor_type=preprocessor_type)
        self.inferencer = ModelInferencer(model_name=model_name)
        self.postprocessor = PostProcessor(formatter="structured")
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Connect operators in the pipeline
        preprocessed = self.preprocessor(inputs=inputs)
        predicted = self.inferencer(inputs=preprocessed)
        return self.postprocessor(inputs=predicted)

# Create a pipeline instance
pipeline = InferencePipeline(model_name="complex-model", preprocessor_type="enhanced")

# Create a distributed version to process multiple items in parallel
distributed_pipeline = pmap(pipeline, num_workers=8)

# Prepare a batch of data items
batch_data = {
    "data": [f"data_sample_{i}" for i in range(50)],
    "config": {"option1": "value1", "option2": "value2"}  # Shared configuration
}

# Execute with device targeting and tracing for performance analysis
with execution_options(device_strategy="gpu", trace_execution=True):
    results = distributed_pipeline(inputs=batch_data)
```

## Performance Considerations

When combining transforms and using execution options, keep these factors in mind:

1. **Worker Count vs. Batch Size**: Tune `max_workers` based on your hardware and batch size. Too many workers for small batches can increase overhead without benefit.

2. **Nested Parallelism Overhead**: When combining `vmap` and `pmap`, be aware of potential thread contention. The total number of threads can grow quickly with the formula:
   ```
   total_threads = pmap_workers × vmap_parallel_operations
   ```

3. **JIT Warm-up**: The first call to a JIT-optimized function includes compilation overhead. Subsequent calls benefit from the cached optimization. For critical applications, consider:
   ```python
   # Warm up the JIT compilation with a small batch
   pipeline(inputs=sample_input)  # Compilation happens here
   
   # Now process the real data
   with execution_options(use_parallel=True):
       results = pipeline(inputs=actual_input)  # Uses cached compilation
   ```

4. **Memory Utilization**: Larger batches with parallel execution increase memory usage. For memory-intensive operations, consider using smaller batches or fewer workers:
   ```python
   # Memory-efficient processing with controlled batch size
   with execution_options(max_workers=4):
       for mini_batch in chunk_data(large_data, size=16):
           partial_results = vectorized_op(inputs=mini_batch)
           process_results(partial_results)
   ```

5. **Caching Trade-offs**: Enabling `enable_caching` can improve performance for repeated operations but increases memory usage. Consider selectively enabling it:
   ```python
   # Enable caching for the expensive inference step
   with execution_options(enable_caching=True):
       embeddings = embedding_model(inputs=texts)
   
   # Disable caching for the less expensive post-processing
   with execution_options(enable_caching=False):
       processed = post_processor(inputs=embeddings)
   ```

## Transform Selection Guide

| Need | Transform | Configuration |
|------|-----------|---------------|
| Process batched inputs efficiently | `vmap` | `in_axes` to specify batch dimensions |
| Parallelize across cores | `pmap` | `num_workers` based on CPU cores |
| Optimize complex operator graphs | `structural_jit` | `execution_strategy="auto"` for adaptive execution |
| Distribute batches across workers | `pmap(vmap(...))` | Tune `max_workers` and `in_axes` |
| Optimize then distribute | `pmap(structural_jit(...))` | Use `enable_caching=True` for repeated execution |

## Best Practices

1. **Prefer the context manager** over global settings to limit the scope of changes.

2. **Match workers to workload and hardware**:  
   Set `max_workers` based on your workload characteristics and available CPU cores. A good starting point is the number of physical cores for compute-bound tasks, or more for I/O-bound tasks.

3. **Consider caching for repeated operations**:  
   Enable caching when the same operation is performed multiple times with identical inputs.

4. **Set appropriate timeouts**:  
   Use `timeout_seconds` to prevent operations from running indefinitely, especially in production environments.

5. **Benchmark different settings**:  
   Test different combinations of options to find the optimal configuration for your specific use case.

6. **Choose the right transform combination**:
   - Use `vmap` for vectorizing operations on batched inputs
   - Use `pmap` for parallel execution across workers
   - Use `structural_jit` for optimizing complex operator compositions
   - Combine transforms based on workload characteristics

## Legacy Support

For compatibility with older code, the `scheduler="sequential"` option is supported and will automatically set `use_parallel=False`. This ensures older code continues to work without modification.

```python
# Legacy approach (still works)
with execution_options(scheduler="sequential"):
    result = vectorized_op(inputs={"prompt": prompt, "seed": seeds})
    
# Modern approach (preferred)
with execution_options(use_parallel=False):
    result = vectorized_op(inputs={"prompt": prompt, "seed": seeds})
```

## Exception Handling

Invalid option names will raise a `ValueError` with a message indicating the invalid option. This helps catch configuration errors early in development:

```python
try:
    with execution_options(invalid_option=True):
        result = op(inputs=data)
except ValueError as e:
    print(f"Configuration error: {e}")  # Output: Configuration error: Invalid execution option: invalid_option
```