# Ember Operators - Quickstart Guide

This guide introduces Ember's operator system - the core building blocks for constructing AI workflows. Operators in Ember are conceptually similar to PyTorch modules, providing a flexible and composable way to build complex processing pipelines.

## 1. Operator Fundamentals

Operators are stateful, typed computation units that:
- Take structured inputs and produce structured outputs
- Can be composed into complex graphs
- Are automatically parallelizable when possible
- Provide clear typing guarantees with EmberModel

```python
from typing import List
from ember.core.registry.operator.base import Operator
from ember.core.types.ember_model import EmberModel

# Define input/output types
class ClassifierInput(EmberModel):
    text: str
    categories: List[str]

class ClassifierOutput(EmberModel):
    category: str
    confidence: float
```

## 2. Creating a Basic Operator

```python
from typing import ClassVar
from ember.core.registry.model.model_module import LMModule, LMModuleConfig
from ember.core.registry.specification import Specification

# Define the specification (input/output schema + prompt template)
class ClassifierSpecification(Specification):
    input_model = ClassifierInput
    structured_output = ClassifierOutput
    prompt_template = """Classify the following text into one of these categories: {categories}
    
Text: {text}

Respond with a JSON object with two keys:
- "category": The best matching category
- "confidence": A number between 0 and 1 indicating confidence
"""

# Create the operator
class TextClassifierOperator(Operator[ClassifierInput, ClassifierOutput]):
    # Class-level specification declaration
    specification: ClassVar[Specification] = ClassifierSpecification()
    
    # Class-level field declarations
    lm_module: LMModule
    
    def __init__(self, model_name: str = "openai:gpt-4o", temperature: float = 0.0):
        # Initialize fields
        self.lm_module = LMModule(LMModuleConfig(
            model_name=model_name,
            temperature=temperature,
            response_format={"type": "json_object"}
        ))
    
    def forward(self, *, inputs: ClassifierInput) -> ClassifierOutput:
        # Render prompt from input
        prompt = self.specification.render_prompt(inputs=inputs)
        
        # Call LLM
        response = self.lm_module(prompt)
        
        # Parse JSON response
        try:
            import json
            result = json.loads(response)
            # Return a proper instance of the output model
            return ClassifierOutput(
                category=result["category"],
                confidence=result["confidence"]
            )
        except Exception as e:
            # Error handling
            raise ValueError(f"Failed to parse LLM response: {e}")
```

## 3. Using Operators

```python
# Instantiate the operator
classifier = TextClassifierOperator(model_name="anthropic:claude-3-haiku")

# Use kwargs format for clean inputs
result = classifier(inputs={
    "text": "The sky is blue and the sun is shining brightly today.",
    "categories": ["weather", "politics", "technology", "sports"]
})

print(f"Category: {result.category}, Confidence: {result.confidence:.2f}")
```

## 4. Composition Patterns

### Sequential Composition

```python
from ember.core import non

# Create a pipeline of operators
pipeline = non.Sequential(operators=[
    preprocessor,
    classifier,
    postprocessor
])

# Execute the entire pipeline with a single call
final_result = pipeline(inputs={"text": "Your input text here"})
```

### Parallel Composition (via graphs)

```python
from ember.xcs.graph import XCSGraph
from ember.xcs.engine import execute_graph

# Create operators
sentiment = SentimentAnalysisOperator(model_name="openai:gpt-4o")
classifier = TextClassifierOperator(model_name="anthropic:claude-3-haiku")
summarizer = TextSummarizerOperator(model_name="openai:gpt-4o-mini")

# Build graph
graph = XCSGraph()
graph.add_node(operator=sentiment, node_id="sentiment")
graph.add_node(operator=classifier, node_id="classifier")
graph.add_node(operator=summarizer, node_id="summarizer")
graph.add_node(operator=aggregator, node_id="aggregator")

# Define data flows
graph.add_edge(from_id="sentiment", to_id="aggregator")
graph.add_edge(from_id="classifier", to_id="aggregator")
graph.add_edge(from_id="summarizer", to_id="aggregator")

# Execute with automatic parallelization
result = execute_graph(
    graph=graph,
    global_input={"text": "Your input text here"},
    max_workers=3
)
```

## 5. Built-in Operators

Ember provides pre-built operators for common tasks:

```python
from ember.core import non

# Create an ensemble of models
ensemble = non.UniformEnsemble(
    num_units=3,
    model_name="openai:gpt-4o", 
    temperature=0.7
)

# Create a judge to evaluate and combine responses
judge = non.JudgeSynthesis(model_name="anthropic:claude-3-sonnet")

# Execute them in a pipeline with kwargs format
ensemble_result = ensemble(inputs={"query": "What is the capital of France?"})
final_result = judge(inputs={
    "query": "What is the capital of France?", 
    "responses": ensemble_result["responses"]
})
```

## 6. Operator Best Practices

1. **Type Everything**: Define proper EmberModel classes for inputs and outputs
2. **Class-Level Fields**: Declare operator fields at the class level with type hints
3. **Specifications**: Always declare `specification: ClassVar[Specification]` for consistency with the core API; this indicates the specification belongs to the class, not the instance
4. **Use kwargs Format**: Use dict-style inputs with `inputs={"key": value}` for cleaner code
5. **Return Model Instances**: Return properly typed model instances from forward methods
6. **Error Handling**: Robustly handle parsing errors and LLM failures
7. **Pure Forward Methods**: Keep `forward()` methods deterministic
8. **Use Named Parameters**: Always use `inputs=` when calling operators
9. **Reuse Operators**: Compose small, specialized operators for complex tasks
10. **Flexible Initialization**: Allow customization via constructor parameters
11. **Stateless When Possible**: Prefer immutable state after initialization

## 7. Advanced: Custom Operator Transformations

```python
from typing import ClassVar, Type, TypeVar

from ember.xcs.tracer import jit
from ember.core.registry.specification import Specification
from ember.core.types.ember_model import EmberModel

I = TypeVar('I', bound=EmberModel)
O = TypeVar('O', bound=EmberModel)

# Create a decorator to retry operations on failure
def with_retry(max_attempts=3):
    def decorator(op_class: Type[Operator[I, O]]) -> Type[Operator[I, O]]:
        class RetryOperator(op_class):
            # Maintain the specification using ClassVar for consistency
            specification: ClassVar[Specification] = op_class.specification
            
            def forward(self, *, inputs: I) -> O:
                for attempt in range(max_attempts):
                    try:
                        return super().forward(inputs=inputs)
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            raise
                        print(f"Attempt {attempt+1} failed, retrying...")
        return RetryOperator
    return decorator

# Apply the decorator
@with_retry(max_attempts=3)
class ReliableClassifier(TextClassifierOperator):
    pass

# Create a JIT-compiled pipeline with the reliable classifier
@jit
class AnalysisPipeline(Operator[ClassifierInput, ClassifierOutput]):
    # Class-level specification declaration with ClassVar
    specification: ClassVar[Specification] = AnalysisSpecification()
    
    # Class-level field declarations
    classifier: ReliableClassifier
    
    def __init__(self):
        # Initialize fields
        self.classifier = ReliableClassifier()
    
    def forward(self, *, inputs: ClassifierInput) -> ClassifierOutput:
        result = self.classifier(inputs=inputs)
        return result

# Use the pipeline with kwargs format
result = AnalysisPipeline()(inputs={
    "text": "Solar panels convert sunlight into electricity.",
    "categories": ["technology", "environment", "politics", "entertainment"]
})
```

## Next Steps

Learn more about:
- [Prompt Specifications](specifications.md) - Type-safe prompt templating
- [Model Registry](model_registry.md) - Managing LLM configurations
- [NON Patterns](non.md) - Networks of Networks composition
- [XCS Graphs](../advanced/xcs_graphs.md) - Advanced parallel execution