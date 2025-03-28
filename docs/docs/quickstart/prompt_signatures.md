# Ember Prompt Specifications - Quickstart Guide

This guide introduces Ember's Prompt Specification system, which provides type-safe, composable prompt engineering with strong validation guarantees.

## 1. Introduction to Specifications

Specifications in Ember define the contract between inputs and outputs for an operator, including:
- Input schema (what data is required for the operation)
- Output schema (what structure the result will have)
- Prompt template (how to format inputs into a prompt)

They provide automatic validation, clear error messages, and consistent prompt formatting.

## 2. Creating a Basic Specification

```python
from ember.core.registry.specification import Specification
from pydantic import BaseModel, Field
from typing import List

# Define input schema
class QuestionAnsweringInput(BaseModel):
    question: str
    context: str = Field(description="Background information to answer the question")

# Define output schema
class QuestionAnsweringOutput(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Step-by-step reasoning process")

# Create the specification
class QASpecification(Specification):
    input_model = QuestionAnsweringInput
    structured_output = QuestionAnsweringOutput
    prompt_template = """Answer the following question based on the provided context.

Context: {context}

Question: {question}

Provide a clear, concise answer along with your confidence level and reasoning.
"""
```

## 3. Using Specifications with Operators

```python
from ember.core.registry.operator.base import Operator
from ember.core.registry.model.model_module import LMModule, LMModuleConfig

class QuestionAnsweringOperator(Operator[QuestionAnsweringInput, QuestionAnsweringOutput]):
    specification = QASpecification()
    
    def __init__(self, model_name: str = "openai:gpt-4o"):
        self.lm_module = LMModule(LMModuleConfig(
            model_name=model_name,
            response_format={"type": "json_object"}
        ))
    
    def forward(self, *, inputs: QuestionAnsweringInput) -> QuestionAnsweringOutput:
        # The specification automatically validates inputs
        prompt = self.specification.render_prompt(inputs=inputs)
        
        # Call LLM and parse response
        response_text = self.lm_module(prompt)
        import json
        response_data = json.loads(response_text)
        
        # The specification automatically validates output
        return self.specification.validate_output(output=response_data)
```

## 4. Specification Features

### Automatic Validation

```python
try:
    # This will fail validation (missing required field)
    invalid_input = {"question": "Who was Ada Lovelace?"}
    validated = specification.validate_inputs(inputs=invalid_input)
except Exception as e:
    print(f"Validation error: {e}")
    
# This will pass validation
valid_input = {
    "question": "Who was Ada Lovelace?",
    "context": "Ada Lovelace was an English mathematician and writer..."
}
validated = specification.validate_inputs(inputs=valid_input)
```

### Flexible Prompt Rendering

```python
# Render from dictionary
prompt1 = specification.render_prompt(inputs={
    "question": "What is quantum computing?",
    "context": "Quantum computing uses quantum bits or qubits..."
})

# Render from validated model
input_model = QuestionAnsweringInput(
    question="What is quantum computing?",
    context="Quantum computing uses quantum bits or qubits..."
)
prompt2 = specification.render_prompt(inputs=input_model)
```

### JSON Schema Generation

```python
# Generate JSON schema for documentation or client SDKs
input_schema = specification.model_json_schema(by_alias=True)
print(input_schema)
```

## 5. Advanced Specification Patterns

### Inheritance and Composition

```python
# Create a base specification
class BaseQASpecification(Specification):
    input_model = QuestionAnsweringInput
    structured_output = QuestionAnsweringOutput

# Extended version with different prompt
class DetailedQASpecification(BaseQASpecification):
    prompt_template = """Analyze the following question in detail, using the context.

Context:
{context}

Question:
{question}

Provide a detailed answer with reasoning and your confidence level.
"""
```

### Dynamic Templates

```python
class DynamicSpecification(Specification):
    input_model = QuestionAnsweringInput
    structured_output = QuestionAnsweringOutput
    
    def __init__(self, style: str = "concise"):
        super().__init__()
        if style == "concise":
            self.prompt_template = "Answer briefly: {question}\nContext: {context}"
        elif style == "detailed":
            self.prompt_template = """Provide a detailed analysis.
            
Question: {question}

Full context:
{context}

Include reasoning and confidence level.
"""
```

### Custom Validation Logic

```python
from pydantic import model_validator

class ValidatedInput(BaseModel):
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=100)
    
    @model_validator(mode="after")
    def validate_temperature_range(self):
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        return self

class CustomValidationSpecification(Specification):
    input_model = ValidatedInput
    
    def validate_inputs(self, *, inputs):
        validated = super().validate_inputs(inputs=inputs)
        # Add custom validation logic
        if validated.temperature > 0.9 and validated.max_tokens < 50:
            raise ValueError("High temperature requires higher max_tokens")
        return validated
```

## 6. Best Practices

1. **Type Everything**: Define clear input and output models
2. **Explicit Placeholders**: Make all placeholders explicit in your template
3. **Nested Models**: Use nested Pydantic models for complex data structures
4. **Clear Errors**: Provide descriptive error messages in custom validators
5. **Reuse Specifications**: Create base specifications and extend them for specific use cases
6. **Documentation**: Use Field descriptions to document your schemas

## 7. Using with Different LLM Providers

```python
from ember.core.registry.model.model_module import LMModuleConfig
from ember.core.registry.model.base.services import ModelService
from ember import initialize_ember

# Initialize model registry
service = initialize_ember(auto_register=True)

# Create operator with specification
class FlexibleQAOperator(Operator[QuestionAnsweringInput, QuestionAnsweringOutput]):
    specification = QASpecification()
    
    def __init__(self, model_id: str):
        self.model = service.get_model(model_id)
    
    def forward(self, *, inputs: QuestionAnsweringInput) -> QuestionAnsweringOutput:
        prompt = self.specification.render_prompt(inputs=inputs)
        response_text = self.model(prompt)
        # Process response and validate output
        # ...

# Create instances for different providers
openai_qa = FlexibleQAOperator("openai:gpt-4o")
anthropic_qa = FlexibleQAOperator("anthropic:claude-3-sonnet")
```

## Next Steps

Learn more about:
- [Operators](operators.md) - Building computational units in Ember
- [Model Registry](model_registry.md) - Managing LLM configurations
- [Non Patterns](non.md) - Networks of Networks composition
- [Structured Output](../advanced/structured_output.md) - Advanced output parsing techniques