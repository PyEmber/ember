# Ember Exception Architecture

## Overview

Ember uses a hierarchical exception system to provide precise error information. Each exception includes context data, error codes, and recovery hints to simplify debugging. All exceptions inherit from a base `EmberError` class and follow domain boundaries that match the framework's structure.

## Core Features

### 1. Contextual Information

Exceptions capture detailed context data:

```python
try:
    # Operation that fails
except EmberError as e:
    print(f"Error context: {e.get_context()}")
```

### 2. Error Codes

Exceptions use numeric codes organized by domain for consistent error identification:

```
[Error 3001] Provider API error for 'openai': Rate limit exceeded 
```

### 3. Recovery Hints

Exceptions include actionable recovery suggestions:

```
[Error 6004] Missing configuration key 'model.api_key' 
[Recovery: Add the missing configuration key to your config file]
```

### 4. Logging Integration

Exceptions integrate with the logging system:

```python
try:
    # Operation that fails
except EmberError as e:
    e.log_with_context(logger)
```

## Exception Hierarchy

The exception hierarchy is organized by domain:

```
EmberError
├── ErrorGroup - Aggregates multiple related errors
├── Core Framework (1000-1999)
│   ├── InvalidArgumentError
│   ├── ValidationError
│   ├── NotImplementedFeatureError
│   ├── DeprecationError
│   ├── IncompatibleTypeError
│   └── InitializationError
├── Registry Exceptions (1100-1199)
│   ├── RegistryError
│   ├── ItemNotFoundError
│   ├── DuplicateItemError
│   └── RegistrationError
├── Operator Framework (2000-2999)
│   ├── OperatorError
│   ├── OperatorSpecificationError
│   ├── OperatorExecutionError
│   ├── SpecificationValidationError
│   ├── TreeTransformationError
│   └── FlattenError
├── Model Framework (3000-3999)
│   ├── ModelError
│   ├── ModelProviderError
│   ├── ModelNotFoundError
│   ├── ProviderAPIError
│   ├── ProviderConfigError
│   ├── ModelDiscoveryError
│   ├── ModelRegistrationError
│   ├── MissingLMModuleError
│   └── InvalidPromptError
├── Data Framework (4000-4999)
│   ├── DataError
│   ├── DataValidationError
│   ├── DataTransformationError
│   ├── DataLoadError
│   └── DatasetNotFoundError
├── XCS Framework (5000-5999)
│   ├── XCSError
│   ├── TraceError
│   ├── CompilationError
│   ├── ExecutionError
│   ├── TransformError
│   ├── ParallelExecutionError
│   ├── DataFlowError
│   └── SchedulerError
├── Configuration (6000-6999)
│   ├── ConfigError
│   ├── ConfigValidationError
│   ├── ConfigFileError
│   ├── ConfigValueError
│   └── MissingConfigError
├── API (7000-7999)
│   └── APIError
├── CLI (8000-8999)
│   └── CLIError
└── Plugin System (9000-9999)
    ├── PluginError
    ├── PluginNotFoundError
    ├── PluginConfigError
    └── PluginLoadError
```

## Usage Patterns

### Type-Specific Exceptions

Use the most specific exception type for the error condition:

```python
if model_name not in available_models:
    raise ModelNotFoundError.for_model(
        model_name=model_name, 
        provider_name=provider
    )
```

### Context Enrichment

Include relevant context data:

```python
try:
    # Operation
except Exception as e:
    raise DataLoadError(
        message="Failed to load dataset",
        context={"dataset_name": name, "format": format},
        cause=e
    )
```

### Factory Methods

Use factory methods to create exceptions with consistent context:

```python
# Instead of:
raise ItemNotFoundError(f"Item '{item_name}' not found in {registry_name}")

# Use:
raise ItemNotFoundError.for_item(item_name, registry_name)
```

### Error Aggregation

Group related errors when appropriate:

```python
validation_errors = []
# Collect multiple validation errors
if validation_errors:
    raise ErrorGroup(
        message="Multiple validation errors occurred",
        errors=validation_errors
    )
```

## Standard Python Exception Conversion

Replace standard Python exceptions with Ember equivalents:

```python
# Instead of:
if not isinstance(inputs, dict):
    raise ValueError("Inputs must be a dictionary")

# Use:
if not isinstance(inputs, dict):
    raise ValidationError(
        "Inputs must be a dictionary",
        context={"input_type": type(inputs).__name__}
    )
```

## Creating Custom Exceptions

Define domain-specific exceptions by extending the base classes:

```python
from ember.core.exceptions import EmberError

class MyDomainError(EmberError):
    """Base exception for my domain."""
    DEFAULT_ERROR_CODE = 9500  # Use a unique range
    DEFAULT_RECOVERY_HINT = "Check domain configuration"
```