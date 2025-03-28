# Ember Model Module - Quickstart Guide

This quickstart guide will help you integrate LLM models into your project using Ember's model module. The guide follows SOLID principles and best practices for modularity and maintainability.

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/pyember/ember.git
cd ember

# Install using uv
uv pip install -e "."
```

## 2. API Key Setup

Set your API keys as environment variables:

```bash
# For bash/zsh
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"

# For Windows PowerShell
$env:OPENAI_API_KEY="your-openai-key"
$env:ANTHROPIC_API_KEY="your-anthropic-key"
$env:GOOGLE_API_KEY="your-google-key"
```

Alternatively, create a `.env` file in your project:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
```

## 3. Basic Usage (One Line)

```python
import ember

# Initialize and use in one line (automatically uses configuration system)
service = ember.init()
response = service("openai:gpt-4o", "Hello world!")
print(response.data)
```

## 4. Standard Usage Pattern

```python
from ember.core.registry.model.initialization import initialize_registry
from ember.core.config.manager import create_config_manager
from ember.core.registry.model.base.services.model_service import ModelService

# Initialize the configuration and registry
config_manager = create_config_manager()
registry = initialize_registry(config_manager=config_manager)

# Create a service instance
service = ModelService(registry=registry)

# Use the service to invoke models
response = service.invoke_model(
    model_id="anthropic:claude-3-5-sonnet", 
    prompt="Explain quantum computing in 50 words"
)
print(response.data)
```

## 5. Direct Model Access (Pytorch-like)

```python
# Get model directly for more control
model = service.get_model("openai:gpt-4o")

# Use the model directly
response = model("What's the capital of France?")
print(response.data)
```

## 6. Usage Tracking

```python
from ember.core.registry.model.initialization import initialize_registry
from ember.core.config.manager import create_config_manager
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.services.usage_service import UsageService

# Initialize configuration and registry with usage tracking
config_manager = create_config_manager()
registry = initialize_registry(config_manager=config_manager)
usage_service = UsageService()
service = ModelService(registry=registry, usage_service=usage_service)

# Make model requests
response = service("openai:gpt-4o", "Hello world!")

# Get usage statistics
total_usage = usage_service.get_total_usage()
print(f"Total tokens: {total_usage.tokens}")
print(f"Total cost: ${total_usage.cost}")
```

## 7. Available Models

You can use any of these models by their ID or corresponding ModelEnum:

### OpenAI Models
- `openai:gpt-4o` or `ModelEnum.gpt_4o`
- `openai:gpt-4o-mini` or `ModelEnum.gpt_4o_mini`
- `openai:gpt-4` or `ModelEnum.gpt_4`
- `openai:gpt-4-turbo` or `ModelEnum.gpt_4_turbo`
- `openai:gpt-3.5-turbo` or `ModelEnum.gpt_3_5_turbo`
- `openai:o1-2024-12-17` or `ModelEnum.o1`

### Anthropic Models
- `anthropic:claude-3.7-sonnet` or `ModelEnum.claude_3_7_sonnet`
- `anthropic:claude-3-5-sonnet` or `ModelEnum.claude_3_5_sonnet`
- `anthropic:claude-3-5-haiku` or `ModelEnum.claude_3_5_haiku`
- `anthropic:claude-3-opus` or `ModelEnum.claude_3_opus`
- `anthropic:claude-3-haiku` or `ModelEnum.claude_3_haiku`

### Deepmind Models
- `deepmind:gemini-1.5-pro` or `ModelEnum.gemini_1_5_pro`
- `deepmind:gemini-1.5-flash` or `ModelEnum.gemini_1_5_flash`
- `deepmind:gemini-1.5-flash-8b` or `ModelEnum.gemini_1_5_flash_8b`
- `deepmind:gemini-2.0-flash` or `ModelEnum.gemini_2_0_flash`
- `deepmind:gemini-2.0-flash-lite` or `ModelEnum.gemini_2_0_flash_lite`
- `deepmind:gemini-2.0-pro` or `ModelEnum.gemini_2_0_pro`

## 8. Error Handling

```python
try:
    response = service("openai:gpt-4o", "Hello world!")
    print(response.data)
except Exception as e:
    print(f"Error: {str(e)}")
```

## 9. Advanced: Adding Custom Models

```python
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.model_info import ModelInfo

# Create model info
custom_model = ModelInfo(
    id="custom:my-model",
    name="my-custom-model",
    provider={
        "name": "CustomProvider",
        "default_api_key": "${CUSTOM_API_KEY}"
    }
)

# Register custom model
registry = ModelRegistry()
registry.register_model(custom_model)
```

## 10. Type-safe Model Invocation with Enums

```python
from ember import initialize_ember
from ember.api.models import ModelEnum

# Initialize Ember with the simplified API
service = initialize_ember(usage_tracking=True)

# Use enum for type-safety
response = service(ModelEnum.gpt_4o, "Hello world!")
print(response.data)
```

## Next Steps

For more advanced usage, check out:
- Custom provider integration
- Multi-model ensembling
- Prompt templating
- Streaming responses

## Provider Discovery System

The model registry integrates a robust provider discovery mechanism with three key design principles:

### 1. Resilience

The discovery system ensures application stability through:
- **Graceful degradation** - Fallback models when APIs are unreachable
- **Timeout protection** - Automatic 30s timeout with ThreadPoolExecutor to prevent indefinite blocking
- **Error isolation** - API failures in one provider don't affect others

### 2. Testing Strategy

Testing model discovery requires both:
- **Unit tests** - Fast, deterministic verification with mocks
- **Integration tests** - Selective API verification with real credentials

Enable integration tests with environment flags:
```bash
RUN_PROVIDER_INTEGRATION_TESTS=1 pytest tests/integration/core/registry/test_provider_discovery.py -v
```

### 3. Design Patterns

The discovery system employs:
- **Adapter Pattern** - Unified interface across varying provider APIs
- **Factory Pattern** - Dynamic provider instantiation based on available credentials
- **Dependency Injection** - Explicit API key configuration to avoid global state
- **Template Method** - Common discovery workflow with provider-specific implementations