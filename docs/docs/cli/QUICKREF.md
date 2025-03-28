# Ember CLI Quick Reference

This quick reference guide covers the most common Ember CLI commands and usage patterns.

## Getting Started

```bash
# Install the CLI
npm install -g ember-cli

# Check version
ember version

# Get help
ember help
```

## Provider Management

```bash
# List providers
ember provider list

# Configure a provider
ember provider configure openai

# Set default provider
ember provider use openai
```

## Model Management

```bash
# List models
ember model list

# List models for a specific provider
ember model list --provider openai

# Get model information
ember model info openai:gpt-4o

# Set default model
ember model use openai:gpt-4o
```

## Invoking Models

```bash
# Basic invocation (uses default model if set)
ember invoke --prompt "Hello, world!"

# Invoke specific model
ember invoke --model openai:gpt-4o --prompt "Hello, world!"

# Read prompt from file
ember invoke --model openai:gpt-4o --file myprompt.txt

# Add system prompt
ember invoke --model openai:gpt-4o --prompt "List 5 capitals" --system "You are a geography expert"

# Save output to file
ember invoke --model openai:gpt-4o --prompt "Write a poem" --output poem.txt

# Show token usage
ember invoke --model openai:gpt-4o --prompt "Explain quantum computing" --show-usage

# Stream response
ember invoke --model openai:gpt-4o --prompt "Write a story" --stream
```

## Project Management

```bash
# Create new project
ember project new myproject

# Create project with specific template
ember project new myproject --template api

# List available templates
ember project templates

# Analyze project
ember project analyze ./myproject
```

## Configuration Management

```bash
# List configuration
ember config list

# Set configuration
ember config set defaultModel openai:gpt-4o

# Get specific configuration
ember config get defaultProvider

# Export/import configuration
ember config export config.json
ember config import config.json

# Reset configuration
ember config reset
```

## JSON Output (for scripting)

```bash
# Get providers as JSON
ember provider list --json

# Get models as JSON
ember model list --json

# Run model and get JSON result
ember invoke --model openai:gpt-4o --prompt "Hello" --json > result.json
```

## Environment Variables

```bash
# Set API keys
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"

# Set defaults
export EMBER_DEFAULT_PROVIDER="openai"
export EMBER_DEFAULT_MODEL="openai:gpt-4o"

# Other options
export EMBER_DEBUG=1
export EMBER_NO_COLOR=1
```

## Common Options

```bash
# Debug mode
ember <command> --debug

# Disable colors
ember <command> --no-color

# Quiet mode (minimal output)
ember <command> --quiet

# JSON output
ember <command> --json
```

## Shell Completion

```bash
# Install shell completion
ember completion install

# Generate completion for a specific shell
ember completion bash
ember completion zsh
ember completion fish
ember completion powershell

# Output to a file
ember completion bash ~/.bash_completion.d/ember
```