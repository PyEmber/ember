# Ember CLI Documentation

The Ember CLI is a powerful command-line interface for interacting with the Ember AI framework. It provides a beautiful, intuitive interface for managing models, providers, configurations, and projects.

## Installation

### Prerequisites

- Node.js 16.0 or higher
- Python 3.11 or higher
- Ember AI package installed (`uv pip install ember-ai`)

### Installing the CLI

```bash
# Global installation
npm install -g ember-cli

# Local installation
npm install ember-cli
```

## Getting Started

### Quick Start

```bash
# Display the version
ember version

# List available providers
ember provider list

# Configure a provider with your API key
ember provider configure openai

# List available models
ember model list

# Invoke a model
ember invoke --model openai:gpt-4o-mini --prompt "Hello, world!"
```

### Basic Concepts

Ember CLI organizes functionality around these core concepts:

1. **Providers**: AI service providers like OpenAI, Anthropic, etc.
2. **Models**: Specific LLM models offered by providers, referenced as `provider:model`
3. **Projects**: Ember applications that utilize the framework
4. **Configuration**: Settings for CLI, providers, and models

## Command Reference

### Global Options

These options work with all commands:

| Option | Description |
|--------|-------------|
| `--debug` | Enable debug mode with detailed logs |
| `--json` | Output results as JSON |
| `--quiet` | Suppress non-essential output |
| `--no-color` | Disable colored output |

### Shell Completion

Ember CLI supports shell completion for Bash, Zsh, Fish, and PowerShell. This provides tab completion for commands, options, and even dynamic values like model IDs and provider names.

```bash
# To install shell completion for your current shell
ember completion install

# Generate completion script for a specific shell
ember completion bash > ~/.bash_completion.d/ember
ember completion zsh > ~/.zsh/completion/_ember
ember completion fish > ~/.config/fish/completions/ember.fish
```

For detailed instructions, see [Shell Completion Documentation](SHELL_COMPLETION.md).

### Core Commands

#### Version

Display version information about the CLI and backend.

```bash
ember version [options]

Options:
  --check    Check for updates
```

#### Providers

Manage LLM providers in Ember.

```bash
# List available providers
ember provider list

# Configure a provider with API key
ember provider configure <provider> [options]
Options:
  -k, --key <key>    API key (omit to be prompted securely)
  -f, --force        Overwrite existing configuration

# Display provider information
ember provider info <provider>

# Set a provider as default
ember provider use <provider>
```

#### Models

Manage LLM models in Ember.

```bash
# List available models
ember model list [options]
Options:
  -p, --provider <provider>    Filter models by provider

# Display model information
ember model info <model>

# Set a model as default
ember model use <model>

# Benchmark a model's performance
ember model benchmark <model> [options]
Options:
  -t, --tests <tests>              Number of tests to run (default: 5)
  -c, --concurrency <concurrency>  Concurrency level (default: 1)
```

#### Invoke

Invoke a model with a prompt directly from the CLI.

```bash
ember invoke [options]

Options:
  -m, --model <model>          Model ID to use
  -p, --prompt <prompt>        Prompt text to send to the model
  -f, --file <file>            Read prompt from file
  -s, --system <system>        System prompt (for chat models)
  -t, --temperature <temp>     Temperature setting (0.0-2.0) (default: 1.0)
  -u, --show-usage             Show token usage statistics
  -o, --output <file>          Save output to file
  -r, --raw                    Display raw output without formatting
  --stream                     Stream the response token by token
```

#### Projects

Create and manage Ember projects.

```bash
# Create a new project
ember project new <name> [options]
Options:
  -t, --template <template>     Project template (default: basic)
  -d, --directory <directory>   Project directory (defaults to name)
  -p, --provider <provider>     Default provider to use
  -m, --model <model>           Default model to use

# List available project templates
ember project templates

# Analyze an existing project
ember project analyze [directory]
```

#### Configuration

Manage Ember CLI configuration.

```bash
# List configuration settings
ember config list [options]
Options:
  --show-keys    Show API keys (not recommended)

# Set a configuration value
ember config set <key> <value>

# Get a configuration value
ember config get <key>

# Reset configuration to defaults
ember config reset [options]
Options:
  -f, --force    Skip confirmation

# Export configuration to a file
ember config export <file>

# Import configuration from a file
ember config import <file> [options]
Options:
  -f, --force    Overwrite existing configuration

# Configure usage tracking
ember config usage-tracking <enabled>
```

## Usage Examples

### Configuring Providers

```bash
# List available providers
$ ember provider list
▸ Available Providers

Provider    Status           Default
──────────  ───────────────  ───────
openai      Not Configured    
anthropic   Not Configured    
google      Not Configured    

💡 Tip: Configure a provider with ember provider configure <provider>

# Configure OpenAI
$ ember provider configure openai
Enter API key for openai: [input is hidden]
✅ Provider openai configured successfully.
✅ Provider openai set as default.

# Check provider information
$ ember provider info openai
▸ Provider: openai

Name: OpenAI
Description: Provider for OpenAI models including GPT-4 and GPT-3.5
Website: https://openai.com

Available Models:
  • gpt-4o
  • gpt-4o-mini
  • gpt-4-turbo
  • gpt-3.5-turbo

Authentication:
  Environment Variable: OPENAI_API_KEY
  Status: Configured
```

### Working with Models

```bash
# List all models
$ ember model list

▸ Available Models

openai
  ✓ gpt-4o
   gpt-4o-mini
   gpt-4-turbo
   gpt-3.5-turbo

anthropic
   claude-3-opus
   claude-3-sonnet

💡 Tip: Get model details with ember model info <model>

# Get model information
$ ember model info openai:gpt-4o
▸ Model: openai:gpt-4o

Name: GPT-4o
Description: OpenAI's most advanced model for vision, language, audio and video.
Provider: openai

Capabilities:
  ✓ text
  ✓ images
  ✓ audio
  ✓ function_calling

Context Size: 128,000 tokens

Cost:
  Input: $0.005 per 1K tokens
  Output: $0.015 per 1K tokens

💡 Tip: Invoke this model with ember invoke --model openai:gpt-4o --prompt "Your prompt"

# Set default model
$ ember model use openai:gpt-4o
✅ Model openai:gpt-4o set as default.
```

### Invoking Models

```bash
# Invoke a model with a prompt
$ ember invoke --model openai:gpt-4o --prompt "Explain quantum computing in simple terms"
Invoking openai:gpt-4o...
────────────────────────────────────────────────────────────────
Quantum computing is like having a super-powerful calculator that works in a completely different way than regular computers.

Regular computers use bits (0s and 1s) to process information one step at a time. Quantum computers use quantum bits or "qubits" that can be 0, 1, or both at the same time (called superposition).

This means quantum computers can look at many possible solutions simultaneously instead of checking them one by one. It's like being able to take all possible paths through a maze at once, rather than trying each path individually.

This makes quantum computers potentially much faster at solving certain types of problems, like breaking encryption codes, simulating molecules for drug discovery, or optimizing complex systems like traffic flows in a city.

However, quantum computers are still very new, expensive, and difficult to build and maintain. They need to be kept extremely cold, and they're prone to errors. So while they show promise for specific applications, they won't replace your laptop anytime soon!

# Invoke with system prompt and save output
$ ember invoke -m openai:gpt-4o -p "List 5 renewable energy sources" -s "You are an environmental scientist" -o energy.txt
Invoking openai:gpt-4o...
────────────────────────────────────────────────────────────────
As an environmental scientist, I can identify these 5 key renewable energy sources:

1. Solar Energy: Harnessed through photovoltaic panels or concentrated solar power systems that convert sunlight directly into electricity or heat.

2. Wind Energy: Generated by turbines that convert kinetic energy from wind into mechanical power then electricity, with offshore wind farms showing particularly high efficiency.

3. Hydroelectric Power: Utilizes the gravitational force of flowing or falling water, typically through dams or run-of-river systems, representing one of the most established renewable sources.

4. Geothermal Energy: Taps into Earth's internal thermal energy through wells that access heated water and steam reservoirs, providing consistent baseload power with minimal land footprint.

5. Biomass Energy: Derived from organic materials like wood, agricultural residues, or dedicated energy crops that store solar energy through photosynthesis, which can be converted to heat, electricity, or liquid biofuels.

Each of these sources represents a sustainable alternative to fossil fuels with significantly lower lifecycle carbon emissions, though each has specific geographic, environmental, and economic considerations for optimal implementation.

✅ Output saved to /Users/username/energy.txt
```

### Creating Projects

```bash
# List available templates
$ ember project templates

▸ Available Project Templates

Basic Project (basic)
Simple starter project with minimal dependencies

Features:
  • Direct model invocation
  • Basic usage examples
──────────────────────────────────────────────────

Complete Project (complete)
Full-featured project with all operators and utilities

Features:
  • Operators
  • Evaluation tools
  • Configuration examples
  • Advanced usage patterns
──────────────────────────────────────────────────

API Project (api)
Project template for building APIs with Ember

Features:
  • FastAPI integration
  • API endpoint examples
  • Authentication boilerplate
──────────────────────────────────────────────────

Notebook Project (notebook)
Jupyter notebook-based project for experimentation

Features:
  • Jupyter notebooks
  • Example experiments
  • Visualization tools
──────────────────────────────────────────────────

💡 Tip: Create a project with ember project new <name> --template <template>

# Create a new project
$ ember project new my-ember-app --template complete
✅ Initializing project
✅ Installing dependencies
✅ Setting up configuration
✅ Finalizing project
✅ Project my-ember-app created successfully in my-ember-app

To get started, run:
  cd my-ember-app
  uv pip install -e "."

💡 Tip: Set up your API keys as environment variables or configure them with ember provider configure <provider>

# Analyze a project
$ ember project analyze my-ember-app

▸ Project Analysis: my-ember-app

Project Type: Complete Project
Directory: /Users/username/my-ember-app

Project Structure:
  Operators: 2
  Models: 1
  Data Sources: 1
  Test Files: 3

Dependencies:
  Required: ember-ai, openai, pandas
  All dependencies satisfied

Configuration:
  Providers: openai
  Status: Configured

Suggestions:
  • Add automated tests for your operators
  • Consider implementing usage tracking
  • Update to the latest Ember version
```

### Managing Configuration

```bash
# List configuration
$ ember config list

▸ Ember CLI Configuration

Default Provider: openai
Default Model: openai:gpt-4o
Usage Tracking: Enabled

Configured Providers:
  ✓ openai
   anthropic

💡 Tip: Set a configuration value with ember config set <key> <value>

# Set configuration
$ ember config set defaultModel openai:gpt-4o-mini
✅ Configuration value for defaultModel set to "openai:gpt-4o-mini"

# Export configuration
$ ember config export ember-config.json
✅ Configuration exported to /Users/username/ember-config.json
⚠️ This file contains sensitive information such as API keys.

# Reset configuration
$ ember config reset
This will reset all configuration to defaults. Continue? (y/N) y
✅ Configuration reset to defaults
```

## Environment Variables

The CLI respects the following environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google API key |
| `EMBER_DEFAULT_PROVIDER` | Default provider |
| `EMBER_DEFAULT_MODEL` | Default model |
| `EMBER_NO_COLOR` | Disable colors when set to any value |
| `EMBER_DEBUG` | Enable debug mode when set to any value |

## Extending the CLI

The Ember CLI is designed to be extensible. You can create plugins and extensions by following our plugin development guidelines.

## Troubleshooting

### Common Issues

1. **Python Bridge Errors**
   - Ensure Python 3.11+ is installed and in your PATH
   - Verify Ember AI is installed (`uv pip list | grep ember-ai`)

2. **API Key Issues**
   - Check provider configuration (`ember config list`)
   - Verify environment variables are set correctly

3. **Performance Problems**
   - For large responses, use the `--output` option to save to file
   - Consider using streaming with `--stream` option

### Getting Help

```bash
# Get general help
ember help

# Get help for a specific command
ember provider --help
```

## Advanced Topics

### Scripting with the CLI

The `--json` flag makes the CLI suitable for scripting:

```bash
# Get models as JSON and filter with jq
models=$(ember model list --json | jq '.models[]' | grep 'gpt-4')

# Use in shell scripts
for model in $models; do
  echo "Testing $model"
  ember invoke --model $model --prompt "Hello" --json > results_$model.json
done
```

### Using Configuration Files

You can use configuration files to manage settings across environments:

```bash
# Export production configuration
ember config export prod-config.json

# Import configuration on another machine
ember config import prod-config.json

# Use different configurations for different projects
cd project1
ember config import project1-config.json
```

## Best Practices

1. **Security**
   - Never commit configuration files with API keys
   - Use environment variables when possible
   - Rotate API keys regularly

2. **Usage**
   - Set default provider and model to reduce typing
   - Use templates for consistent project structure
   - Leverage the `invoke` command for quick experiments

3. **Performance**
   - Use streaming for large responses
   - Consider model benchmark results for latency-sensitive applications
   - Start with smaller, more efficient models when appropriate