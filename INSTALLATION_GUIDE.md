# Ember Installation Guide

This guide provides detailed instructions for installing Ember in different environments.

## System Requirements

- **Python**: 3.9 or newer (3.10, 3.11, and 3.12 supported)
- **Operating System**: macOS, Linux, or Windows

## Installation Methods

### Method 1: Basic Installation with uv (Recommended)

[uv](https://astral.sh/uv) is the recommended package manager for Ember. It is extremely fast (10-100x faster than pip) and simplifies Python environment management.

1. **Install uv** if you don't have it already:
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Or with pip if you prefer
   pip install uv
   
   # Verify the installation
   uv --version
   ```

2. **Install Ember from PyPI**:
   ```bash
   # Install Ember directly (creates a virtual environment automatically if needed)
   uv pip install ember-ai
   
   # Run examples without activating an environment
   ```

3. **Install from source**:
   ```bash
   # Clone the repository
   git clone https://github.com/pyember/ember.git
   cd pyember
   
   # Install in development mode (editable installation)
   uv pip install -e "."
   
   # Run examples directly without environment activation
   uv run python src/ember/examples/basic/minimal_example.py
   ```
   
   By default, this installs Ember with OpenAI, Anthropic, and Google/Deepmind provider support.

### Method 2: Development Installation with uv

If you want to develop or contribute to Ember:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pyember/ember.git
   cd ember
   ```

2. **Install with development dependencies**:
   ```bash
   # Install including development dependencies
   uv pip install -e ".[dev]"
   
   # Run commands directly with uv
   uv run pytest
   ```

3. **Running tools**:
   ```bash
   # Run linters, formatters, and other tools without installation
   uvx black src tests
   uvx mypy src
   uvx pytest
   ```

### Method 3: Traditional pip Installation (Alternative)

If you prefer using standard pip or don't want to install uv:

```bash
# Create a virtual environment (recommended)
python -m venv ember_env
source ember_env/bin/activate  # On Windows: ember_env\Scripts\activate

# Install Ember with pip
pip install ember-ai

# For development installation
pip install -e ".[dev]"
```

Note: This method is significantly slower for dependency resolution and doesn't provide the environment management benefits of uv.

## OS-Specific Installation Notes

### macOS

On macOS, you might encounter issues with the default Python installation:

```bash
# If you encounter Python-related errors:
# Install Python using Homebrew (recommended)
brew install python@3.11

# Use the Homebrew Python with uv
/opt/homebrew/bin/python3.11 -m pip install uv
/opt/homebrew/bin/python3.11 -m uv pip install -e "."
```

### Windows

On Windows, ensure you have the latest Python installed from python.org:

```powershell
# Add uv to your PATH if needed
$env:PATH += ";$env:USERPROFILE\.uv\bin"

# Install and run directly
uv pip install -e "."
uv run python src/ember/examples/basic/minimal_example.py
```

## Troubleshooting

### Python Version Issues

If you encounter Python version errors:

```bash
# Check your Python version
python --version

# Specify a Python version for uv
uv venv --python=3.11
source .venv/bin/activate

# Or run with a specific Python version
uv run --python=3.11 -- python script.py
```

### uv Installation Issues

If you have problems with uv:

```bash
# Ensure uv is in your PATH
which uv

# Update uv to the latest version
uv self update

# Reinstall uv if needed
pip install --upgrade uv
```

### Virtual Environment Issues

If you have problems with virtual environments:

```bash
# Create a fresh virtual environment
uv venv --force

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

See [ENVIRONMENT_MANAGEMENT.md](ENVIRONMENT_MANAGEMENT.md) for more details on managing environments.

### Dependency Conflicts

If you encounter dependency conflicts:

```bash
# Try reinstalling without using cache
uv pip install -e "." --no-cache

# Install with specific package versions if needed
uv pip install -e "." --no-deps
uv pip install "specific-package==version"
```

### Other Known Installation Issue Resolutions

When using conda with or without uv, you may encounter known pyarrow installation issues.
```
# Try installing pyarrow from conda-forge
conda install -c conda-forge pyarrow
```

## Testing Your Installation

After installation, verify everything is working:

```bash
# From the project root directory, using uv
uv run python src/ember/examples/basic/minimal_example.py

# Or if you're in an activated virtual environment
python src/ember/examples/basic/minimal_example.py
```

## Getting Help

If you encounter issues with installation:
- Check our [GitHub Issues](https://github.com/pyember/ember/issues)
- Review the [ENVIRONMENT_MANAGEMENT.md](ENVIRONMENT_MANAGEMENT.md) guide
- See the [TESTING_INSTALLATION.md](TESTING_INSTALLATION.md) for verification steps