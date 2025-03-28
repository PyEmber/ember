# Ember CLI Shell Completion

Ember CLI provides powerful shell completion for Bash, Zsh, Fish, and PowerShell. This feature gives you tab completion for commands, options, and arguments, making the CLI much more user-friendly and efficient.

## Completion Features

The shell completion provides context-aware suggestions for:

- Commands and subcommands
- Options and flags
- Provider names
- Model IDs
- Project templates
- File and directory paths
- Configuration keys

## Installation

### Quick Installation

The easiest way to install completion is to use the built-in installation command:

```bash
# Install for the current shell
ember completion install

# Force overwrite of existing completion
ember completion install --force
```

### Manual Installation

You can also generate the completion script and install it manually:

#### Bash

```bash
# Generate completion script
ember completion bash > ~/.bash_completion.d/ember

# Add to your ~/.bashrc
echo 'source ~/.bash_completion.d/ember' >> ~/.bashrc
```

#### Zsh

```bash
# Create completions directory if it doesn't exist
mkdir -p ~/.zsh/completion

# Generate completion script
ember completion zsh > ~/.zsh/completion/_ember

# Add to your ~/.zshrc
echo 'fpath=(~/.zsh/completion $fpath)' >> ~/.zshrc
echo 'autoload -U compinit' >> ~/.zshrc
echo 'compinit' >> ~/.zshrc
```

#### Fish

```bash
# Generate completion script
ember completion fish > ~/.config/fish/completions/ember.fish
```

#### PowerShell

```powershell
# Generate completion script
ember completion powershell > $PROFILE.CurrentUserAllHosts
```

## Usage Examples

Once installed, you can use tab completion throughout the Ember CLI:

### Command Completion

```bash
# Press TAB to see available commands
ember [TAB]
# Output: version  provider  model  invoke  project  config  completion

# Press TAB to see subcommands
ember provider [TAB]
# Output: list  configure  info  use
```

### Option Completion

```bash
# Press TAB to see available options
ember invoke --[TAB]
# Output: --model  --prompt  --file  --system  --temperature  --show-usage  --output  --raw  --stream

# Complete option values
ember model list --provider [TAB]
# Output: openai  anthropic  google  ibm
```

### Dynamic Completion

The completion system can fetch data from your Ember installation to provide dynamic suggestions:

```bash
# Complete provider names
ember provider configure [TAB]
# Output will list available providers

# Complete model IDs
ember invoke --model [TAB]
# Output will list available models
```

## Troubleshooting

If you encounter issues with completion:

1. Make sure your shell startup files are properly configured
2. Try reinstalling completion with `ember completion install --force`
3. Check that the completion script is being sourced properly
4. Restart your shell or terminal

### Common Issues

- **No completion in Bash**: Check that your `~/.bash_completion.d/ember` file exists and is sourced in your `~/.bashrc`
- **No completion in Zsh**: Check that `fpath` is properly configured in your `~/.zshrc`
- **Outdated completions**: Regenerate completions after updating Ember CLI

## How It Works

The shell completion is implemented using:

- Command metadata declared in TypeScript interfaces
- Shell-specific generators that create native completion scripts
- Dynamic data sourcing from the Ember backend
- Context-aware argument parsing

The system follows a declarative approach, where commands and their arguments are defined once and specialized generators produce shell-specific implementations.

## For Developers

If you're extending the CLI with new commands, you should update the command specifications in `src/cli/utils/completion.ts` to ensure proper completion support.

The completion system follows SOLID principles, particularly:

- **Single Responsibility**: Each generator is responsible for one shell
- **Open/Closed**: You can add new shell support without modifying existing code
- **Liskov Substitution**: All generators implement the same interface
- **Interface Segregation**: Clean separation of concerns
- **Dependency Inversion**: High-level completion logic does not depend on shell-specific details