# Ember CLI Developer Guide

This guide is intended for developers who want to contribute to the Ember CLI or extend it with plugins and custom functionality.

## Architecture Overview

The Ember CLI uses a hybrid architecture:

1. **Frontend**: Node.js/TypeScript CLI interface built with Commander.js
2. **Backend**: Python Ember core framework accessed via python-bridge
3. **Configuration**: Local storage using the conf package with encryption

```
┌─────────────────────────────────────────┐
│             Node.js Frontend            │
├─────────────┬───────────────┬───────────┤
│ UI/UX       │ Command       │ Config    │
│ Components  │ Handlers      │ Manager   │
├─────────────┴───────┬───────┴───────────┤
│        Python Bridge                    │
├─────────────────────┼───────────────────┤
│        Python Backend (Ember Core)      │
└─────────────────────────────────────────┘
```

### Key Components

- **CLI Entry Point** (`src/cli/index.ts`): Main entry point that sets up commands
- **Command Modules** (`src/cli/commands/`): Individual command implementations
- **Python Bridge** (`src/cli/bridge/`): Interface between TypeScript and Python
- **UI Components** (`src/cli/ui/`): User interface elements
- **Services** (`src/cli/services/`): Shared functionality
- **Utilities** (`src/cli/utils/`): Helper functions

## Development Setup

### Prerequisites

- Node.js 16+ 
- Python 3.11+
- TypeScript 4.9+
- Ember AI package

### Installation

```bash
# Clone the repository
git clone https://github.com/pyember/ember.git
cd ember

# Install dependencies
npm install

# Build the CLI
npm run build

# Create a link for local development
npm link
```

### Development Workflow

```bash
# Run in development mode with watch
npm run watch

# Test your changes
ember --debug <command>

# Lint code
npm run lint

# Run tests
npm test
```

## Adding a New Command

To add a new command to the CLI, follow these steps:

1. Create a new file in `src/cli/commands/` for your command
2. Implement the command with the Commander.js pattern
3. Register your command in `src/cli/index.ts`

Here's an example of a new command:

```typescript
// src/cli/commands/example.ts
import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';

import { getPythonBridge } from '../bridge/python-bridge';
import { isJsonOutput } from '../utils/options';
import { displaySection, displaySuccess } from '../ui/intro';

/**
 * Register example command with the CLI program
 * 
 * @param program The commander program instance
 */
export function registerExampleCommand(program: Command): void {
  program
    .command('example')
    .description('Example command to demonstrate development')
    .option('-n, --name <name>', 'Name parameter')
    .action(async (options) => {
      await runExample(options);
    });
}

/**
 * Run the example command
 * 
 * @param options Command options
 */
async function runExample(options: any): Promise<void> {
  const spinner = ora('Running example...').start();
  
  try {
    // Get name from options or use default
    const name = options.name || 'world';
    
    // Get Python bridge
    const bridge = getPythonBridge();
    await bridge.initialize();
    
    // Call Python backend (example)
    const version = await bridge.getVersion();
    
    // Stop spinner
    spinner.stop();
    
    // Format and display result
    if (isJsonOutput()) {
      // JSON output
      console.log(JSON.stringify({
        message: `Hello, ${name}!`,
        version
      }, null, 2));
    } else {
      // Human-readable output
      displaySection('Example Command');
      console.log(`Hello, ${chalk.cyan(name)}!`);
      console.log(`Ember version: ${version}`);
      displaySuccess('Command completed successfully');
    }
  } catch (error: any) {
    // Handle errors
    spinner.fail('Command failed');
    console.error(chalk.red('Error:'), error.message);
  }
}
```

Then register your command in `src/cli/index.ts`:

```typescript
// src/cli/index.ts
import { registerExampleCommand } from './commands/example';

// Register commands
registerVersionCommand(program);
registerProviderCommands(program);
registerModelCommands(program);
registerProjectCommands(program);
registerInvokeCommand(program);
registerConfigCommands(program);
registerExampleCommand(program); // Add your new command here
```

## Extending the Python Bridge

If you need to add new functionality to the Python bridge:

1. Add a method to the `EmberPythonBridge` interface in `src/cli/bridge/python-bridge.ts`
2. Implement the method in the `PythonBridgeImpl` class
3. Add corresponding Python code that will be executed by the bridge

Example:

```typescript
// Add to the EmberPythonBridge interface
/**
 * Run a custom function in the Python backend
 */
runCustomFunction(name: string, args: Record<string, any>): Promise<any>;

// Implement in the PythonBridgeImpl class
async runCustomFunction(name: string, args: Record<string, any>): Promise<any> {
  await this.ensureInitialized();
  
  const argsJson = JSON.stringify(args);
  
  return await this.bridge.eval`
import json
try:
    # Call the function dynamically
    result = getattr(service, ${name})(**json.loads(${argsJson}))
    json.dumps(result)
except Exception as e:
    json.dumps({"error": str(e)})
`;
}
```

## UI Components

The CLI uses several UI components to create a beautiful user experience:

- **Banner**: Displays the Ember CLI logo
- **Spinners**: Shows progress for async operations
- **Tables**: Formats tabular data
- **Colors**: Highlights important information
- **Emoji**: Adds visual cues

When creating new UI components, follow these guidelines:

1. Use the `chalk` library for colors
2. Use `ora` for spinners
3. Use `table` for tabular data
4. Use `emoji` for visual cues
5. Always respect the `--quiet` and `--no-color` flags

## Configuration Management

The CLI uses the `conf` library to store configuration. Key features:

- **Encryption**: Sensitive data is encrypted
- **Schema Validation**: Configuration follows a defined schema
- **Persistence**: Configuration is stored between runs

When accessing configuration, always use the `ConfigManager` class:

```typescript
// Get config manager singleton
const configManager = ConfigManager.getInstance();

// Get a setting
const value = configManager.getSetting('my.setting', defaultValue);

// Set a setting
configManager.setSetting('my.setting', newValue);
```

## Error Handling

Follow these guidelines for error handling:

1. Always catch exceptions in async functions
2. Use `try/catch` blocks around Python bridge calls
3. Show user-friendly error messages
4. Include technical details in debug mode
5. Use spinners to indicate progress/failure

Example:

```typescript
try {
  // Code that might throw
  const result = await riskyOperation();
  // Handle success
} catch (error: any) {
  // Handle error
  console.error(chalk.red('Error:'), error.message);
  if (isDebugMode()) {
    console.error(error.stack);
  }
}
```

## Testing

The CLI includes a comprehensive test suite:

- **Unit Tests**: Test individual components
- **Integration Tests**: Test command flows
- **Mock Tests**: Test with mocked Python bridge

To write tests:

1. Create test files in the `__tests__` directory
2. Use Jest for testing
3. Mock external dependencies
4. Test both success and failure cases

Example test:

```typescript
// __tests__/commands/version.test.ts
import { registerVersionCommand } from '../../src/cli/commands/version';
import { getPythonBridge } from '../../src/cli/bridge/python-bridge';

// Mock the Python bridge
jest.mock('../../src/cli/bridge/python-bridge');

describe('Version Command', () => {
  beforeEach(() => {
    // Setup mocks
    (getPythonBridge as jest.Mock).mockImplementation(() => ({
      initialize: jest.fn().mockResolvedValue(undefined),
      getVersion: jest.fn().mockResolvedValue('0.1.0')
    }));
  });
  
  it('should display version information', async () => {
    // Create a mock Commander instance
    const program = {
      command: jest.fn().mockReturnThis(),
      description: jest.fn().mockReturnThis(),
      option: jest.fn().mockReturnThis(),
      action: jest.fn().mockReturnThis()
    };
    
    // Register command
    registerVersionCommand(program as any);
    
    // Verify command was registered
    expect(program.command).toHaveBeenCalledWith('version');
    
    // Get the action callback
    const actionCallback = program.action.mock.calls[0][0];
    
    // Create a mock console.log
    const originalLog = console.log;
    console.log = jest.fn();
    
    // Run the action
    await actionCallback({});
    
    // Verify output
    expect(console.log).toHaveBeenCalled();
    expect(getPythonBridge().getVersion).toHaveBeenCalled();
    
    // Restore console.log
    console.log = originalLog;
  });
});
```

## Building and Publishing

To build and publish the CLI:

```bash
# Build the CLI
npm run build

# Test the build
node dist/index.js version

# Publish to npm
npm publish
```

## Style Guide

Follow these coding style guidelines:

1. Use TypeScript for all JavaScript code
2. Use async/await for asynchronous operations
3. Document all public functions and classes with JSDoc
4. Use SOLID principles for code organization
5. Use semantic versioning for releases

## Best Practices

1. **Security**: 
   - Never log API keys or sensitive information
   - Always encrypt stored credentials
   - Validate user input

2. **Performance**:
   - Minimize Python bridge calls
   - Batch operations when possible
   - Use streaming for large responses

3. **User Experience**:
   - Always show progress for long operations
   - Provide clear error messages
   - Include helpful tips and examples

4. **Code Quality**:
   - Write unit and integration tests
   - Use TypeScript types strictly
   - Document complex code
   - Follow the SOLID principles

## Troubleshooting Development Issues

### Python Bridge Issues

If you encounter problems with the Python bridge:

1. Check that Python 3.11+ is installed and accessible
2. Verify Ember AI is installed in the Python environment
3. Use debug mode to see Python errors
4. Check that Python bridge is being initialized correctly

### TypeScript Compilation Issues

For TypeScript errors:

1. Run `npm run lint` to find code issues
2. Check import paths (case-sensitive)
3. Ensure type definitions are correct
4. Run `tsc --noEmit` to type-check without building

## Getting Help

For development questions:

1. Check the existing code for examples
2. Review the documentation in code comments
3. Open an issue on GitHub for unanswered questions

## Contributing

We welcome contributions to the Ember CLI! Follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests for your changes
5. Run the test suite
6. Submit a pull request

## License

The Ember CLI is licensed under the MIT License.