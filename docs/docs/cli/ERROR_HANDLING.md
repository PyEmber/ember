# Error Handling in Ember CLI

Ember CLI includes a robust error handling system that provides structured errors, detailed information, and helpful suggestions to users. This document describes how error handling works and how to use it when developing new commands or features.

## Key Features

- **Structured Errors**: All errors extend a base `EmberCliError` class
- **Error Categorization**: Errors are categorized by type with specific error codes
- **Rich Error Information**: Errors can include suggestions, documentation links, and context
- **Python Error Mapping**: Python exceptions are properly mapped to TypeScript errors
- **Consistent Display**: Errors are displayed consistently across the CLI

## Error Class Hierarchy

The Ember CLI error classes follow this hierarchy:

```
EmberCliError (base class)
├── PythonBridgeError - For Python bridge communication issues
├── ModelError - For model and provider errors
├── ProjectError - For project-related errors
├── ValidationError - For user input validation errors
├── ConfigurationError - For configuration errors
├── AuthorizationError - For authentication and authorization errors
└── NetworkError - For network-related errors
```

## Error Codes

Errors are assigned unique error codes for identification and documentation:

- **1000-1999**: General CLI errors
- **2000-2999**: Python bridge errors
- **3000-3999**: Model and provider errors
- **4000-4999**: Project errors

## Using Error Handling in Commands

### Creating and Throwing Errors

When you need to create an error in your command:

```typescript
import { createModelError, ModelErrorCodes } from '../errors';

// Create and throw an error
throw createModelError(
  'Model not found: gpt-4',
  ModelErrorCodes.MODEL_NOT_FOUND,
  {
    suggestions: [
      'Run `ember models` to list available models',
      'Check that your OpenAI API key is set correctly'
    ]
  }
);
```

### Using the Try-Catch Helper

You can use the `tryCatch` helper to handle errors consistently:

```typescript
import { tryCatch } from '../errors';

await tryCatch(
  async () => {
    // Your code here
    const result = await someAsyncFunction();
    return result;
  },
  {}, // Error format options
  true // Exit process on error
);
```

### Handling Errors in Command Actions

For command actions, wrap your function in a try-catch block:

```typescript
.action(async (options) => {
  try {
    await commandFunction(options);
  } catch (error) {
    handleError(error, {}, true);
  }
});
```

## Error Format Options

When displaying errors with `handleError`, you can customize the output:

```typescript
handleError(error, {
  includeCode: true,       // Include error code in output
  includeSuggestions: true, // Include suggestions
  includeDocsLinks: true,   // Include documentation links
  includeStack: false,      // Include stack trace
  useColor: true,           // Use colored output
  asJson: false             // Format as JSON
});
```

## Python Error Translation

The Python bridge translates Python exceptions to TypeScript errors:

```
Python Exception         →  TypeScript Error
-----------------           ---------------
ModelNotFoundError      →  ModelError (MODEL_NOT_FOUND)
ProviderAPIError        →  ModelError (PROVIDER_API_ERROR)
ValidationError         →  ValidationError (INVALID_ARGUMENT)
FileNotFoundError       →  ValidationError (FILE_NOT_FOUND)
```

## Best Practices

1. **Use Specific Error Types**: Use the most specific error type for the situation
2. **Include Helpful Suggestions**: Always include suggestions to help users resolve the issue
3. **Add Context**: Include relevant context information for debugging
4. **Clean Up Resources**: Always clean up resources in error handlers (like spinners)
5. **Validate Early**: Validate user input early to avoid deeper errors
6. **Handle All Errors**: Never let errors go unhandled or display as generic errors
7. **Add Documentation Links**: For complex issues, include links to documentation

## Example: Command Error Handling

Here's a complete example of error handling in a command:

```typescript
async function myCommand(options: any): Promise<void> {
  const spinner = ora('Loading...').start();
  
  try {
    // Validate input
    if (!options.required) {
      throw createValidationError(
        'Missing required option',
        GeneralErrorCodes.MISSING_REQUIRED_ARGUMENT
      );
    }
    
    // Do something that might fail
    const result = await someAsyncOperation();
    
    // Handle success
    spinner.succeed('Operation completed');
    console.log(result);
  } catch (error) {
    // Stop spinner
    spinner.stop();
    
    // Rethrow (will be handled by command wrapper)
    throw error;
  }
}
```