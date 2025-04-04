"""
Application Context System for Ember Framework

This module implements Ember's application context architecture, providing a dependency
injection system and global service management. The context serves as the composition
root for framework services, while offering thread-safe singleton access when needed.

The design supports both direct dependency injection for components and convenient
global access for application code, with special accommodations for testing scenarios.

For complete architectural details, see ARCHITECTURE.md
"""

import logging
import os
import threading
from threading import Lock
from typing import Any, ClassVar, Optional

from ember.core.config.manager import ConfigManager, create_config_manager
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.services.usage_service import UsageService
from ember.core.registry.model.initialization import initialize_registry
from ember.core.utils.logging import configure_logging

# Re-import for patching to work correctly
import logging


class EmberAppContext:
    """
    Core dependency container for Ember's service architecture.

    The EmberAppContext serves as the composition root for the entire framework,
    centralizing all service dependencies and their lifecycle management. It implements
    the Dependency Inversion Principle by decoupling service consumers from their
    concrete implementations, enabling flexible configuration, testing, and extension.

    This design intentionally avoids service location within services themselves,
    instead passing dependencies explicitly through constructors. This ensures
    clear dependency graphs, simplified testing, and resilience to architectural changes.

    For detailed implementation guidance, see ARCHITECTURE.md
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        model_registry: ModelRegistry,
        usage_service: Optional[UsageService],
        logger: logging.Logger,
    ) -> None:
        """
        Initialize the application context with core services.

        Args:
            config_manager: Configuration manager implementation
            model_registry: Model registry service
            usage_service: Optional usage tracking service
            logger: Logger instance
        """
        self.config_manager = config_manager
        self.model_registry = model_registry
        self.usage_service = usage_service
        self.logger = logger


def create_ember_app(
    config_path: Optional[str] = None, verbose: bool = False
) -> EmberAppContext:
    """
    Creates a fully initialized EmberAppContext with all core services.

    This factory function serves as the primary composition root for the Ember framework,
    implementing the Builder pattern to construct and wire together all essential services.
    The step-by-step initialization process ensures proper dependency order and consistent
    error handling.

    Implementation details:
    1. Creates a centralized logger with consistent formatting
    2. Constructs the ConfigManager with fallback mechanisms for missing configs
    3. Injects all API keys into the environment with secure handling
    4. Builds the ModelRegistry with automatic provider discovery
    5. Instantiates and wires secondary services with their dependencies

    Args:
        config_path: Optional path to a configuration file. If not provided,
                    the system will search standard locations and use environment
                    variables as fallbacks.
        verbose: Whether to use verbose logging. If False (default), reduces verbosity
                for non-essential components like model discovery and HTTP libraries.

    Returns:
        A fully configured EmberAppContext instance with all services initialized.
    """
    # Configure logging first thing
    configure_logging(verbose=verbose)

    # Get the main ember logger
    logger = logging.getLogger("ember")

    # 1) Create the configuration manager
    config_manager = create_config_manager(config_path=config_path, logger=logger)
    # Configuration will be loaded on first access
    logger.debug("Configuration manager initialized")

    # 2) Initialize API keys from environment variables
    _initialize_api_keys_from_env(config_manager)

    # 3) Create and initialize the model registry
    model_registry = initialize_registry(
        config_manager=config_manager, auto_discover=True
    )
    logger.debug("Model registry initialized")

    # 4) Create additional services
    usage_service = UsageService(logger=logger)

    # 5) Return the aggregated application context
    return EmberAppContext(
        config_manager=config_manager,
        model_registry=model_registry,
        usage_service=usage_service,
        logger=logger,
    )


def _initialize_api_keys_from_env(config_manager: ConfigManager) -> None:
    """
    Initialize API keys from environment variables.

    Args:
        config_manager: Configuration manager to update
    """
    # Common API keys to check for
    env_keys = {
        "OPENAI_API_KEY": "openai",
        "ANTHROPIC_API_KEY": "anthropic",
        "GOOGLE_API_KEY": "google",
        "LITELLM_API_KEY": "litellm",
    }

    # Set API keys from environment if available
    for env_var, provider in env_keys.items():
        api_key = os.environ.get(env_var)
        if api_key:
            config_manager.set_provider_api_key(provider, api_key)


class EmberContext:
    """
    Thread-safe global access point for Ember's core service architecture.

    The EmberContext combines three critical design patterns to provide a robust
    foundation for the framework:

    1. Singleton Pattern: Guarantees a single global instance in standard usage,
       ensuring consistent state and memory efficiency

    2. Lazy Initialization: Defers resource creation until first access, avoiding
       circular dependencies and reducing startup overhead

    3. Test Mode Pattern: Provides a configurable escape hatch from the singleton
       constraint specifically for testing scenarios, enabling isolated test contexts

    This hybrid approach satisfies both production needs (centralized, consistent access)
    and testing requirements (isolation, deterministic setup/teardown). All operations
    are thread-safe through careful lock management, making the context suitable for
    concurrent applications.

    The EmberContext implements the Facade pattern, providing simplified access to
    the underlying service architecture through property accessors while encapsulating
    all initialization complexity.

    For internal use, consider direct dependency injection. This global context
    is primarily intended for application code that cannot easily receive injected
    dependencies.
    """

    _instance: ClassVar[Optional["EmberContext"]] = None
    _lock: ClassVar[Lock] = Lock()
    _app_context: Optional[EmberAppContext] = None
    _test_mode: ClassVar[bool] = False
    _thread_local: ClassVar = threading.local()

    # Backwards compatibility attributes for transition
    model_registry: ModelRegistry

    @classmethod
    def enable_test_mode(cls) -> None:
        """
        Enables isolated context instances for concurrent testing environments.

        This method reconfigures the EmberContext to bypass its singleton constraint,
        allowing each test to create completely isolated instances without interference.
        After enabling test mode, calls to EmberContext() will produce independent contexts
        with separate service instances and configurations.

        Test mode is critical for:
        - Parallel test execution without shared state contamination
        - Dependency isolation for deterministic test outcomes
        - Preventing test order dependencies
        - Avoiding deadlocks from concurrent singleton access

        Important implementation notes:
        - This is a class-level operation affecting all future instantiations
        - Existing instances are not affected, but their references are cleared
        - Thread-local storage is reset to prevent cross-test leakage

        This method should typically be called in test setup fixtures or
        at the beginning of test modules.
        """
        cls._test_mode = True
        # Clear instance to ensure fresh instances even after a previous singleton was created
        cls._instance = None
        if hasattr(cls._thread_local, "instances"):
            delattr(cls._thread_local, "instances")

    @classmethod
    def disable_test_mode(cls) -> None:
        """
        Disable test mode and return to normal singleton behavior.
        """
        cls._test_mode = False
        cls._instance = None
        if hasattr(cls._thread_local, "instances"):
            delattr(cls._thread_local, "instances")

    @classmethod
    def clear_test_context(cls) -> None:
        """
        Alias for disable_test_mode for backward compatibility with tests.
        """
        cls.disable_test_mode()

    def __getattr__(self, name: str) -> Any:
        """
        Fallback attribute access that delegates to the app_context.

        This enables backwards compatibility for code written against
        the old global registry pattern.

        Args:
            name: Attribute name

        Returns:
            The requested attribute from the app_context

        Raises:
            AttributeError: If the attribute doesn't exist on app_context
        """
        if self._app_context is None:
            self._app_context = create_ember_app()

        if hasattr(self._app_context, name):
            return getattr(self._app_context, name)

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __new__(cls) -> "EmberContext":
        if cls._test_mode:
            # In test mode, always create a new instance
            instance = super().__new__(cls)
            instance._app_context = None
            return instance

        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._app_context = None
            return cls._instance

    @classmethod
    def initialize(
        cls,
        config_path: Optional[str] = None,
        app_context: Optional[EmberAppContext] = None,
    ) -> "EmberContext":
        """
        Initializes or reconfigures the global context with explicit configuration.

        This method serves as the canonical initialization point for the EmberContext,
        supporting both automatic configuration and explicit dependency injection
        scenarios. It handles thread synchronization, singleton enforcement (when not
        in test mode), and lifecycle management.

        The implementation follows a clear priority order for initialization sources:
        1. If app_context is provided, it is used directly (full DI scenario)
        2. If only config_path is provided, a new app_context is created using that path
        3. If neither is provided, default configuration discovery is used

        This flexibility enables multiple initialization patterns:
        - Standard usage: EmberContext.initialize()
        - Custom config: EmberContext.initialize(config_path="/path/to/config.yaml")
        - Mock services: EmberContext.initialize(app_context=mock_context)

        Thread safety is guaranteed through careful lock management, making this
        method safe to call from concurrent initialization code.

        Args:
            config_path: Optional path to a YAML configuration file for initializing
                       a new app context when one isn't explicitly provided
            app_context: Optional pre-configured EmberAppContext for dependency
                       injection scenarios, typically in testing or when services
                       need custom configuration

        Returns:
            The initialized EmberContext instance, providing a typed interface to
            the underlying services. In normal mode, this is the global singleton;
            in test mode, it's a new independent instance.
        """
        if cls._test_mode:
            # In test mode, create a new instance
            instance = cls()

            # Set app_context or create a new one
            if app_context is not None:
                instance._app_context = app_context
            else:
                instance._app_context = create_ember_app(config_path=config_path)

            return instance

        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()

            # If app_context is provided, use it directly
            if app_context is not None:
                cls._instance._app_context = app_context
            # Otherwise create a new one if needed
            elif cls._instance._app_context is None:
                cls._instance._app_context = create_ember_app(config_path=config_path)

            return cls._instance

    @classmethod
    def get(cls) -> "EmberContext":
        """
        Get the current global context, initializing if needed.

        Returns:
            The EmberContext instance (singleton in normal mode, new in test mode)
        """
        if cls._test_mode:
            # In test mode, create a new instance
            return cls.initialize()

        with cls._lock:
            if cls._instance is None:
                return cls.initialize()
            return cls._instance

    @property
    def app_context(self) -> EmberAppContext:
        """
        Get the EmberAppContext instance, initializing if needed.

        Returns:
            The EmberAppContext instance

        Raises:
            RuntimeError: If accessed before initialization
        """
        if self._app_context is None:
            self._app_context = create_ember_app()
        return self._app_context

    @property
    def registry(self) -> ModelRegistry:
        """Access the model registry service."""
        return self.app_context.model_registry

    @property
    def config_manager(self) -> ConfigManager:
        """Access the configuration manager."""
        return self.app_context.config_manager

    @property
    def usage_service(self) -> Optional[UsageService]:
        """Access the usage tracking service."""
        return self.app_context.usage_service

    @property
    def logger(self) -> logging.Logger:
        """Access the logger instance."""
        return self.app_context.logger

    @classmethod
    def reset(cls) -> None:
        """
        Reset the EmberContext to its initial state.

        This is primarily useful for testing to ensure a clean state.
        """
        cls._instance = None
        if hasattr(cls._thread_local, "instances"):
            delattr(cls._thread_local, "instances")


def get_ember_context() -> EmberContext:
    """
    Retrieves the global EmberContext instance with guaranteed initialization.

    This factory function serves as the primary entry point for application code
    to access Ember's service infrastructure. It encapsulates the complexity of
    context management, ensuring that callers always receive a valid, initialized
    context without needing to understand the underlying initialization mechanisms.

    Key features of this function:
    - Transparent handling of lazy initialization
    - Consistent thread-safety guarantees
    - Support for both singleton and test modes
    - Zero configuration required for standard usage

    This function implements the Service Locator pattern in its most disciplined form,
    providing centralized access while minimizing the drawbacks typically associated
    with service location. It should be used in application code that cannot easily
    receive dependencies through normal injection.

    Example usage:
        context = get_ember_context()
        model_service = context.model_service
        result = model_service.invoke_model("gpt-4", "Hello world")

    Returns:
        A fully initialized EmberContext instance with access to all framework services.
        In standard operation, this will always be the same singleton instance.
    """
    return EmberContext.get()


__all__ = [
    "EmberAppContext",
    "create_ember_app",
    "EmberContext",
    "get_ember_context",
]

# Alias for backward compatibility
get_app_context = get_ember_context
