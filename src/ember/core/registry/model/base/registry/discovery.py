import logging
import os
import threading
import time
from typing import Any, Dict, List

from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.providers.anthropic.anthropic_discovery import (
    AnthropicDiscovery,
)
from ember.core.registry.model.providers.base_discovery import (
    BaseDiscoveryProvider,
    ModelDiscoveryError,
)
from ember.core.registry.model.providers.deepmind.deepmind_discovery import (
    DeepmindDiscovery,
)
from ember.core.registry.model.providers.openai.openai_discovery import OpenAIDiscovery

from ember.core.registry.model.providers.huggingface.huggingface_discovery import (
    HuggingFaceDiscovery,
) 

logger: logging.Logger = logging.getLogger(__name__)
# Set default log level to WARNING to reduce verbosity
logger.setLevel(logging.WARNING)


class ModelDiscoveryService:
    """Service for aggregating and merging model metadata from various discovery providers.

    This service collects model information from different providers and merges the data
    with local configuration overrides. Results are cached (with thread-safe access) to
    reduce redundant network calls.
    """

    def __init__(self, ttl: int = 3600) -> None:
        """Initialize the ModelDiscoveryService.

        Args:
            ttl (int): Cache time-to-live in seconds. Defaults to 3600.
        """
        # Loading discovery providers dynamically based on available API keys
        self.providers: List[BaseDiscoveryProvider] = self._initialize_providers()

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_update: float = 0.0
        self.ttl: int = ttl
        # Using reentrant lock for thread safety to allow nested lock acquisition
        self._lock: threading.RLock = threading.RLock()

    def _initialize_providers(self) -> List[BaseDiscoveryProvider]:
        """Initialize discovery providers based on available API keys.

        This method creates provider instances with appropriate configuration:
        1. Checks for required API keys for each provider
        2. Only initializes providers with valid credentials
        3. Configures providers with appropriate settings

        Returns:
            List[BaseDiscoveryProvider]: A list of initialized discovery providers.
        """
        from typing import Callable, Dict, Optional, Tuple, Type

        # Define provider configurations
        ProviderConfig = Tuple[
            Type[BaseDiscoveryProvider], str, Optional[Callable[[], Dict[str, str]]]
        ]

        provider_configs: List[ProviderConfig] = [
            # Provider class, env var name, optional config function
            (
                OpenAIDiscovery,
                "OPENAI_API_KEY",
                lambda: {"api_key": os.environ.get("OPENAI_API_KEY", "")},
            ),
            (
                AnthropicDiscovery,
                "ANTHROPIC_API_KEY",
                lambda: {"api_key": os.environ.get("ANTHROPIC_API_KEY", "")},
            ),
            (
                DeepmindDiscovery,
                "GOOGLE_API_KEY",
                lambda: {"api_key": os.environ.get("GOOGLE_API_KEY", "")},
            ),
            (
                HuggingFaceDiscovery,
                "HUGGINGFACE_API_KEY", #Now searches for Hugging Face API key
                lambda: {"api_key": os.environ.get("HUGGINGFACE_API_KEY", "")},
            ),
        ]

        # Initializing providers with available credentials
        providers: List[BaseDiscoveryProvider] = []

        for provider_class, env_var_name, config_fn in provider_configs:
            api_key = os.environ.get(env_var_name)
            if api_key:
                try:
                    # Initialize with configuration if available
                    if config_fn:
                        config = config_fn()
                        if hasattr(provider_class, "configure"):
                            # If provider has a configure method, using it
                            instance = provider_class()
                            instance.configure(**config)  # type: ignore
                            providers.append(instance)
                        else:
                            # Otherwise trying to pass config to constructor
                            providers.append(provider_class(**config))  # type: ignore
                    else:
                        # Simple initialization without config
                        providers.append(provider_class())

                    logger.debug(
                        "%s found, initialized %s successfully",
                        env_var_name,
                        provider_class.__name__,
                    )
                except Exception as init_error:
                    logger.error(
                        "Failed to initialize %s: %s",
                        provider_class.__name__,
                        init_error,
                    )
            else:
                logger.info(
                    "%s not found, skipping %s", env_var_name, provider_class.__name__
                )

        if not providers:
            logger.warning(
                "No API keys found for any providers. "
                "Set one of %s environment variables to enable discovery.",
                ", ".join(env_var for _, env_var, _ in provider_configs),
            )

        return providers

    def discover_models(self) -> Dict[str, Dict[str, Any]]:
        """Discover models using registered providers, with caching based on TTL.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping from model ID to its metadata.

        Raises:
            ModelDiscoveryError: If no models can be discovered due to provider errors.
        """
        current_time: float = time.time()
        with self._lock:
            if self._cache and (current_time - self._last_update) < self.ttl:
                logger.info("Returning cached discovery results.")
                return self._cache.copy()

        aggregated_models: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []

        provider_threads = []
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            event = threading.Event()
            result_container: List[Dict[str, Any]] = []
            error_container: List[Exception] = []

            def fetch_with_timeout(
                prov=provider,
                pname=provider_name,
                ev=event,
                res=result_container,
                err=error_container,
            ):
                try:
                    start = time.time()
                    result = prov.fetch_models()
                    duration = time.time() - start
                    logger.info(f"Provider {pname} completed in {duration:.2f}s")
                    res.append(result)
                except Exception as e:
                    logger.error(f"Error fetching models from {pname}: {e}")
                    err.append(e)
                finally:
                    ev.set()

            t = threading.Thread(target=fetch_with_timeout)
            t.daemon = True
            provider_threads.append(
                (provider_name, event, result_container, error_container, t)
            )
            t.start()

        for (
            provider_name,
            event,
            result_container,
            error_container,
            t,
        ) in provider_threads:
            if not event.wait(15.0):
                logger.error(f"Timeout while fetching models from {provider_name}")
                errors.append(f"{provider_name}: Timeout after 15 seconds")
            else:
                if error_container:
                    errors.append(f"{provider_name}: {str(error_container[0])}")
                elif not result_container:
                    logger.error(f"No results from {provider_name}")
                    errors.append(f"{provider_name}: No results returned")
                else:
                    result = result_container[0]
                    logger.info(
                        f"Successfully received {len(result)} models from {provider_name}"
                    )
                    aggregated_models.update(result)

        if not aggregated_models and errors:
            raise ModelDiscoveryError(
                f"No models discovered. Errors: {'; '.join(errors)}"
            )

        with self._lock:
            self._cache = aggregated_models.copy()
            self._last_update = time.time()
            logger.info(
                f"Discovered {len(aggregated_models)} models: {list(aggregated_models.keys())}"
            )

        return aggregated_models.copy()

    def merge_with_config(
        self, discovered: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ModelInfo]:
        """Merge discovered model metadata with local configuration overrides.

        Local configuration (loaded from YAML) takes precedence over API-reported data.
        This method also ensures that environment variables are considered for API keys.

        Args:
            discovered: Discovered model metadata from providers.

        Returns:
            Dict[str, ModelInfo]: Mapping from model ID to merged ModelInfo objects.
        """
        import os

        from ember.core.config.schema import EmberSettings

        settings = EmberSettings()

        # Get models from local configuration
        local_models: Dict[str, Dict[str, Any]] = {}

        # Check if registry has models attribute before accessing it
        if hasattr(settings.registry, "models"):
            try:
                # Handle both actual Model objects and test objects
                local_models = {
                    model.id: model.model_dump() for model in settings.registry.models
                }
                logger.debug(
                    f"Loaded {len(local_models)} models from local configuration"
                )
                # Log model names for debugging
                for model_id, model_data in local_models.items():
                    logger.debug(
                        f"Local model: {model_id} - Name: {model_data.get('name', 'Unknown')}"
                    )
            except Exception as e:
                logger.error(f"Error loading models from settings: {e}")

        # Map provider prefixes to their environment variable keys
        provider_api_keys: Dict[str, str] = {
            "openai": os.environ.get("OPENAI_API_KEY", ""),
            "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
            "google": os.environ.get("GOOGLE_API_KEY", ""),
            "deepmind": os.environ.get("GOOGLE_API_KEY", ""),  # Uses same key as Google
            "huggingface": os.environ.get("HUGGINGFACE_API_KEY", ""),
            "mock": os.environ.get("MOCK_API_KEY", ""),  # For testing purposes
        }

        def get_provider_from_model_id(model_id: str) -> str:
            """Extract provider name from model ID."""
            if ":" in model_id:
                return model_id.split(":", 1)[0].lower()
            return "unknown"

        merged_models: Dict[str, ModelInfo] = {}

        for model_id, api_metadata in discovered.items():
            provider_name = get_provider_from_model_id(model_id)
            api_key = provider_api_keys.get(provider_name, "")

            if model_id in local_models:
                # Local configuration overrides API metadata except for API keys
                # which we take from environment if available.
                # Order is important here - local_models should override api_metadata
                merged_data: Dict[str, Any] = {**api_metadata, **local_models[model_id]}

                # Override with environment API key if available and not explicitly set
                if api_key and not merged_data.get("provider", {}).get(
                    "default_api_key"
                ):
                    if "provider" not in merged_data:
                        merged_data["provider"] = {}
                    if isinstance(merged_data["provider"], dict):
                        merged_data["provider"]["default_api_key"] = api_key
            else:
                # For discovered models not in local config, create reasonable defaults
                logger.debug(
                    "Model %s discovered via API but not in local config; using defaults with environment API key.",
                    model_id,
                )

                # Extract provider prefix from model ID
                provider_prefix = provider_name.capitalize()

                merged_data = {
                    "id": model_id,
                    "name": api_metadata.get(
                        "name", api_metadata.get("model_name", model_id.split(":")[-1])
                    ),
                    "cost": {
                        "input_cost_per_thousand": 0.0,
                        "output_cost_per_thousand": 0.0,
                    },
                    "rate_limit": {"tokens_per_minute": 0, "requests_per_minute": 0},
                    "provider": {
                        "name": provider_prefix,
                        "default_api_key": api_key,
                    },
                }

            # Validate model info and add to results if valid
            try:
                # Skip models without API keys - they can't be used anyway
                if not merged_data.get("provider", {}).get("default_api_key"):
                    logger.warning(
                        "Skipping model %s because no API key is available", model_id
                    )
                    continue

                # Create ModelInfo instance
                model_info = ModelInfo(**merged_data)
                merged_models[model_id] = model_info
                logger.debug("Successfully merged model info for %s", model_id)

            except Exception as validation_error:
                logger.error(
                    "Failed to merge model info for %s: %s", model_id, validation_error
                )

        if not merged_models:
            logger.warning(
                "No valid models found after merging with configuration. "
                "Check that API keys are set in environment variables and model schemas are valid."
            )

        return merged_models

    def refresh(self) -> Dict[str, ModelInfo]:
        """Force a refresh of model discovery and merge with local configuration."""
        # Invalidate the cache with minimal lock scope
        with self._lock:
            self._last_update = 0.0

        try:
            # Perform discovery outside the lock to prevent deadlocks
            discovered: Dict[str, Dict[str, Any]] = self.discover_models()

            # Ensure discovered is a proper dict before proceeding
            if discovered and not isinstance(discovered, dict):
                logger.error(
                    "Discovery returned non-dict result: %s (type: %s). Converting to empty dict.",
                    discovered,
                    type(discovered).__name__,
                )
                discovered = {}
            elif discovered and isinstance(discovered, dict):
                # Check each value to ensure it's a dict for merge_with_config to work properly
                for k, v in list(discovered.items()):
                    if not isinstance(v, dict):
                        logger.error(
                            "Model data for %s is not a dict: %s (type: %s). Removing from results.",
                            k,
                            v,
                            type(v).__name__,
                        )
                        discovered.pop(k)

            # Only merge and update cache in a separate lock acquisition
            with self._lock:
                merged = self.merge_with_config(discovered=discovered)
                # Only updating cache if discovery is successful
                self._cache = discovered.copy()
                self._last_update = time.time()
                return merged
        except Exception as e:
            logger.error("Failed to refresh model discovery: %s", e)
            # Return last known good cache if available - with minimal lock scope
            with self._lock:
                return (
                    self.merge_with_config(discovered=self._cache.copy())
                    if self._cache
                    else {}
                )

    def invalidate_cache(self) -> None:
        """Manually invalidate the cache, forcing a refresh on next discovery."""
        with self._lock:
            # Clear the cache dictionary
            if hasattr(self, "_cache"):
                self._cache.clear()
            # Reset the last update timestamp
            self._last_update = 0.0
            logger.info("Cache invalidated; next discovery will fetch fresh data.")

    async def discover_models_async(self) -> Dict[str, Dict[str, Any]]:
        """Asynchronously discover models using registered providers, with caching based on TTL."""
        import asyncio

        # Check cache with minimal lock scope
        current_time: float = time.time()
        with self._lock:
            if self._cache and (current_time - self._last_update) < self.ttl:
                logger.info("Returning cached discovery results.")
                return self._cache.copy()

        aggregated_models: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []

        async def fetch_from_provider(
            provider: BaseDiscoveryProvider,
        ) -> Dict[str, Any]:
            """Async wrapper for provider fetch with proper error handling."""
            provider_name = provider.__class__.__name__
            logger.info(f"Starting async model discovery for: {provider_name}")

            try:
                # Check if the provider has an async implementation
                if hasattr(provider, "fetch_models_async") and callable(
                    provider.fetch_models_async
                ):
                    # Use the async implementation directly with shorter timeout
                    logger.info(
                        f"Using native async implementation for {provider_name}"
                    )
                    provider_models = await asyncio.wait_for(
                        provider.fetch_models_async(), timeout=15
                    )
                else:
                    # Fall back to running the sync version in a thread pool
                    logger.info(f"Using thread pool for sync method of {provider_name}")
                    fetch_task = asyncio.create_task(
                        asyncio.get_event_loop().run_in_executor(
                            None, provider.fetch_models
                        )
                    )
                    provider_models = await asyncio.wait_for(fetch_task, timeout=15)

                logger.info(
                    f"Successfully received {len(provider_models)} models from {provider_name}"
                )
                return {
                    "success": True,
                    "provider": provider_name,
                    "models": provider_models,
                }
            except asyncio.TimeoutError:
                logger.error(
                    f"Async timeout while fetching models from {provider_name}"
                )
                return {
                    "success": False,
                    "provider": provider_name,
                    "error": "Timeout after 15 seconds",
                }
            except Exception as e:
                logger.error(f"Error in async fetch from {provider_name}: {e}")
                return {"success": False, "provider": provider_name, "error": str(e)}

        # Process each provider with independent timeout protection
        # This is better than gather which might wait for all tasks
        results = []
        for provider in self.providers:
            try:
                result = await fetch_from_provider(provider)
                results.append(result)
            except Exception as e:
                logger.error(f"Unexpected error in async discovery: {e}")
                # Add error entry but continue with other providers
                results.append(
                    {
                        "success": False,
                        "provider": provider.__class__.__name__,
                        "error": str(e),
                    }
                )

        # Process all provider results
        for result in results:
            if result["success"]:
                aggregated_models.update(result["models"])
            else:
                errors.append(f"{result['provider']}: {result['error']}")

        # Handle case where no models were discovered
        if not aggregated_models and errors:
            raise ModelDiscoveryError(
                f"No models discovered. Errors: {'; '.join(errors)}"
            )

        # Ensure aggregated_models is a proper dict before proceeding
        if not isinstance(aggregated_models, dict):
            logger.error(
                "Async discovery returned non-dict result: %s (type: %s). Converting to empty dict.",
                aggregated_models,
                type(aggregated_models).__name__,
            )
            aggregated_models = {}
        else:
            # Check each value to ensure it's a dict
            for k, v in list(aggregated_models.items()):
                if not isinstance(v, dict):
                    logger.error(
                        "Async model data for %s is not a dict: %s (type: %s). Removing from results.",
                        k,
                        v,
                        type(v).__name__,
                    )
                    aggregated_models.pop(k)

        # Update cache with minimal lock scope
        with self._lock:
            self._cache = aggregated_models.copy()
            self._last_update = time.time()
            if aggregated_models:
                logger.info(
                    f"Discovered {len(aggregated_models)} models: {list(aggregated_models.keys())}"
                )
            else:
                logger.info("No valid models discovered during async discovery")

        return aggregated_models.copy()
