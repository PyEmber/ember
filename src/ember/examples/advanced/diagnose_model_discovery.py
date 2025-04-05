"""Diagnose Model Discovery Issues in Ember

This script is designed to pinpoint exactly why model auto-discovery isn't working.
It adds detailed logging and performs targeted tests to identify issues in the model
discovery process.

To run:
    uv run python src/ember/examples/advanced/diagnose_model_discovery.py
"""

import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(name)s: %(message)s")
logger = logging.getLogger("model_discovery_diagnostics")

# Add src to path if needed
src_path = str(Path(__file__).parent.parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)
    logger.debug(f"Added {src_path} to sys.path")

# Import with careful error handling
try:
    from ember.api import models
    from ember.core.registry.model.base.registry.discovery import ModelDiscoveryService
    from ember.core.registry.model.base.registry.model_registry import ModelRegistry
    from ember.core.registry.model.providers.base_discovery import BaseDiscoveryProvider
    from ember.core.registry.model.providers.registry import PROVIDER_REGISTRY

    logger.debug("Successfully imported core model registry modules")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)


def check_api_keys():
    """Check if API keys are set in environment variables."""
    keys_to_check = {"OPENAI_API_KEY": "OpenAI", "ANTHROPIC_API_KEY": "Anthropic"}

    logger.info("Checking for API keys...")
    for key, provider in keys_to_check.items():
        value = os.environ.get(key)
        if value:
            logger.info(f"✅ {key} is set for {provider}")
        else:
            logger.warning(f"❌ {key} is NOT set for {provider}")


def check_config_paths():
    """Check if config files exist at expected paths."""
    logger.info("Checking configuration file paths...")

    # Define paths to check
    config_paths = [
        "src/ember/core/registry/model/config/model_registry_config.yaml",
        "src/ember/core/registry/model/providers/openai/openai_config.yaml",
        "src/ember/core/registry/model/providers/anthropic/anthropic_config.yaml",
    ]

    base_dir = Path(__file__).parent.parent.parent.parent.parent

    for path in config_paths:
        full_path = base_dir / path
        if full_path.exists():
            logger.debug(f"✅ Config file exists: {full_path}")
            # Check the content of the config file for path references
            with open(full_path, "r") as f:
                content = f.read()
                logger.debug(f"Config content for {full_path}:\n{content}")
        else:
            logger.error(f"❌ Config file NOT found: {full_path}")

            # Try to find similar files
            parent_dir = full_path.parent
            if parent_dir.exists():
                similar_files = list(parent_dir.glob("*.yaml"))
                if similar_files:
                    logger.info(f"Similar files in {parent_dir}:")
                    for file in similar_files:
                        logger.info(f"  - {file.name}")


def check_provider_registry():
    """Check the provider registry population."""
    logger.info("Checking provider registry...")

    try:
        # Check the PROVIDER_REGISTRY dictionary
        logger.debug(f"PROVIDER_REGISTRY contents: {PROVIDER_REGISTRY}")

        if not PROVIDER_REGISTRY:
            logger.error("❌ PROVIDER_REGISTRY is empty")

            # Try to find provider decorators
            try:
                from ember.core.registry.model.providers import (
                    anthropic_provider,
                    openai_provider,
                )

                logger.debug(f"Anthropic provider module: {anthropic_provider}")
                logger.debug(f"OpenAI provider module: {openai_provider}")

                # Check if providers are correctly decorated
                for name, module in [
                    ("Anthropic", anthropic_provider),
                    ("OpenAI", openai_provider),
                ]:
                    for item_name in dir(module):
                        item = getattr(module, item_name)
                        if inspect.isclass(item):
                            logger.debug(f"Found class {item_name} in {name} provider")
                            if hasattr(item, "__provider_name__"):
                                logger.debug(
                                    f"Class {item_name} has __provider_name__ = {getattr(item, '__provider_name__')}"
                                )
            except ImportError as e:
                logger.error(f"Failed to import provider modules: {e}")
        else:
            logger.info(
                f"✅ PROVIDER_REGISTRY contains {len(PROVIDER_REGISTRY)} providers"
            )
            for provider_name, provider_class in PROVIDER_REGISTRY.items():
                logger.info(f"  - {provider_name}: {provider_class}")
    except Exception as e:
        logger.error(f"❌ Error checking provider registry: {e}")


def test_discovery_imports():
    """Test import paths for discovery modules."""
    logger.info("Testing discovery module imports...")

    import_paths = [
        "ember.core.registry.model.providers",
        "ember.core.registry.model.providers.anthropic",
        "ember.core.registry.model.providers.openai",
        "ember.core.registry.model.providers.anthropic.anthropic_discovery",
        "ember.core.registry.model.providers.openai.openai_discovery",
    ]

    for path in import_paths:
        try:
            module = importlib.import_module(path)
            logger.debug(f"✅ Successfully imported {path}")

            # For discovery modules, check what they contain
            if "discovery" in path:
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    if inspect.isclass(item) and issubclass(
                        item, BaseDiscoveryProvider
                    ):
                        logger.debug(f"Found discovery class: {item_name}")
                        # Print the fetch_models method source
                        if hasattr(item, "fetch_models"):
                            fetch_method = getattr(item, "fetch_models")
                            logger.debug(f"fetch_models method: {fetch_method}")
                            if hasattr(fetch_method, "__code__"):
                                source_lines = inspect.getsourcelines(fetch_method)
                                logger.debug(f"fetch_models source: {source_lines[0]}")
        except ImportError as e:
            logger.error(f"❌ Failed to import {path}: {e}")

            # Try to find similar modules
            parent_path = ".".join(path.split(".")[:-1])
            try:
                parent_module = importlib.import_module(parent_path)
                logger.debug(f"Parent module {parent_path} exists")
                logger.debug(f"Contents of {parent_path}: {dir(parent_module)}")
            except ImportError:
                logger.error(f"Parent module {parent_path} doesn't exist")


def test_individual_discovery():
    """Test each discovery provider individually."""
    logger.info("Testing individual discovery providers...")

    discovery_classes = [
        (
            "ember.core.registry.model.providers.anthropic.anthropic_discovery",
            "AnthropicDiscovery",
        ),
        (
            "ember.core.registry.model.providers.openai.openai_discovery",
            "OpenAIDiscovery",
        ),
    ]

    for module_path, class_name in discovery_classes:
        try:
            module = importlib.import_module(module_path)
            discovery_class = getattr(module, class_name, None)

            if discovery_class:
                logger.debug(f"Found discovery class {class_name} in {module_path}")

                # Try instantiating the discovery class
                try:
                    discovery_instance = discovery_class()
                    logger.debug(f"Successfully instantiated {class_name}")

                    # Try fetching models
                    try:
                        logger.debug(f"Calling fetch_models() on {class_name}...")
                        models = discovery_instance.fetch_models()
                        logger.debug(f"fetch_models() returned: {models}")
                    except Exception as e:
                        logger.error(
                            f"❌ Error calling fetch_models() on {class_name}: {e}"
                        )
                except Exception as e:
                    logger.error(f"❌ Error instantiating {class_name}: {e}")
            else:
                logger.error(f"❌ Could not find class {class_name} in {module_path}")
        except ImportError as e:
            logger.error(f"❌ Could not import {module_path}: {e}")


def test_model_registry_initialization():
    """Test the full model registry initialization process."""
    logger.info("Testing model registry initialization...")

    try:
        # First check the API approach
        logger.debug("Testing models.initialize_registry with auto_discover=True...")
        try:
            registry = models.initialize_registry(auto_discover=True)
            logger.debug(f"Registry initialized: {registry}")

            # Check which models were discovered
            model_ids = registry.list_models()
            logger.debug(f"Discovered models: {model_ids}")

            if not model_ids:
                logger.warning("❌ No models were discovered automatically")
        except Exception as e:
            logger.error(f"❌ Error initializing registry through API: {e}")

        # Test direct instantiation
        logger.debug("Testing direct ModelRegistry instantiation...")
        try:
            registry = ModelRegistry()
            logger.debug(f"Registry directly instantiated: {registry}")

            # Try to discover models manually
            logger.debug("Manually running discovery process...")
            registry.discover_models()

            # Check which models were discovered
            model_ids = registry.list_models()
            logger.debug(f"Manually discovered models: {model_ids}")

            if not model_ids:
                logger.warning("❌ No models were discovered through manual discovery")
        except Exception as e:
            logger.error(f"❌ Error with direct registry initialization: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error in registry initialization test: {e}")


def check_model_registry_implementation():
    """Check the implementation of model discovery in ModelRegistry."""
    logger.info("Examining ModelRegistry implementation...")

    try:
        # Get source for discover_models method
        from ember.core.registry.model.base.registry.model_registry import ModelRegistry

        discover_method = ModelRegistry.discover_models

        if hasattr(discover_method, "__code__"):
            source_lines, line_no = inspect.getsourcelines(discover_method)
            logger.debug(
                f"discover_models source (line {line_no}):\n{''.join(source_lines)}"
            )

        # Check the factory implementation
        from ember.core.registry.model.base.registry.factory import (
            create_provider_instance,
        )

        factory_method = create_provider_instance

        if hasattr(factory_method, "__code__"):
            source_lines, line_no = inspect.getsourcelines(factory_method)
            logger.debug(
                f"create_provider_instance source (line {line_no}):\n{''.join(source_lines)}"
            )
    except Exception as e:
        logger.error(f"❌ Error examining ModelRegistry implementation: {e}")


def main():
    """Run all diagnostic tests."""
    logger.info("======= STARTING MODEL DISCOVERY DIAGNOSTICS =======")

    # Run all diagnostic checks
    check_api_keys()
    check_config_paths()
    check_provider_registry()
    test_discovery_imports()
    check_model_registry_implementation()
    test_individual_discovery()
    test_model_registry_initialization()

    logger.info("======= DIAGNOSTICS COMPLETE =======")

    # Provide a summary of findings
    logger.info("\n===== DIAGNOSTIC SUMMARY =====")
    logger.info(
        "Review the log output above to identify where the model discovery process is failing."
    )
    logger.info("Key areas to check:")
    logger.info("1. API keys - are they present and correctly formatted?")
    logger.info("2. Configuration files - do they exist at the expected paths?")
    logger.info("3. Provider registry - is it being populated correctly?")
    logger.info("4. Import paths - are all modules being found correctly?")
    logger.info("5. Discovery implementations - are they properly implemented?")
    logger.info("6. ModelRegistry initialization - is it properly configured?")


if __name__ == "__main__":
    main()
