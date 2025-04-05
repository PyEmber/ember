"""Mistral-7B-Instruct-v0.2 usage example using Hugging Face Inference API.

This module demonstrates how to use the Mistral-7B-Instruct-v0.2 model
through the Ember framework, showcasing instruction following capabilities
and chat-oriented generation.
"""

import logging
import os
from typing import Any, Dict, Optional, Union, cast

from ember.core.registry.model.base.schemas.chat_schemas import ChatResponse
from ember.core.registry.model.base.schemas.model_info import ModelInfo, ProviderInfo
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.initialization import initialize_registry
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model_service() -> ModelService:
    """Get the model service with Mistral-7B-Instruct-v0.2 registered.

    Returns:
        A ModelService instance with Mistral-7B-Instruct-v0.2 registered
    """
    # Initialize the registry
    registry = initialize_registry(auto_discover=True)
    service = ModelService(registry=registry)
    
    # Register Mistral-7B-Instruct-v0.2 if not already registered
    try:
        # Try to get the model - will raise an exception if not found
        service.get_model("huggingface:mistralai/Mistral-7B-Instruct-v0.2")
        logger.info("Mistral model already registered")
    except Exception:
        # Model not found, register it
        logger.info("Registering Mistral model")
        
        # Get API key from environment variable or use a default for testing
        api_key = os.environ.get("HUGGINGFACE_API_KEY", "your_api_key_here")
        
        # Create model info
        model_info = ModelInfo(
            id="huggingface:mistralai/Mistral-7B-Instruct-v0.2",
            name="mistralai/Mistral-7B-Instruct-v0.2",
            provider=ProviderInfo(name="HuggingFace", default_api_key=api_key),
            cost=ModelCost(
                input_cost_per_thousand=0.0,
                output_cost_per_thousand=0.0,
            ),
        )
        
        # Register the model
        registry.register_model(model_info)
    
    return service


def basic_instruction_example(service: ModelService) -> None:
    """Basic instruction following with Mistral-7B-Instruct-v0.2.
    
    This example shows how to provide instructions and get responses.
    
    Args:
        service: The model service with Mistral-7B-Instruct-v0.2 registered
    """
    try:
        # Get the model
        mistral_model = service.get_model("huggingface:mistralai/Mistral-7B-Instruct-v0.2")
        
        # Generate a response to an instruction
        response = mistral_model(
            "Explain what artificial intelligence is to a 10-year old child.",
            temperature=0.7,
            max_tokens=150
        )
        
        print("\n=== Basic Instruction ===")
        print(f"Response: {response.data}")
        print(f"Used {response.usage.total_tokens} tokens")
        
    except Exception as error:
        logger.exception("Error during basic instruction: %s", error)


def creative_writing_example(service: ModelService) -> None:
    """Creative writing example with Mistral-7B-Instruct-v0.2.
    
    This example demonstrates using the model for creative writing tasks.
    
    Args:
        service: The model service with Mistral-7B-Instruct-v0.2 registered
    """
    try:
        # Get the model
        mistral_model = service.get_model("huggingface:mistralai/Mistral-7B-Instruct-v0.2")
        
        # Create a short story prompt
        response = mistral_model(
            "Write a short story about a robot discovering emotions for the first time. Make it touching and meaningful.",
            temperature=0.8,  # Higher temperature for more creativity
            max_tokens=300
        )
        
        print("\n=== Creative Writing ===")
        print(f"Response: {response.data}")
        print(f"Used {response.usage.total_tokens} tokens")
        
    except Exception as error:
        logger.exception("Error during creative writing: %s", error)


def factual_qa_example(service: ModelService) -> None:
    """Factual Q&A example with Mistral-7B-Instruct-v0.2.
    
    This example tests the model's ability to answer factual questions.
    
    Args:
        service: The model service with Mistral-7B-Instruct-v0.2 registered
    """
    try:
        # Get the model
        mistral_model = service.get_model("huggingface:mistralai/Mistral-7B-Instruct-v0.2")
        
        # Ask a factual question
        response = mistral_model(
            "What are the main components of the solar system? List the planets in order from the sun.",
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=200
        )
        
        print("\n=== Factual Q&A ===")
        print(f"Response: {response.data}")
        print(f"Used {response.usage.total_tokens} tokens")
        
    except Exception as error:
        logger.exception("Error during factual Q&A: %s", error)


def main() -> None:
    """Run all Mistral-7B-Instruct-v0.2 examples."""
    print("Mistral-7B-Instruct-v0.2 Usage Examples")
    print("========================================")
    
    try:
        # Get the model service with Mistral registered
        service = get_model_service()
        
        # Run examples
        basic_instruction_example(service)
        creative_writing_example(service)
        factual_qa_example(service)
        
    except Exception as error:
        logger.exception("Error during examples: %s", error)


if __name__ == "__main__":
    main() 