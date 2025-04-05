#TODO Just a placeholder implementation, need to implement discovery by author later
"""Discovery mechanism for Hugging Face models.

This module provides functionality to discover and register Hugging Face models
available through the Hugging Face Hub.
"""

import logging
from typing import Dict, List, Optional, Set

from ember.core.registry.model.providers.base_discovery import (
    BaseDiscoveryProvider,
    ModelDiscoveryError
)
from ember.core.registry.model.base.schemas.model_info import ModelInfo, ProviderInfo

logger = logging.getLogger(__name__)


class HuggingFaceDiscovery(BaseDiscoveryProvider):
    """Discovery implementation for Hugging Face models.
    
    This class provides methods to discover models available through the
    Hugging Face Hub and register them with the Ember model registry.
    """
    
    PROVIDER_NAME = "HuggingFace"
    
    def discover_models(self) -> List[ModelInfo]:
        """Discover available Hugging Face models.
        
        Returns:
            List[ModelInfo]: A list of model information objects for discovered models.
        """
        logger.info("Discovering Hugging Face models...")
        
        # This is a simplified implementation
        # In a real implementation, you might query the Hugging Face API
        # to get a list of popular or featured models
        
        # For now, just return a predefined list of popular models
        models = [
            # Prioritize the Mistral Instruct model
            self._create_model_info("mistralai/Mistral-7B-Instruct-v0.2"),
            self._create_model_info("meta-llama/Llama-2-7b-chat-hf"),
            # Keep the base model for comparison
            self._create_model_info("mistralai/Mistral-7B-v0.3"),
            self._create_model_info("google/gemma-7b-it"),
        ]
        
        logger.info("Discovered %d Hugging Face models", len(models))
        return models
    
    def _create_model_info(self, model_name: str) -> ModelInfo:
        """Create model information for a Hugging Face model.
        
        Args:
            model_name: The name of the model on the Hugging Face Hub.
            
        Returns:
            ModelInfo: The model information object.
        """
        return ModelInfo(
            id=f"huggingface:{model_name}",
            name=model_name,
            provider=ProviderInfo(
                name=self.PROVIDER_NAME,
                # API key will be filled in from environment or config
                default_api_key=None,
            ),
        ) 