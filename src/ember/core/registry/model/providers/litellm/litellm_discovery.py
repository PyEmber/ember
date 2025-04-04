import logging
import os
from typing import List, Dict, Any, Optional

from litellm import get_valid_models

from ember.core.exceptions import ModelDiscoveryError
from ember.core.registry.model.providers.base_discovery import BaseDiscoveryProvider

# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class LitellmDiscovery(BaseDiscoveryProvider):
    """
    Discovery provider for LiteLLM models.
    Retrieves models from the LiteLLM API using the latest LiteLLM SDK.
    """
    
    def __init__(
        self, api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the LitellmDiscovery instance.
        
        The API key is provided either as an argument or via the LITELLM_API_KEY environment variable.
        Optionally, a custom base URL (for example for Azure endpoints) may be specified.
        
        Args:
            api_key (Optional[str]): The LiteLLM API key for authentication.
            base_url (Optional[str]): An optional custom base URL for the LiteLLM API.
        
        Raises:
            ModelDiscoveryError: If the API key is not set.
        """
        self._api_key: Optional[str] = api_key or os.environ.get("LITELLM_API_KEY")
        if not self._api_key:
            raise ModelDiscoveryError(
                "LiteLLM API key is not set in config or environment"
            )
        
    def discover_models(self) -> List[Dict[str, Any]]:
        """
        Discover available LiteLLM models.

        Returns:
            List[Dict[str, Any]]: A list of discovered models with their details.
        """
        try:
            # Set the API key for LiteLLM
            litellm.api_key = self._api_key
            
            # Discover models
            valid_models = get_valid_models(check_provider_endpoint=True)
            logger.info("Discovered LiteLLM models: %s", valid_models)
            return valid_models
        except ModelDiscoveryError:
            raise
        except Exception as e:
            logger.error("Failed to discover LiteLLM models: %s", str(e))
            return []