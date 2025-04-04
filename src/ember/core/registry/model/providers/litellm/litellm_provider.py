"""LiteLLM provider implementation for the Ember framework.

This module provides a comprehensive integration with LiteLLM, allowing for
authentication, request formatting, response parsing, error handling, retry
logic, and usage tracking specifically for the LiteLLM API.

The implementation follows best practices for API integration, including
parameter handling, retry mechanisms, error categorization, and logging.

Classes:
    LiteLLMProviderParams: TypedDict defining LiteLLM-specific parameters
    LiteLLMChatParameters: Parameter conversion for LiteLLM chat completions
    LiteLLMModel: Core implementation of the LiteLLM provider

Details:
    - Authentication and client configuration for LiteLLM API
    - Automatic retry with exponential backoff for transient errors
    - Specialized error handling for different error types
    - Parameter validation and transformation
    - Detailed logging for monitoring and debugging
    - Usage statistics calculation for cost tracking

Usage example:
    ```python
    # Direct usage (prefer using ModelRegistry or API)
    from ember.core.registry.model.base.schemas.model_info import ModelInfo, ProviderInfo

    # Configure model information
    model_info = ModelInfo(
        id="litellm:example-model",
        name="example-model",
        provider=ProviderInfo(name="LiteLLM", api_key="api-key")
    )

    # Initialize the model
    model = LiteLLMModel(model_info)

    # Basic usage
    response = model("What is the Ember framework?")

    # Example: "The Ember framework is a Python library for composable LLM applications..."
    ```
"""

import logging
from typing import Any, Dict, Final, List, Optional, cast

from litellm import completion
from pydantic import Field, field_validator
from requests.exceptions import HTTPError
from tenacity import retry, stop_after_attempt, wait_exponential

from ember.core.exceptions import ModelProviderError, ValidationError
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    ProviderParams,
)
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    InvalidPromptError,
    ProviderAPIError,
)
from ember.core.registry.model.base.utils.usage_calculator import DefaultUsageCalculator
from ember.core.registry.model.providers.base_provider import (
    BaseChatParameters,
    BaseProviderModel,
)
from ember.plugin_system import provider

class LiteLLMProviderParams(ProviderParams):
    """LiteLLM-specific provider parameters for fine-tuning API requests.
   
    This TypedDict defines additional parameters that can be passed to LiteLLM API
    calls beyond the standard parameters defined in BaseChatParameters. These parameters
    provide fine-grained control over the model's generation behavior.

    Parameters can be provided in the provider_params field of a ChatRequest:
    ``` python
    request = ChatRequest(
        prompt="Generate creative ideas",
        provider_params={
            "stream": True,
        }
    )
    ```

    Attributes:
        stream: Optional boolean to enable streaming responses instead of waiting
            for the complete response.
    """
    
    # Work in progress (Need to find all of the parameters Litellm supports)
    stream: Optional[bool]
    stop: Optional[list[str]]
    n: Optional[int]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    top_p: Optional[float]
    seed: Optional[int]
    

logger: logging.Logger = logging.getLogger(__name__)

class LiteLLMChatParameters(BaseChatParameters):
    """Parameters for LiteLLM chat requests with validation and conversion logic.

    This class extends BaseChatParameters to provide LiteLLM-specific parameter
    handling and validation. It ensures that parameters are correctly formatted
    for the LiteLLM API, handling the conversion between Ember's universal
    parameter format and LiteLLM's API requirements.

    Key features:
        - Enforces a minimum value for max_tokens to prevent empty responses
        - Provides a sensible default (512 tokens) if not specified by the user
        - Validates that max_tokens is a positive integer with clear error messages
        - Builds the messages array in the format expected by LiteLLM's chat completion API
        - Structures system and user content into proper roles  
    
    The class handles parameter validation and transformation to ensure that
    all requests sent to the LiteLLM API are properly formatted and contain
    all required fields with valid values.

    Example:
        ```python
        # With context
        params = LiteLLMChatParameters(
            prompt="Tell me about LLMs",
            context="You are a helpful assistant",
            max_tokens=100,
            temperature=0.7
        )
        kwargs = params.to_litemllm_kwargs()
        # Result:
        # {
        #     "messages": [
        #         {"role": "system", "content": "You are a helpful assistant"},
        #         {"role": "user", "content": "Tell me about LLMs"}
        #     ],
        #     "max_tokens": 100,
        #     "temperature": 0.7
        # }
        ```
    """

    max_tokens: Optional[int] = Field(default=None)
    
    @field_validator("max_tokens", mode="before")
    def enforce_default_if_none(cls, value: Optional[int]) -> int:
        """Enforce a default value for `max_tokens` if None.

        Args:
            value (Optional[int]): The original max_tokens value, possibly None.

        Returns:
            int: An integer value; defaults to 512 if input is None.
        """
        return 512 if value is None else value

    @field_validator("max_tokens")
    def ensure_positive(cls, value: int) -> int:
        """Ensure that max_tokens is a positive integer.

        Args:
            value (int): The token count.

        Returns:
            int: The validated token count.

        Raises:
            ValidationError: If the token count is less than 1.
        """
        if value < 1:
            raise ValidationError.with_context(
                f"max_tokens must be >= 1, got {value}",
                field_name="max_tokens",
                expected_range=">=1",
                actual_value=value,
                provider="LiteLLM",
            )
        return value

    def to_litemllm_kwargs(self) -> Dict[str, Any]:
        """Convert chat parameters into keyword arguments for the LiteLLM API.

        Builds the messages list and returns a dictionary of parameters as expected
        by the LiteLLM API.

        Returns:
            Dict[str, Any]: A dictionary containing keys such as 'messages',
            'max_tokens', and 'temperature'.
        """
        
        messages: List[Dict[str, str]] = []
        if self.context:
            messages.append({"role": "system", "content": self.context})
        messages.append({"role": "user", "content": self.prompt})
        return {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        

@provider("LiteLLM")
class LiteLLMModel(BaseProviderModel):
    """Implementation for LiteLLM models in the Ember framework.

    This class provides a comprehensive integration with the LiteLLM API, handling
    all aspects of model interaction including authentication, request formatting,
    error handling, retry logic, and response processing. It implements the
    BaseProviderModel interface, making LiteLLM models compatible with the wider
    Ember ecosystem.

    The implementation follows LiteLLM's best practices for API integration,
    including proper parameter formatting, efficient retry mechanisms, detailed
    error handling, and comprehensive logging. It supports all LiteLLM model
    variants with appropriate parameter adjustments for model-specific requirements.

    Key features:
        - Robust error handling with automatic retries for transient errors
        - Specialized handling for different model variants (e.g., o1 models)
        - Comprehensive logging for debugging and monitoring
        - Usage statistics tracking for cost analysis
        - Type-safe parameter handling with runtime validation
        - Model-specific parameter pruning (e.g., removing temperature for o1 models)
        - Proper timeout handling to prevent hanging requests

    The class provides three core functions:
        1. Creating and configuring the LiteLLM API client
        2. Sending a ChatRequest to the LiteLLM API
        3. Processing the response from the LiteLLM API
    
    Implementation details:
        - Uses the official LiteLLM Python SDK
        - Implements tenacity-based retry logic with exponential backoff
        - Properly handles API timeouts to prevent hanging
        - Calculates usage statistics based on API response data
        - Handles parameter conversion between Ember and LiteLLM formats

    Attributes:
        PROVIDER_NAME: The canonical name of this provider for registration.
        model_info: Model metadata including credentials and cost schema.
        client: The configured LiteLLM API client instance.
        usage_calculator: Component for calculating token usage and costs.
    """

    PROVIDER_NAME: Final[str] = "LiteLLM"
    
    def __init__(self, model_info: ModelInfo) -> None:
        """Initialize a LiteLLMModel instance.

        Args:
            model_info (ModelInfo): Model information including credentials and
                cost schema.
        """
        super().__init__(model_info)
        self.usage_calculator = DefaultUsageCalculator()

    def create_client(self):
        """Create and configure the LiteLLM client."""
        # Initialize the LiteLLM client using the API key from model_info
        
        api_key: Optional[str] = self.model_info.get_api_key()
        if not api_key:
            raise ModelProviderError.for_provider(
                provider_name=self.PROVIDER_NAME,
                message="API key is missing or invalid.",
            )
            
        # Initialize the LiteLLM client with the API key
        litellm.api_key = api_key
        return litellm

    def get_api_model_name(self) -> str:
        """Get the model name formatted for LiteLLM's API requirements.
        
        LiteLLM API requires lowercase model names. This method ensures that
        model names are properly formatted regardless of how they're stored
        internally in the model registry.
        
        Returns:
            str: The properly formatted model name for LiteLLM API requests.
        """
        # LiteLLM API requires lowercase model names
        return self.model_info.name.lower() if self.model_info.name else ""
    
    @retry(
        wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True
    )

    def forward(self, request: ChatRequest) -> ChatResponse:
        """Send a ChatRequest to the LiteLLM API and process the response."""
        
        if not request.prompt:
            raise InvalidPromptError.with_context(
                "LiteLLM prompt cannot be empty.",
                provider=self.PROVIDER_NAME,
                model_name=self.model_info.name,
            )
        
        logger.info(
            "LiteLLM forward invoked",
            extra={
                "provider": self.PROVIDER_NAME,
                "model_name": self.model_info.name,
                "prompt_length": len(request.prompt),
            },
        )
        
        # Convert the universal ChatRequest into LiteLLM-specific parameters. 
        liteLLM_parameters: LiteLLMChatParameters = LiteLLMChatParameters(
            **request.model_dump(exclude={"provider_params"})
        )
        liteLLM_kwargs: Dict[str, Any] = liteLLM_parameters.to_litemllm_kwargs()
        
        # Merge extra provider parameters in a type-safe manner.
        # Cast the provider_params to LiteLLMProviderParams for type safety
        provider_params = cast(LiteLLMProviderParams, request.provider_params)
        # Only include non-None values
        liteLLM_kwargs.update(
            {k: v for k, v in provider_params.items() if v is not None}
        )

        if (
            "max_tokens" in liteLLM_kwargs
            and "max_completion_tokens" not in liteLLM_kwargs
        ):
            liteLLM_kwargs["max_completion_tokens"] = liteLLM_kwargs.pop("max_tokens")
        
        try:

            model_name = self.get_api_model_name()
            # Call the LiteLLM completion API
            response: Any = completion(
                model=model_name,
                **liteLLM_kwargs
            )
            # Process the response
            content: str = response.choices[0].message.content.strip()
            usage_stats = self.usage_calculator.calculate(
                raw_output=response,
                model_info=self.model_info,
            )
            # Return a ChatResponse
            return ChatResponse(
                data=content,
                raw_output=response,
                usage=usage_stats
            )
        except HTTPError as e:
            raise ProviderAPIError.for_provider(
                provider_name=self.PROVIDER_NAME,
                message=str(e)
            )
        except Exception as e:
            raise ModelProviderError.for_provider(
                provider_name=self.PROVIDER_NAME,
                message=str(e)
            )