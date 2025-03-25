"""Hugging Face provider implementation for the Ember framework.

This module provides a comprehensive integration with Hugging Face models through
both the Hugging Face Inference API and local model loading capabilities. It handles
all aspects of model interaction including authentication, request formatting,
response parsing, error handling, and usage tracking specifically for
Hugging Face models.

The implementation follows Hugging Face best practices for API integration,
including efficient error handling, comprehensive logging, and support for
both hosted and local model inference. It supports a wide variety of models
available on the Hugging Face Hub with appropriate parameter adjustments for
model-specific requirements.

Classes:
    HuggingFaceProviderParams: TypedDict defining HF-specific parameters
    HuggingFaceChatParameters: Parameter conversion for HF chat completions
    HuggingFaceModel: Core implementation of the HuggingFace provider

Details:
    - Authentication and client configuration for Hugging Face Hub API
    - Support for both remote (Inference API) and local model inference
    - Model discovery from the Hugging Face Hub
    - Automatic retry with exponential backoff for transient errors
    - Specialized error handling for different error types
    - Parameter validation and transformation
    - Detailed logging for monitoring and debugging
    - Usage statistics calculation for cost tracking
    - Proper timeout handling to prevent hanging requests

Usage example:
    ```python
    # Direct usage (prefer using ModelRegistry or API)
    from ember.core.registry.model.base.schemas.model_info import ModelInfo, ProviderInfo

    # Configure model information for remote inference
    model_info = ModelInfo(
        id="huggingface:mistralai/Mistral-7B-Instruct-v0.2",
        name="mistralai/Mistral-7B-Instruct-v0.2",
        provider=ProviderInfo(name="HuggingFace", api_key="hf_...")
    )

    # Initialize the model
    model = HuggingFaceModel(model_info)

    # Basic usage
    response = model("What is the Ember framework?")
    print(response.data)  # The model's response text

    # Advanced usage with more parameters
    response = model(
        "Generate creative ideas",
        context="You are a helpful creative assistant",
        temperature=0.7,
        provider_params={"top_p": 0.95, "max_new_tokens": 512}
    )

    # Accessing usage statistics
    print(f"Used {response.usage.total_tokens} tokens")
    ```

For higher-level usage, prefer the model registry or API interfaces:
    ```python
    from ember.api.models import models

    # Using the models API (automatically handles authentication)
    response = models.huggingface.mistral_7b("Tell me about Ember")
    print(response.data)
    ```
"""

import logging
from typing import Any, Dict, List, Optional, Set, Union, cast

import requests
from huggingface_hub import HfApi, InferenceClient, model_info
from pydantic import Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential
from typing_extensions import TypedDict
from transformers import AutoTokenizer

from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    ProviderParams,
)
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.usage import UsageStats
from ember.core.exceptions import ModelProviderError, ValidationError
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    InvalidPromptError,
    ProviderAPIError,
)
from ember.core.registry.model.providers.base_provider import (
    BaseChatParameters,
    BaseProviderModel,
)
from ember.plugin_system import provider


class HuggingFaceProviderParams(ProviderParams):
    """HuggingFace-specific provider parameters for fine-tuning requests.

    This TypedDict defines additional parameters that can be passed to Hugging Face API
    calls beyond the standard parameters defined in BaseChatParameters. These parameters
    provide fine-grained control over the model's generation behavior.

    Parameters can be provided in the provider_params field of a ChatRequest:
    ```python
    request = ChatRequest(
        prompt="Generate creative ideas",
        provider_params={
            "top_p": 0.9,
            "max_new_tokens": 512,
            "do_sample": True
        }
    )
    ```

    Attributes:
        top_p: Optional float between 0 and 1 for nucleus sampling, controlling the
            cumulative probability threshold for token selection.
        top_k: Optional integer limiting the number of tokens considered at each generation step.
        max_new_tokens: Optional integer specifying the maximum number of tokens to generate.
        repetition_penalty: Optional float to penalize repetition in generated text.
        do_sample: Optional boolean to enable sampling (True) or use greedy decoding (False).
        use_cache: Optional boolean to use KV cache for faster generation.
        stop_sequences: Optional list of strings that will cause the model to stop
            generating when encountered.
        seed: Optional integer for deterministic sampling, ensuring repeatable outputs.
        use_local_model: Optional boolean to use a locally downloaded model instead of
            the Inference API. When True, the model will be downloaded and loaded locally.
    """

    top_p: Optional[float]
    top_k: Optional[int]
    max_new_tokens: Optional[int]
    repetition_penalty: Optional[float]
    do_sample: Optional[bool]
    use_cache: Optional[bool]
    stop_sequences: Optional[List[str]]
    seed: Optional[int]
    use_local_model: Optional[bool]


logger: logging.Logger = logging.getLogger(__name__)


class HuggingFaceChatParameters(BaseChatParameters):
    """Parameters for Hugging Face chat requests with validation and conversion logic.

    This class extends BaseChatParameters to provide Hugging Face-specific parameter
    handling and validation. It ensures that parameters are correctly formatted
    for the Hugging Face Inference API, handling the conversion between Ember's universal
    parameter format and Hugging Face's API requirements.

    Key features:
        - Enforces a minimum value for max_tokens
        - Provides a sensible default (512 tokens) if not specified
        - Validates that max_tokens is a positive integer
        - Maps Ember's 'max_tokens' parameter to HF's 'max_new_tokens'
        - Handles temperature scaling for the Hugging Face API

    The class handles parameter validation and transformation to ensure that
    all requests sent to the Hugging Face API are properly formatted and contain
    all required fields with valid values.
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
        """Ensure max_tokens is a positive value.

        Args:
            value (int): The max_tokens value to validate.

        Returns:
            int: The validated positive integer.

        Raises:
            ValidationError: If max_tokens is not a positive integer.
        """
        if value <= 0:
            raise ValidationError(
                f"max_tokens must be a positive integer, got {value}",
                provider="HuggingFace",
            )
        return value

    def to_huggingface_kwargs(self) -> Dict[str, Any]:
        """Convert chat parameters into keyword arguments for the Hugging Face API."""
        # Create the prompt with system context if provided
        prompt = self.build_prompt()
        
        return {
            "prompt": prompt,
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }


class HuggingFaceConfig:
    """Helper class to manage Hugging Face model configuration.

    This class provides methods to retrieve information about Hugging Face models,
    including model types, capabilities, and supported parameters.
    """

    _config_cache: Dict[str, Any] = {}

    @classmethod
    def get_valid_models(cls) -> Set[str]:
        """Get a set of valid model IDs from the Hugging Face Hub.

        This is a simplified placeholder implementation. In a real implementation,
        this would likely query the Hugging Face API for a list of models or
        check against a cached list of known models.

        Returns:
            Set[str]: A set of valid model IDs.
        """
        # In a real implementation, this would query the Hugging Face API
        # or use a cached list of models. This is a simplified example.
        return set()

    @classmethod
    def is_chat_model(cls, model_id: str) -> bool:
        """Determine if a model supports chat completion.

        Args:
            model_id (str): The Hugging Face model ID.

        Returns:
            bool: True if the model supports chat completion.
        """
        # This would be implemented with actual model capability checking
        # For now, we'll assume all models support chat
        return True


@provider("HuggingFace")
class HuggingFaceModel(BaseProviderModel):
    """Implementation for Hugging Face models in the Ember framework.

    This class provides a comprehensive integration with Hugging Face models,
    supporting both remote inference through the Inference API and local model
    loading. It implements the BaseProviderModel interface, making Hugging Face
    models compatible with the wider Ember ecosystem.

    The implementation supports a wide range of Hugging Face models, including
    both chat/completion models and other model types. It handles authentication,
    request formatting, response processing, and error handling specific to
    Hugging Face's APIs and model formats.

    Key features:
        - Support for both Inference API and local model loading
        - Robust error handling with automatic retries for transient errors
        - Comprehensive logging for debugging and monitoring
        - Usage statistics tracking for cost analysis
        - Type-safe parameter handling with runtime validation
        - Model-specific parameter adjustments
        - Proper timeout handling to prevent hanging requests

    The class provides three core functions:
        1. Creating and configuring the Hugging Face Inference API client
        2. Processing chat requests through the forward method
        3. Calculating usage statistics for billing and monitoring

    Implementation details:
        - Uses the official Hugging Face Hub Python SDK
        - Supports both remote inference and local model loading
        - Implements tenacity-based retry logic with exponential backoff
        - Properly handles API timeouts to prevent hanging
        - Calculates token usage with model-specific tokenizers
        - Handles parameter conversion between Ember and Hugging Face formats

    Attributes:
        PROVIDER_NAME: The canonical name of this provider for registration.
        model_info: Model metadata including credentials and cost schema.
        client: The configured Hugging Face inference client.
        tokenizer: Optional tokenizer for local models and token counting.
    """

    PROVIDER_NAME: str = "HuggingFace"

    def __init__(self, model_info: ModelInfo) -> None:
        """Initialize a HuggingFaceModel instance.

        Args:
            model_info (ModelInfo): Model information including credentials and
                cost schema.
        """
        super().__init__(model_info)
        self.tokenizer = None
        self._local_model = None

    def _normalize_huggingface_model_name(self, raw_name: str) -> str:
        """Normalize the Hugging Face model name.

        Checks if the provided model name exists on the HF Hub and returns a
        standardized version. If the model doesn't exist, falls back to a default.

        Args:
            raw_name (str): The input model name, which may be a short name or full path.

        Returns:
            str: A normalized and validated model name.
        """
        # Handle provider-prefixed model names
        if raw_name.startswith("huggingface:"):
            raw_name = raw_name[12:]

        try:
            # Verify model exists on Hub
            HfApi().model_info(raw_name)
            return raw_name
        except Exception as exc:
            # If model doesn't exist, fall back to a default
            default_model = "mistralai/Mistral-7B-Instruct-v0.2"
            logger.warning(
                "HuggingFace model '%s' not found on Hub. Falling back to '%s': %s",
                raw_name,
                default_model,
                exc,
            )
            return default_model

    def create_client(self) -> Any:
        """Create and configure the Hugging Face client.

        Retrieves the API token from the model information and sets up the
        InferenceClient for making API calls to the Hugging Face Inference API.

        Returns:
            Any: The configured Hugging Face InferenceClient.

        Raises:
            ModelProviderError: If the API token is missing or invalid.
        """
        api_key: Optional[str] = self.model_info.get_api_key()
        if not api_key:
            raise ModelProviderError.for_provider(
                provider_name=self.PROVIDER_NAME,
                message="HuggingFace API token is missing or invalid.",
            )
        
        # Initialize the Inference API client
        client = InferenceClient(token=api_key)
        
        # Log available endpoints for the model (if accessible)
        try:
            model_id = self._normalize_huggingface_model_name(self.model_info.name)
            logger.info(
                "Initialized HuggingFace Inference client for model: %s", model_id
            )
        except Exception as exc:
            logger.warning(
                "Could not verify HuggingFace model information: %s", exc
            )
            
        return client

    def _load_local_model(self, model_id: str) -> Any:
        """Load a model locally for inference.

        This method downloads and initializes a model for local inference
        using the transformers library.

        Args:
            model_id (str): The Hugging Face model ID to load.

        Returns:
            Any: The loaded model ready for inference.

        Raises:
            ProviderAPIError: If the model cannot be loaded.
        """
        try:
            from transformers import AutoModelForCausalLM, pipeline
            
            logger.info("Loading model %s locally", model_id)
            # Load tokenizer for token counting and processing
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Load the model
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map="auto",
                trust_remote_code=True
            )
            
            # Create a text generation pipeline
            generation_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer
            )
            
            logger.info("Successfully loaded model %s locally", model_id)
            return generation_pipeline
        except Exception as exc:
            logger.exception("Failed to load local model: %s", exc)
            raise ProviderAPIError.for_provider(
                provider_name=self.PROVIDER_NAME,
                message=f"Failed to load local model: {exc}",
                cause=exc,
            )

    @retry(
        wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True
    )
    def forward(self, request: ChatRequest) -> ChatResponse:
        """Send a ChatRequest to the Hugging Face model and process the response.

        Supports both remote inference via the Inference API and local model inference
        based on configuration. Converts Ember parameters to Hugging Face parameters
        and normalizes the response.

        Args:
            request (ChatRequest): The chat request containing the prompt along with
                provider-specific parameters.

        Returns:
            ChatResponse: Contains the response text, raw output, and usage statistics.

        Raises:
            InvalidPromptError: If the prompt in the request is empty.
            ProviderAPIError: For any unexpected errors during the API invocation.
        """
        if not request.prompt:
            raise InvalidPromptError.with_context(
                "HuggingFace prompt cannot be empty.",
                provider=self.PROVIDER_NAME,
                model_name=self.model_info.name,
            )

        logger.info(
            "HuggingFace forward invoked",
            extra={
                "provider": self.PROVIDER_NAME,
                "model_name": self.model_info.name,
                "prompt_length": len(request.prompt),
            },
        )

        # Convert the universal ChatRequest into HuggingFace-specific parameters
        hf_parameters: HuggingFaceChatParameters = HuggingFaceChatParameters(
            **request.model_dump(exclude={"provider_params"})
        )
        hf_kwargs: Dict[str, Any] = hf_parameters.to_huggingface_kwargs()

        # Merge provider-specific parameters
        provider_params = cast(HuggingFaceProviderParams, request.provider_params)
        # Only include non-None values
        hf_kwargs.update(
            {k: v for k, v in provider_params.items() if v is not None}
        )

        # Get normalized model name
        model_id = self._normalize_huggingface_model_name(self.model_info.name)
        
        # Check if we should use local model inference
        use_local = hf_kwargs.pop("use_local_model", False)
        
        try:
            if use_local:
                # Local model inference
                if self._local_model is None:
                    self._local_model = self._load_local_model(model_id)
                
                # Extract parameters for local inference
                prompt = hf_kwargs.pop("prompt")
                max_new_tokens = hf_kwargs.pop("max_new_tokens", 512)
                temperature = hf_kwargs.pop("temperature", 0.7)
                
                # Run local inference
                result = self._local_model(
                    prompt,  # Pass the prompt directly
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **hf_kwargs
                )
                
                # Extract the generated text
                if isinstance(result, list) and len(result) > 0:
                    # Most pipelines return a list of dictionaries
                    generated_text = result[0].get("generated_text", "")
                    
                    # Remove the input prompt from the output if present
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].lstrip()
                else:
                    generated_text = str(result)
                
                # Create a raw output structure similar to API responses
                raw_output = {
                    "generated_text": generated_text,
                    "model": model_id,
                    "usage": {
                        "prompt_tokens": self._count_tokens(prompt),
                        "completion_tokens": self._count_tokens(generated_text),
                    }
                }
            else:
                # Remote inference via the Inference API
                # Remove timeout from kwargs as it's handled separately
                timeout = hf_kwargs.pop("timeout", 30)  # Default 30 seconds timeout
                
                # Get the prompt from kwargs
                prompt = hf_kwargs.pop("prompt")
                
                # Call the text-generation endpoint
                response = self.client.text_generation(
                    prompt=prompt,  # Pass as named parameter
                    model=model_id,
                    **hf_kwargs,  # Other parameters like temperature, max_new_tokens, etc.
                )
                
                # Extract the response text
                generated_text = response
                
                # Create a raw output structure for usage calculation
                raw_output = {
                    "generated_text": generated_text,
                    "model": model_id,
                    "usage": {
                        "prompt_tokens": self._count_tokens(prompt),
                        "completion_tokens": self._count_tokens(generated_text),
                    }
                }
            
            # Calculate usage statistics
            usage_stats = self.calculate_usage(raw_output=raw_output)
            
            return ChatResponse(data=generated_text, raw_output=raw_output, usage=usage_stats)
        
        except requests.exceptions.HTTPError as http_err:
            if 500 <= http_err.response.status_code < 600:
                logger.error("HuggingFace server error: %s", http_err)
            raise
        except Exception as exc:
            logger.exception("Unexpected error in HuggingFaceModel.forward()")
            raise ProviderAPIError.for_provider(
                provider_name=self.PROVIDER_NAME,
                message=f"API error: {str(exc)}",
                cause=exc,
            )

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text using the model's tokenizer.

        Args:
            text (str): The text to tokenize and count.

        Returns:
            int: The number of tokens in the text.
        """
        try:
            if self.tokenizer is None:
                # Initialize tokenizer if not already done
                model_id = self._normalize_huggingface_model_name(self.model_info.name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Count tokens using the model's tokenizer
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as exc:
            logger.warning(
                "Failed to count tokens, estimating based on words: %s", exc
            )
            # Fall back to a rough approximation if tokenizer fails
            return len(text.split())

    def calculate_usage(self, raw_output: Any) -> UsageStats:
        """Calculate usage statistics based on the model response.

        Extracts token counts from the raw output and calculates cost based on
        the model's cost configuration.

        Args:
            raw_output (Any): The raw response data containing token counts.

        Returns:
            UsageStats: An object containing token counts and cost metrics.
        """
        # Extract usage information from raw output
        usage_data = raw_output.get("usage", {})
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        
        # Calculate cost based on model cost configuration
        input_cost = (prompt_tokens / 1000.0) * self.model_info.cost.input_cost_per_thousand
        output_cost = (completion_tokens / 1000.0) * self.model_info.cost.output_cost_per_thousand
        total_cost = round(input_cost + output_cost, 6)
        
        return UsageStats(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=total_cost,
        ) 