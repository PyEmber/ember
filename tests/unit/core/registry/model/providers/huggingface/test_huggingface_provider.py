"""Unit tests for the HuggingFace provider implementation.

Tests model creation, parameter handling, and request/response processing.
"""

import pytest
from unittest.mock import MagicMock, patch

from ember.core.registry.model.base.schemas.chat_schemas import ChatRequest
from ember.core.registry.model.base.schemas.model_info import ModelInfo, ProviderInfo
from ember.core.registry.model.providers.huggingface.huggingface_provider import (
    HuggingFaceModel,
    HuggingFaceChatParameters,
)


class DummyHfResponse:
    """Dummy response object mimicking Hugging Face API responses."""
    
    def __init__(self):
        """Initialize with test data."""
        self.text = "Test response from Hugging Face."


def create_dummy_model_info():
    """Create dummy model info for testing."""
    return ModelInfo(
        id="huggingface:gpt2",
        name="gpt2",
        provider=ProviderInfo(name="HuggingFace", default_api_key="dummy_key"),
    )


@pytest.fixture
def hf_model():
    """Return a HuggingFace model instance with mocked client."""
    with patch("huggingface_hub.InferenceClient") as mock_client_cls:
        # Configure the mock client
        mock_client = MagicMock()
        # Create a new mock for text_generation that returns a string
        mock_client.text_generation = MagicMock(return_value="Test response from Hugging Face.")
        mock_client_cls.return_value = mock_client
        
        # Mock model_info to avoid API calls
        with patch("huggingface_hub.model_info") as mock_model_info:
            # Create model with the mocked client
            model = HuggingFaceModel(create_dummy_model_info())
            
            # Replace the client to ensure our mock is used
            model.client = mock_client
            
            yield model


def test_normalize_model_name(hf_model):
    """Test that model names are properly normalized."""
    # Test with normal model ID
    normalized = hf_model._normalize_huggingface_model_name("gpt2")
    assert normalized == "gpt2"
    
    # Test with namespaced model ID
    normalized = hf_model._normalize_huggingface_model_name("huggingface:gpt2")
    assert normalized == "gpt2"
    
    # Test with org/model format
    normalized = hf_model._normalize_huggingface_model_name("mistralai/Mistral-7B-Instruct-v0.2")
    assert normalized == "mistralai/Mistral-7B-Instruct-v0.2"


def test_chat_parameters_conversion():
    """Test conversion of parameters to HuggingFace format."""
    params = HuggingFaceChatParameters(
        prompt="Hello Hugging Face",
        context="You are a helpful assistant.",
        temperature=0.5,
        max_tokens=100,
    )
    
    hf_kwargs = params.to_huggingface_kwargs()
    
    assert hf_kwargs["prompt"] == "You are a helpful assistant.\n\nHello Hugging Face"
    assert hf_kwargs["temperature"] == 0.5
    assert hf_kwargs["max_new_tokens"] == 100
    assert "top_p" not in hf_kwargs  # Should not include defaults


def test_token_counting(hf_model, monkeypatch):
    """Test token counting functionality."""
    # Mock the tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
    
    monkeypatch.setattr(hf_model, "tokenizer", mock_tokenizer)
    
    # Test token counting
    token_count = hf_model._count_tokens("Test text")
    assert token_count == 5
    mock_tokenizer.encode.assert_called_once_with("Test text")
    
    # Test fallback when tokenizer raises exception
    mock_tokenizer.encode.side_effect = Exception("Tokenizer error")
    token_count = hf_model._count_tokens("welcome to the jungle")
    assert token_count == 4  # Should estimate based on word count


def test_huggingface_forward(hf_model, monkeypatch):
    """Test that forward returns a valid ChatResponse."""
    request = ChatRequest(prompt="Hello Hugging Face", temperature=0.7, max_tokens=100)
    
    # Call the provider
    response = hf_model.forward(request)
    
    # Verify response structure and content
    assert response.__class__.__name__ == "ChatResponse"
    assert hasattr(response, "data")
    assert hasattr(response, "raw_output")
    assert hasattr(response, "usage")
    
    # Verify the actual content
    assert "Test response from Hugging Face." in response.data
    assert response.usage.total_tokens >= 0

    def test_local_model_inference_direct():
        """Test local model inference path directly."""
        # Create a minimal implementation
        model = HuggingFaceModel(create_dummy_model_info())
        
        # Set up the test scenario
        model._local_model = lambda prompt, **kwargs: [{"generated_text": "Local model response"}]
        
        # Create request with use_local_model=True
        request = ChatRequest(
            prompt="Test prompt",
            provider_params={"use_local_model": True}
        )
        
        # Call forward directly
        hf_parameters = HuggingFaceChatParameters(prompt=request.prompt)
        hf_kwargs = hf_parameters.to_huggingface_kwargs()
        
        # Verify the local path works
        if hf_kwargs.get("use_local_model"):
            result = model._local_model("Test prompt")
            assert result[0]["generated_text"] == "Local model response"

def test_huggingface_call_interface(hf_model):
    """Test the callable interface of the model."""
    # Call the model directly with a prompt
    response = hf_model("What is Ember?")
    
    # Verify response
    assert "Test response from Hugging Face." in response.data
    
    # Call with additional parameters
    response = hf_model(
        "What is Ember?",
        temperature=0.8,
        max_tokens=200,
        provider_params={"top_p": 0.95}
    )
    

