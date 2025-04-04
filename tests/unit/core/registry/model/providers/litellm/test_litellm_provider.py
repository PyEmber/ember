#!/usr/bin/env python3
"""
Unit tests for the Litellm provider implementation.
"""

from typing import Any

import pytest

from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.providers.litellm.litellm_provider import (
    LitellmChatParameters,
    LitellmModel,
)

class DummyMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class DummyChoice:
    def __init__(self, message_content: str) -> None:
        self.message = DummyMessage(message_content)

class DummyLiteLLMResponse:
    def __init__(self) -> None:
        self.choices = [DummyChoice("Test response.")]
        self.usage = type(
            "Usage",
            (),
            {"total_tokens": 100, "prompt_tokens": 40, "completion_tokens": 60},
        )

def create_dummy_model_info() -> ModelInfo:
    return ModelInfo(
        id="gpt-4o-mini",
        name="gpt-4o-mini",
        cost=ModelCost(input_cost_per_thousand=5000, output_cost_per_thousand=15000),
        rate_limit=RateLimit(tokens_per_minute=10000000, requests_per_minute=1500),
        provider=ProviderInfo(name="LiteLLM", default_api_key="dummy_litellm_key"),
        api_key="dummy_litellm_key",
    )

class DummyLiteLLMClient:
    class DummyCompletions:
        @staticmethod
        def create(**kwargs: Any) -> Any:
            return DummyLiteLLMResponse()

    chat = type("DummyChat", (), {"completions": DummyCompletions()})()

@pytest.fixture(autouse=True)
def patch_litemllm(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    monkeypatch.setattr(litellm, "chat", DummyLiteLLMClient().chat)


def test_litellm_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LitellmModel.forward returns a valid ChatResponse."""
    # Fix the type checking issue by directly examining the response content
    import inspect

    # Get the module where ChatResponse is defined
    response_module = inspect.getmodule(ChatResponse)

    dummy_info = create_dummy_model_info()
    model = LitellmModel(dummy_info)
    # Patch client.chat.completions.create to return our dummy response.
    monkeypatch.setattr(
        model.client.chat.completions, "create", lambda **kwargs: DummyLiteLLMResponse()
    )
    request = ChatRequest(prompt="Hello LiteLLM", temperature=0.7, max_tokens=100)
    response = model.forward(request)

    # Verify it's a ChatResponse by checking structure and behavior,
    # not by using isinstance which can be affected by module loading
    assert response.__class__.__name__ == "ChatResponse"
    assert hasattr(response, "data")
    assert hasattr(response, "raw_output")
    assert hasattr(response, "usage")

    # Verify the actual content/behavior
    assert "Test response." in response.data
    usage = response.usage
    assert usage.total_tokens == 100


def test_litellm_parameters() -> None:
    """Test that LitellmChatParameters converts prompt to messages properly."""
    params = LitellmChatParameters(
        prompt="Hello", context="System message", temperature=0.5, max_tokens=None
    )
    kwargs = params.to_litemllm_kwargs()
    assert "messages" in kwargs
    messages = kwargs["messages"]
    assert any(msg["role"] == "system" for msg in messages)
    assert any(msg["role"] == "user" for msg in messages)
