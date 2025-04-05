from __future__ import annotations

from typing import Dict, Type

import pytest

from ember.core.exceptions import (
    InvalidArgumentError,
    InvalidPromptError,
    SpecificationValidationError,
)
from ember.core.registry.specification.specification import Specification
from ember.core.types import EmberModel


class DummyInput(EmberModel):
    """Dummy input model for testing prompt specifications.

    Attributes:
        name (str): The name used for prompt generation.
    """

    name: str


class DummyOutput(EmberModel):
    """Dummy output model for testing prompt specification functionality.

    Attributes:
        result (str): The result produced by the prompt.
    """

    result: str


class DummySpecification(Specification[DummyInput, DummyOutput]):
    """Dummy specification for testing prompt rendering, input validation, and output validation.

    Attributes:
        prompt_template (str): Template string for greeting using a name.
        input_model (Type[DummyInput]): Model used for input validation.
        structured_output (Type[DummyOutput]): Model used for output validation.
        check_all_placeholders (bool): Flag to enforce that all required placeholders are present.
    """

    prompt_template: str = "Hello, {name}!"
    input_model: Type[DummyInput] = DummyInput
    structured_output: Type[DummyOutput] = DummyOutput
    check_all_placeholders: bool = True


def test_render_prompt_valid() -> None:
    """Test that render_prompt produces the correct output for valid input."""
    dummy_specification: DummySpecification = DummySpecification()
    rendered_prompt: str = dummy_specification.render_prompt(inputs={"name": "Test"})
    assert rendered_prompt == "Hello, Test!"


def test_render_prompt_missing_placeholder() -> None:
    """Test that instantiation fails when a required placeholder is missing in the prompt template."""

    class NoPlaceholderSpecification(Specification[DummyInput, DummyOutput]):
        """Specification missing a required placeholder in its prompt template."""

        prompt_template: str = "Hello!"  # Missing the '{name}' placeholder.
        input_model: Type[DummyInput] = DummyInput
        check_all_placeholders: bool = True

    with pytest.raises(InvalidPromptError) as exc_info:
        _ = NoPlaceholderSpecification()  # Validation is triggered upon instantiation.
    assert "name" in str(exc_info.value)


def test_validate_inputs_with_dict() -> None:
    """Test that validate_inputs correctly parses a dictionary input."""
    dummy_specification: DummySpecification = DummySpecification()
    input_data: Dict[str, str] = {"name": "Alice"}
    validated_input = dummy_specification.validate_inputs(inputs=input_data)
    assert isinstance(validated_input, DummyInput)
    assert validated_input.name == "Alice"


def test_validate_inputs_with_model() -> None:
    """Test that validate_inputs accepts an already valid Pydantic model."""
    dummy_specification: DummySpecification = DummySpecification()
    input_instance: DummyInput = DummyInput(name="Bob")
    validated_input = dummy_specification.validate_inputs(inputs=input_instance)
    assert isinstance(validated_input, DummyInput)
    assert validated_input.name == "Bob"


def test_validate_inputs_invalid_type() -> None:
    """Test that validate_inputs raises an error when given an invalid input type."""
    dummy_specification: DummySpecification = DummySpecification()
    with pytest.raises(InvalidArgumentError):
        dummy_specification.validate_inputs(inputs="invalid input type")


def test_validate_output() -> None:
    """Test that validate_output correctly parses dictionary output data."""
    dummy_specification: DummySpecification = DummySpecification()
    output_data: Dict[str, str] = {"result": "Success"}
    validated_output = dummy_specification.validate_output(output=output_data)
    assert isinstance(validated_output, DummyOutput)
    assert validated_output.result == "Success"


def test_misconfigured_specification_missing_input_model() -> None:
    """Test that rendering a prompt raises an error when the input_model is missing."""

    class MisconfiguredSpecification(Specification):
        """Specification configured without an input_model."""

        prompt_template: str = "Hi, {name}!"
        check_all_placeholders: bool = True

    misconfigured_specification = MisconfiguredSpecification()
    with pytest.raises(InvalidPromptError):
        misconfigured_specification.render_prompt(inputs={"name": "Test"})


def test_misconfigured_specification_incompatible_model() -> None:
    """Test that validate_inputs raises an error when the provided input model type is incompatible."""

    class AnotherInput(EmberModel):
        """Alternate input model for testing specification compatibility."""

        other: str

    class IncompatibleSpecification(Specification[AnotherInput, DummyOutput]):
        """Specification expecting a different input model type."""

        prompt_template: str = "Hi, {other}!"
        input_model: Type[AnotherInput] = AnotherInput
        check_all_placeholders: bool = True

    incompatible_specification = IncompatibleSpecification()
    wrong_input_instance: DummyInput = DummyInput(name="Test")
    with pytest.raises(SpecificationValidationError) as exc_info:
        incompatible_specification.validate_inputs(inputs=wrong_input_instance)
    assert "model mismatch" in str(exc_info.value)


def test_render_prompt_with_no_template_but_input_model() -> None:
    """Test that render_prompt falls back to the input_model value when no prompt_template is provided."""

    class NoTemplateSpecification(Specification[DummyInput, DummyOutput]):
        """Specification without a prompt_template but with an input model."""

        input_model: Type[DummyInput] = DummyInput
        check_all_placeholders: bool = False

    no_template_specification = NoTemplateSpecification()
    rendered_prompt: str = no_template_specification.render_prompt(
        inputs={"name": "Test"}
    )
    assert "Test" in rendered_prompt


def test_render_prompt_no_template_no_input_model() -> None:
    """Test that render_prompt raises an error when neither prompt_template nor input_model is provided."""

    class EmptySpecification(Specification):
        """Empty specification with no prompt_template or input_model."""

        check_all_placeholders: bool = False

    empty_specification = EmptySpecification()
    with pytest.raises(InvalidPromptError):
        empty_specification.render_prompt(inputs={})
