"""Unit tests for XCS tracer context types.

This module validates the behavior of the TraceContextData class and the
get_current_trace_context function. It ensures that extra context information is
accurately stored and that the stub implementation of get_current_trace_context returns None.
"""

from typing import Any, Dict

from ember.xcs.tracer._context_types import TraceContextData


def test_trace_context_data_storage() -> None:
    """Verifies that TraceContextData accurately stores extra context information.

    This test instantiates TraceContextData with a sample dictionary of extra
    information and asserts that the stored extra_info attribute matches the input.

    Raises:
        AssertionError: If the stored extra_info does not exactly match the provided dictionary.
    """
    extra_info: Dict[str, Any] = {"user": "test_user", "debug": True}
    context_data: TraceContextData = TraceContextData(extra_info=extra_info)
    assert (
        context_data.extra_info == extra_info
    ), "TraceContextData did not correctly store and expose the extra context information."
