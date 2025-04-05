"""
Unit tests for structured logging utilities.
"""

import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from ember.xcs.utils.structured_logging import (
    LoggingConfig,
    clear_context,
    configure_logging,
    enrich_exception,
    get_context_value,
    get_structured_logger,
    log_context,
    set_context_value,
    time_operation,
    with_context,
)


@pytest.fixture
def reset_logging_config():
    """Reset logging config to default state after each test."""
    original_config = {
        "enabled": LoggingConfig.enabled,
        "sampling_rate": LoggingConfig.sampling_rate,
        "default_threshold_ms": LoggingConfig.default_threshold_ms,
        "include_context_for_level": LoggingConfig.include_context_for_level,
        "trace_all_operations": LoggingConfig.trace_all_operations,
    }

    yield

    # Restore original config
    LoggingConfig.enabled = original_config["enabled"]
    LoggingConfig.sampling_rate = original_config["sampling_rate"]
    LoggingConfig.default_threshold_ms = original_config["default_threshold_ms"]
    LoggingConfig.include_context_for_level = original_config[
        "include_context_for_level"
    ]
    LoggingConfig.trace_all_operations = original_config["trace_all_operations"]

    # Clear thread context
    clear_context()


def test_context_isolation(reset_logging_config):
    """Test that thread context is properly isolated."""
    # Set up initial context in main thread
    set_context_value("thread", "main")

    # Storage for thread values
    results = {}
    event = threading.Event()

    # Thread function that manipulates context
    def thread_fn():
        try:
            # Should start with empty context
            results["initial"] = get_context_value("thread")

            # Set a thread-specific value
            set_context_value("thread", "worker")

            # Signal we're done
            event.set()

            # Wait a bit to ensure main thread's operations occur during thread's lifetime
            time.sleep(0.1)

            # Verify our value is still what we set
            results["final"] = get_context_value("thread")
        except Exception as e:
            results["error"] = str(e)

    # Start the thread
    thread = threading.Thread(target=thread_fn)
    thread.start()

    # Wait for the thread to set its context
    event.wait()

    # Check that main thread's context is unchanged
    assert get_context_value("thread") == "main"

    # Wait for thread to complete
    thread.join()

    # Verify thread had proper isolation
    assert "error" not in results
    assert results["initial"] is None
    assert results["final"] == "worker"
    assert get_context_value("thread") == "main"


def test_context_manager(reset_logging_config):
    """Test the log_context context manager."""
    # Set initial context
    set_context_value("key1", "value1")

    # Use context manager to add a key
    with log_context(key2="value2", key3="value3"):
        assert get_context_value("key1") == "value1"
        assert get_context_value("key2") == "value2"
        assert get_context_value("key3") == "value3"

        # Modify an existing key within the context
        set_context_value("key1", "modified")
        assert get_context_value("key1") == "modified"

        # Nested context
        with log_context(key3="nested", key4="value4"):
            assert get_context_value("key1") == "modified"
            assert get_context_value("key2") == "value2"
            assert get_context_value("key3") == "nested"
            assert get_context_value("key4") == "value4"

        # After nested context
        assert get_context_value("key1") == "modified"
        assert get_context_value("key2") == "value2"
        assert get_context_value("key3") == "value3"
        assert get_context_value("key4") is None

    # After outer context
    assert get_context_value("key1") == "value1"  # Restored to original
    assert get_context_value("key2") is None
    assert get_context_value("key3") is None


def test_with_context_logging(reset_logging_config):
    """Test the with_context logging function."""
    # Create a mock logger
    mock_logger = MagicMock()
    mock_logger.isEnabledFor.return_value = True

    # Patch should_log to always return True to force execution
    with patch("ember.xcs.utils.structured_logging.should_log", return_value=True):
        # Set context and log
        with log_context(user_id="123", operation="test"):
            with_context(mock_logger, logging.INFO, "Test message")

        # Verify the logging call
        mock_logger.log.assert_called_once()
        args, kwargs = mock_logger.log.call_args

        # Check log level and message
        assert args[0] == logging.INFO
        assert args[1] == "Test message"

        # Check context data
        assert "extra" in kwargs
        assert "structured_data" in kwargs["extra"]
        assert "context" in kwargs["extra"]["structured_data"]
        context = kwargs["extra"]["structured_data"]["context"]
        assert context["user_id"] == "123"
        assert context["operation"] == "test"


def test_time_operation_decorator(reset_logging_config):
    """Test the time_operation decorator."""
    # Configure for testing
    LoggingConfig.enabled = True
    LoggingConfig.sampling_rate = 1.0
    LoggingConfig.default_threshold_ms = 0.0

    # Create a mock logger
    mock_logger = MagicMock()
    mock_logger.isEnabledFor.return_value = True

    # Patch should_log and perf_counter to control behavior
    # Need more perf_counter values to account for the additional time.perf_counter() call in with_context
    with patch(
        "ember.xcs.utils.structured_logging.should_log", return_value=True
    ), patch(
        "time.perf_counter", side_effect=[0.0, 0.1, 0.2]
    ):  # Start time, timestamp in log, end time
        # Define decorated function with no threshold
        @time_operation("test_operation", logger=mock_logger, threshold_ms=0.0)
        def test_function():
            return "result"

        # Call the function
        result = test_function()

        # Verify function result
        assert result == "result"

        # Verify logging occurred
        mock_logger.log.assert_called_once()

    # Reset the logger
    mock_logger.reset_mock()

    # Create a new decorated function with explicit threshold
    # Note: We need to create a new function because the threshold is captured at decoration time
    with patch(
        "ember.xcs.utils.structured_logging.should_log", return_value=True
    ), patch(
        "time.perf_counter", side_effect=[0.0, 0.001, 0.002]
    ):  # Start time, timestamp in log, end time

        @time_operation("test_operation", logger=mock_logger, threshold_ms=100.0)
        def test_function_with_threshold():
            return "result"

        # Call the function with threshold
        test_function_with_threshold()

        # Should not log due to threshold
        assert not mock_logger.log.called


def test_performance_options(reset_logging_config):
    """Test performance optimization options."""
    # Save original values
    original_enabled = LoggingConfig.enabled
    original_sampling_rate = LoggingConfig.sampling_rate
    original_threshold = LoggingConfig.default_threshold_ms
    original_trace_all = LoggingConfig.trace_all_operations

    try:
        # Configure for high performance mode
        LoggingConfig.configure(high_performance_mode=True)

        # Check that appropriate settings were applied
        assert (
            LoggingConfig.enabled is False
        ), "Enabled should be False in high performance mode"
        assert (
            LoggingConfig.sampling_rate < 1.0
        ), "Sampling rate should be reduced in high performance mode"
        assert (
            LoggingConfig.default_threshold_ms > 0.0
        ), "Threshold should be increased in high performance mode"
        assert (
            LoggingConfig.trace_all_operations is False
        ), "Trace all should be disabled in high performance mode"

        # Reset to known state before next test
        LoggingConfig.enabled = original_enabled
        LoggingConfig.sampling_rate = original_sampling_rate
        LoggingConfig.default_threshold_ms = original_threshold
        LoggingConfig.trace_all_operations = original_trace_all

        # Test development mode settings
        LoggingConfig.configure(development_mode=True)

        # Check that appropriate settings were applied
        assert (
            LoggingConfig.enabled is True
        ), "Enabled should be True in development mode"
        assert (
            LoggingConfig.sampling_rate == 1.0
        ), "Sampling rate should be 1.0 in development mode"
        assert (
            LoggingConfig.default_threshold_ms == 0.0
        ), "Threshold should be 0.0 in development mode"
        assert (
            LoggingConfig.trace_all_operations is True
        ), "Trace all should be enabled in development mode"
    finally:
        # Restore the original values at the end to avoid affecting other tests
        LoggingConfig.enabled = original_enabled
        LoggingConfig.sampling_rate = original_sampling_rate
        LoggingConfig.default_threshold_ms = original_threshold
        LoggingConfig.trace_all_operations = original_trace_all


def test_enrich_exception(reset_logging_config):
    """Test exception enrichment."""
    # Import directly for consistent mock
    from ember.xcs.exceptions import XCSError

    # Mock for XCSError with add_context method
    error = XCSError("Test error")

    # Create a context
    with log_context(request_id="123", user="test_user"):
        # Enrich the exception
        enriched = enrich_exception(error, operation="test_operation")

        # Should be the same exception object
        assert enriched is error

        # Check context data in diagnostic_context
        context_data = error.diagnostic_context
        assert "request_id" in context_data
        assert "user" in context_data
        assert "operation" in context_data

        # Values should match
        assert context_data["request_id"] == "123"
        assert context_data["user"] == "test_user"
        assert context_data["operation"] == "test_operation"

    # Test with standard exception that doesn't have add_context
    std_error = ValueError("Standard error")
    enriched = enrich_exception(std_error, test_attr="test_value")

    # Should be same object
    assert enriched is std_error

    # Should have attribute added
    assert hasattr(enriched, "test_attr")
    assert enriched.test_attr == "test_value"


def test_should_log_sampling(reset_logging_config):
    """Test logging sampling behavior."""
    from ember.xcs.utils.structured_logging import should_log

    # Force enabled for testing
    LoggingConfig.enabled = True

    # Test with 100% sampling
    LoggingConfig.sampling_rate = 1.0
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = True
        mock_get_logger.return_value = mock_logger

        # Should always log
        assert should_log(logging.INFO) is True

    # Test with 0% sampling
    LoggingConfig.sampling_rate = 0.0
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = True
        mock_get_logger.return_value = mock_logger

        # Should never log
        assert should_log(logging.INFO) is False
