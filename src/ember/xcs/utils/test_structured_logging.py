"""
Testing structured logging functionality.

This module verifies that structured logging operates correctly
with expected context propagation and performance optimizations.
"""

import logging
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

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


class TestStructuredLogging(unittest.TestCase):
    """Test cases for structured logging functionality."""

    def setUp(self):
        """Set up test environment."""
        # Save original config
        self.original_enabled = LoggingConfig.enabled
        self.original_sampling_rate = LoggingConfig.sampling_rate
        self.original_threshold = LoggingConfig.default_threshold_ms

        # Configure for testing
        LoggingConfig.enabled = True
        LoggingConfig.sampling_rate = 1.0
        LoggingConfig.default_threshold_ms = 0.0

        # Clear any existing context
        clear_context()

    def tearDown(self):
        """Tear down test environment."""
        # Restore original config
        LoggingConfig.enabled = self.original_enabled
        LoggingConfig.sampling_rate = self.original_sampling_rate
        LoggingConfig.default_threshold_ms = self.original_threshold

        # Clear context
        clear_context()

    def test_context_management(self):
        """Testing thread-local context management."""
        # Set and verify a value
        set_context_value("test_key", "test_value")
        self.assertEqual(get_context_value("test_key"), "test_value")

        # Test context manager
        with log_context(new_key="new_value"):
            self.assertEqual(get_context_value("test_key"), "test_value")
            self.assertEqual(get_context_value("new_key"), "new_value")

        # Verify context is restored
        self.assertEqual(get_context_value("test_key"), "test_value")
        self.assertIsNone(get_context_value("new_key"))

    @patch("logging.Logger.log")
    def test_structured_logger(self, mock_log):
        """Testing structured logger functionality."""
        # Set up the mock
        mock_log.return_value = None

        # Get a structured logger wrapper
        structured_logger = get_structured_logger("test_logger")

        # Log a message with context
        with log_context(test_context="context_value"):
            structured_logger.info("Test message")

        # Verify the log method was called
        mock_log.assert_called()

    @patch("time.perf_counter")
    def test_time_operation_decorator(self, mock_perf_counter):
        """Testing time operation decorator functionality."""
        # Mock time.perf_counter to return controlled values - need 3 values for the additional timestamp in with_context
        mock_perf_counter.side_effect = [
            0.0,
            0.05,
            0.1,
        ]  # Start, timestamp in log, end = 100ms duration

        # Create a mock logger
        mock_logger = Mock()
        mock_logger.isEnabledFor.return_value = True

        # Define a test function with the decorator - explicitly use threshold_ms to avoid depending on global config
        @time_operation("test_operation", logger=mock_logger, threshold_ms=0.0)
        def test_function():
            return "result"

        # Call the function
        result = test_function()

        # Verify the function returned the correct result
        self.assertEqual(result, "result")

        # Verify logging occurred
        mock_logger.log.assert_called_once()

        # Reset for threshold test
        mock_logger.reset_mock()

        # Create new function with threshold to avoid depending on global config
        # Set a new side effect with enough values for a complete call with logging
        mock_perf_counter.side_effect = [
            0.0,
            0.005,
            0.01,
        ]  # Start, timestamp in log, end = 10ms (below threshold)

        # Define a new function with high threshold
        @time_operation("test_operation", logger=mock_logger, threshold_ms=100.0)
        def test_function_with_threshold():
            return "result"

        # Call the threshold function
        test_function_with_threshold()

        # Verify no logging occurred
        mock_logger.log.assert_not_called()

    def test_configure_logging(self):
        """Testing logging configuration functionality."""
        # Test with different environments
        with patch.object(LoggingConfig, "configure") as mock_configure:
            configure_logging(environment="development")
            mock_configure.assert_called_once()

            # Reset and test another environment
            mock_configure.reset_mock()
            configure_logging(environment="production")
            mock_configure.assert_called_once()

            # Test with explicit parameters
            mock_configure.reset_mock()
            configure_logging(
                environment="production", sampling_rate=0.5, threshold_ms=100
            )
            mock_configure.assert_called_once()

            # Verify the correct parameters were passed
            args = mock_configure.call_args[1]
            self.assertEqual(args.get("sampling_rate"), 0.5)
            self.assertEqual(args.get("default_threshold_ms"), 100)

    def test_enrich_exception(self):
        """Testing exception enrichment functionality."""
        from ember.xcs.exceptions import XCSError

        # Set context values directly
        set_context_value("operation_id", "test_op")
        set_context_value("node_id", "test_node")

        # Test with an XCS error that supports context
        error = XCSError("Test error")

        # Diagnostic context is now automatically populated with caller information
        # Just verify that the context exists but doesn't contain our custom values yet
        self.assertNotIn(
            "operation_id",
            error.diagnostic_context,
            "Context should not contain our custom values yet",
        )
        self.assertNotIn(
            "node_id",
            error.diagnostic_context,
            "Context should not contain our custom values yet",
        )
        self.assertNotIn(
            "extra_context",
            error.diagnostic_context,
            "Context should not contain our custom values yet",
        )

        # Enrich the exception with context
        enriched = enrich_exception(error, extra_context="value")

        # Verify explicit context was added
        self.assertEqual(
            enriched.diagnostic_context.get("extra_context"),
            "value",
            f"Explicit context missing: {enriched.diagnostic_context}",
        )

        # Verify thread-local context was added
        self.assertEqual(
            enriched.diagnostic_context.get("operation_id"),
            "test_op",
            f"Thread context missing: {enriched.diagnostic_context}",
        )
        self.assertEqual(
            enriched.diagnostic_context.get("node_id"),
            "test_node",
            f"Thread context missing: {enriched.diagnostic_context}",
        )


# These function-style tests are kept for backward compatibility
# but are less robust than the class-based tests above


def test_context_management():
    """Legacy test for thread-local context management."""
    test_obj = TestStructuredLogging()
    test_obj.setUp()
    try:
        test_obj.test_context_management()
    finally:
        test_obj.tearDown()


def test_structured_logger():
    """Legacy test for structured logger functionality."""
    with patch("logging.Logger.log"):
        # Get a structured logger wrapper
        structured_logger = get_structured_logger("test_structured_logger")

        # Log a message with context
        with log_context(test_context="context_value"):
            structured_logger.info("Test message")

        # We're not asserting anything here, just making sure it doesn't crash


def test_time_operation_decorator():
    """Legacy test for time operation decorator functionality."""
    with patch("time.perf_counter") as mock_perf_counter:
        # Need more perf_counter values to account for the additional timestamp in with_context
        mock_perf_counter.side_effect = [
            0.0,
            0.05,
            0.1,
            0.0,
            0.005,
            0.01,
        ]  # Three values per call, two calls

        original_enabled = LoggingConfig.enabled
        original_sampling_rate = LoggingConfig.sampling_rate
        original_threshold = LoggingConfig.default_threshold_ms

        try:
            LoggingConfig.enabled = True
            LoggingConfig.sampling_rate = 1.0
            LoggingConfig.default_threshold_ms = 0.0

            # Create a mock logger
            mock_logger = Mock()
            mock_logger.isEnabledFor.return_value = True

            # Define a test function with explicit thresholds to avoid depending on the global config
            @time_operation("test_operation", logger=mock_logger, threshold_ms=0.0)
            def test_function():
                return "result"

            # Call the function - this should log
            result = test_function()

            # Verify the logger was called
            assert mock_logger.log.call_count == 1
            mock_logger.reset_mock()

            # Define a new function with high threshold rather than changing global config
            @time_operation("test_operation", logger=mock_logger, threshold_ms=100.0)
            def test_function_with_threshold():
                return "result"

            # Call with high threshold - should not log due to threshold
            test_function_with_threshold()

            # Verify the logger was not called
            assert mock_logger.log.call_count == 0
        finally:
            # Restore original settings
            LoggingConfig.enabled = original_enabled
            LoggingConfig.sampling_rate = original_sampling_rate
            LoggingConfig.default_threshold_ms = original_threshold


def test_configure_logging():
    """Legacy test for logging configuration functionality."""
    with patch.object(LoggingConfig, "configure") as mock_configure:
        configure_logging(environment="development")
        mock_configure.assert_called_once()


def test_enrich_exception():
    """Legacy test for exception enrichment functionality."""
    from ember.xcs.exceptions import XCSError

    # Create a clean context
    clear_context()
    set_context_value("operation_id", "test_op")
    set_context_value("node_id", "test_node")

    # Test with an XCS error that supports context
    error = XCSError("Test error")

    # Verify our custom values are not in context yet
    assert "operation_id" not in error.diagnostic_context
    assert "extra_context" not in error.diagnostic_context

    # Enrich the exception with context
    enriched = enrich_exception(error, extra_context="value")

    # Verify our values were added
    assert enriched.diagnostic_context.get("operation_id") == "test_op"
    assert enriched.diagnostic_context.get("extra_context") == "value"

    # Clean up
    clear_context()


if __name__ == "__main__":
    unittest.main()
