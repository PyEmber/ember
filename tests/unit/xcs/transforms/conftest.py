"""
Configuration for XCS transforms tests.

This module sets up performance test options and other fixtures shared across
transform test modules.
"""

import os

import pytest


def pytest_addoption(parser):
    """Add custom command-line options for XCS transform tests."""
    parser.addoption(
        "--run-perf-tests",
        action="store_true",
        default=False,
        help="Run performance tests which may take longer to execute",
    )


# Setup test environment for all mesh tests
def pytest_configure(config):
    """Configure test environment for XCS transform tests."""
    # Enable test mode for mesh tests
    os.environ["_TEST_MODE"] = "1"


def pytest_unconfigure(config):
    """Clean up after tests."""
    # Clean up test mode flag
    if "_TEST_MODE" in os.environ:
        del os.environ["_TEST_MODE"]
