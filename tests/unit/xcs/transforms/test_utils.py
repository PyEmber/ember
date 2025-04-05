"""
Utilities for testing XCS transforms.

This module provides common utilities and helper functions used across multiple
transform test modules, promoting code reuse and consistency.
"""

import time
from typing import Any, Callable, Dict, List, Tuple


def generate_batch_inputs(
    batch_size: int, prefix: str = "item"
) -> Dict[str, List[str]]:
    """Generate a batch of test inputs with the given size.

    Args:
        batch_size: Number of items to generate
        prefix: Prefix for each item (default: "item")

    Returns:
        Dictionary with a "prompts" key mapping to a list of generated inputs
    """
    return {"prompts": [f"{prefix}{i}" for i in range(batch_size)]}


def assert_processing_time(
    sequential_time: float,
    parallel_time: float,
    min_speedup: float = 2.0,  # Increased from 1.2 for more rigorous testing
    max_overhead_factor: float = 2.0,  # Decreased from 3.0 for stricter overhead limits
) -> None:
    """Assert that parallel processing is faster than sequential processing.

    Args:
        sequential_time: Time taken for sequential processing
        parallel_time: Time taken for parallel processing
        min_speedup: Minimum expected speedup factor (default: 2.0)
        max_overhead_factor: Maximum overhead factor for small inputs (default: 2.0)

    Raises:
        AssertionError: If the parallel time doesn't meet the speedup expectations
    """
    # For very small inputs or test environments, parallel might be slower due to overhead
    # Still enforce some standards even for small inputs
    if sequential_time < 0.05:  # Reduced threshold from 0.1
        # Even for very small timings, ensure no extreme overhead
        assert (
            parallel_time < sequential_time * max_overhead_factor * 2
        ), f"Extreme overhead for tiny inputs: {sequential_time:.6f}s vs {parallel_time:.6f}s"
    elif sequential_time < 0.3:  # Reduced threshold from 0.5
        # For small inputs, allow moderate overhead but still enforce limits
        assert (
            parallel_time < sequential_time * max_overhead_factor
        ), f"Parallel processing overhead is too high: {sequential_time:.6f}s vs {parallel_time:.6f}s"
    else:
        # For substantial inputs, parallel must demonstrate clear speedup
        speedup = sequential_time / parallel_time
        assert (
            speedup >= min_speedup
        ), f"Expected minimum speedup of {min_speedup}x, but got {speedup:.3f}x"


def time_function_execution(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Tuple[float, Any]:
    """Execute a function and measure its execution time.

    Args:
        func: The function to execute
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Tuple containing (execution_time, function_result)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return (end_time - start_time), result


def count_unique_threads(thread_ids_dict: Dict[int, Any]) -> int:
    """Count the number of unique thread IDs in a dictionary.

    Args:
        thread_ids_dict: Dictionary with thread IDs as keys

    Returns:
        Number of unique thread IDs
    """
    return len(thread_ids_dict.keys())
