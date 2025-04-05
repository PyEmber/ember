"""
Example demonstrating the use of XCS transformations: vmap, pmap/pjit, and mesh.

This example shows how to use the various XCS transformation APIs to parallelize and
distribute computations across devices.

To run:
    uv run python src/ember/examples/data/transformation_example.py
"""

import argparse
import multiprocessing
import sys
import threading
import time
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from pydantic import BaseModel

# Progress tracking utilities
from tqdm import tqdm

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification

# Note: Unused import removed to adhere to clean code practices.
from ember.xcs.transforms import (
    DeviceMesh,
    PartitionSpec,
    mesh_sharded,
    pjit,
    pmap,
    vmap,
)


class ProgressTracker:
    """Tracks progress across multiple parallel processes.

    This class provides a reusable mechanism for tracking progress in parallel
    processing scenarios, supporting tqdm progress bars.
    """

    def __init__(self, total: int, description: str, shared: bool = False) -> None:
        """Initialize a progress tracker.

        Args:
            total: Total number of items to process
            description: Description for the progress bar
            shared: Whether this progress bar is shared across processes
        """
        self.total = total
        self.description = description
        self.shared = shared
        self._lock = threading.Lock()
        self._progress = 0
        self._active = False
        self._pbar = None

    def __enter__(self) -> "ProgressTracker":
        """Context manager entry point."""
        self._active = True
        self._pbar = tqdm(
            total=self.total,
            desc=self.description,
            leave=True,
            position=0,
            file=sys.stdout,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point."""
        self._active = False
        if self._pbar:
            self._pbar.close()

    def update(self, amount: int = 1) -> None:
        """Update progress by the specified amount.

        Args:
            amount: Amount to increment progress
        """
        if not self._active:
            return

        with self._lock:
            self._progress += amount
            if self._pbar:
                self._pbar.update(amount)


def _time_function_call(
    callable_obj: Callable[..., Any], **kwargs: Any
) -> Tuple[float, Any]:
    """Execute a callable with keyword arguments and measure its execution time.

    Args:
        callable_obj: The function (or callable) to execute.
        **kwargs: Arbitrary keyword arguments to pass to the callable.

    Returns:
        A tuple containing:
            - The elapsed time in seconds.
            - The result returned by the callable.
    """
    start: float = perf_counter()
    result: Any = callable_obj(**kwargs)
    elapsed: float = perf_counter() - start
    return elapsed, result


class SimpleInput(BaseModel):
    prompts: Any  # Accept any type for prompts


class SimpleOutput(BaseModel):
    results: List[str]


class SimpleSpecification(Specification):
    input_model: Type[BaseModel] = SimpleInput
    structured_output: Type[BaseModel] = SimpleOutput


class SimpleOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    """A simple operator that processes input prompts with CPU-intensive operations.

    This operator performs computation-heavy work that benefits from parallelization,
    demonstrating the performance advantages of different transformation strategies.
    """

    specification = SimpleSpecification()

    # Class-level configuration
    is_heavyweight: bool = False
    progress_tracker: Optional[ProgressTracker] = None

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process the provided inputs with CPU-intensive operations.

        Args:
            inputs: A validated SimpleInput model. The prompts field should contain
                a single string or a list of strings to be processed.

        Returns:
            A dictionary with the key 'results' mapping to a list of processed prompts.
        """
        # Get prompts from validated input
        prompts: Any = inputs.prompts

        # Perform CPU-intensive computation
        def cpu_intensive_task(text: str) -> str:
            """Perform an EXTREMELY CPU-intensive calculation on the input text.

            This function is designed to create a massively compute-intensive workload
            that will clearly demonstrate parallelization benefits. It uses a combination
            of techniques that are guaranteed to be CPU-bound.
            """
            # Start with the input text
            result = text

            # Determine workload intensity based on mode
            if SimpleOperator.is_heavyweight:
                # MESH-OPTIMIZED computation that will show parallelization benefits
                # The key to mesh parallelization is having computation that:
                # 1. Requires minimal communication between workers
                # 2. Has high compute-to-communication ratio
                # 3. Can be broken into independent chunks

                # Use a deterministic seed based on input text for reproducibility
                # This also ensures different inputs get different workloads
                seed = abs(hash(text)) % 10000
                np.random.seed(seed)

                # Matrix operations optimized for mesh parallelization
                # Using smaller matrices with many iterations is better for mesh
                # as it minimizes memory overhead and communication costs
                matrix_size = 300  # Smaller matrices, more iterations
                iterations = 20  # More iterations of independent computations

                result_accumulator = 0.0

                # Multiple independent iterations that can be distributed across mesh
                for iter_idx in range(iterations):
                    if SimpleOperator.progress_tracker and iter_idx % 2 == 0:
                        SimpleOperator.progress_tracker.update()

                    # Create new matrices each iteration to ensure high computation
                    matrix_a = np.random.rand(matrix_size, matrix_size)
                    matrix_b = np.random.rand(matrix_size, matrix_size)

                    # Compute matrix multiplication (O(n³) complexity)
                    result_matrix = np.matmul(matrix_a, matrix_b)

                    # Apply additional computation to increase CPU intensity
                    # Element-wise operations are very efficient for parallelization
                    result_matrix = np.sin(result_matrix) + np.cos(result_matrix * 0.5)
                    result_matrix = np.power(result_matrix, 2)

                    # Aggregate result using reduction operation
                    result_accumulator += np.sum(result_matrix)

                # Include result in output
                result = f"{result}:{abs(int(result_accumulator)) % 10000}"

                # Perfect for mesh - compute-intensive with minimal data dependencies
                # Each chunk is completely independent and can run on separate mesh cells

                def compute_mandelbrot_segment(
                    x_min: float,
                    x_max: float,
                    y_min: float,
                    y_max: float,
                    width: int,
                    height: int,
                    max_iterations: int,
                ) -> int:
                    """Compute Mandelbrot set density in a region (highly parallelizable)."""
                    total = 0

                    # For each pixel in the segment
                    for ix in range(width):
                        x0 = x_min + (x_max - x_min) * ix / width

                        for iy in range(height):
                            y0 = y_min + (y_max - y_min) * iy / height
                            x, y = 0.0, 0.0
                            iteration = 0

                            # Mandelbrot iteration
                            while x * x + y * y < 4.0 and iteration < max_iterations:
                                x_new = x * x - y * y + x0
                                y = 2 * x * y + y0
                                x = x_new
                                iteration += 1

                            # Count points that escape slowly (in the set)
                            if iteration > max_iterations // 2:
                                total += 1

                    return total

                # Divide computation into independent chunks for mesh parallelization
                # Each chunk computes a different region of the Mandelbrot set
                regions = 8  # Number of independent regions to compute
                mandelbrot_count = 0

                # These chunks are perfect for mesh cells - completely independent work
                for i in range(regions):
                    # Generate different regions based on input seed
                    x_shift = (seed % 100) / 100.0
                    y_shift = ((seed // 100) % 100) / 100.0

                    # Different region for each chunk
                    x_min = -2.0 + 0.5 * (i % 4) + x_shift
                    y_min = -1.5 + 0.5 * (i // 4) + y_shift

                    # Compute the mandelbrot set for this region
                    segment_result = compute_mandelbrot_segment(
                        x_min, x_min + 0.5, y_min, y_min + 0.5, 80, 80, 1000
                    )

                    mandelbrot_count += segment_result

                    if SimpleOperator.progress_tracker and i % 2 == 0:
                        SimpleOperator.progress_tracker.update()

                # Add the mandelbrot computation result
                result = f"{result}:{mandelbrot_count % 10000}"

            else:
                # Standard lightweight computation for quick demo
                def is_prime(n: int) -> bool:
                    """Check if a number is prime."""
                    if n <= 1:
                        return False
                    if n <= 3:
                        return True
                    if n % 2 == 0 or n % 3 == 0:
                        return False
                    i = 5
                    while i * i <= n:
                        if n % i == 0 or n % (i + 2) == 0:
                            return False
                        i += 6
                    return True

                # Find a small number of primes only
                seed = abs(hash(text)) % 10000 + 10000
                count = 0
                num = seed
                prime_count_target = 200

                while count < prime_count_target:
                    if is_prime(num):
                        result = f"{result}:{num % 100}"
                        count += 1
                    num += 1

            # Return a consistent result format
            return f"{text} -> processed:{hash(result) % 10000}"

        # Process each prompt with our CPU-intensive task
        if isinstance(prompts, list):
            processed_results: List[str] = [
                cpu_intensive_task(prompt) for prompt in prompts
            ]
        else:
            processed_results = [cpu_intensive_task(prompts)]

        return {"results": processed_results}


def demonstrate_vmap() -> None:
    """Demonstrate the vmap (vectorized mapping) transformation.

    This function creates a vectorized version of a simple operator and compares
    its performance against sequential processing by timing both approaches.
    VMAP is ideal for batch processing without communication between items.
    """
    print("\n=== VMAP Demonstration (Vectorized Mapping) ===")
    print("VMAP transforms a function that operates on single elements into")
    print("one that efficiently processes multiple inputs in parallel.")
    print("It's ideal for batch processing with minimal overhead.\n")

    # In heavyweight mode, use a smaller batch size to make each item extremely intensive
    # Rather than many small items, we'll use fewer extremely heavy items
    cpu_count = multiprocessing.cpu_count()
    if SimpleOperator.is_heavyweight:
        # Use a batch size that's enough to demonstrate vectorization benefits
        # but not so large that each item takes too long
        batch_size = max(4, cpu_count)  # Ensure at least 4 items for good demonstration
    else:
        # Use more items for lightweight demo
        batch_size = cpu_count * 2

    print(
        f"Processing batch of {batch_size} items with{'out' if not SimpleOperator.is_heavyweight else ''} heavy computation..."
    )

    simple_operator: SimpleOperator = SimpleOperator()
    vectorized_operator: Callable[..., Any] = vmap(simple_operator)

    # Create batch inputs with appropriate size
    batch_inputs: Dict[str, List[str]] = {
        "prompts": [f"VMap item {i:03d} batch" for i in range(batch_size)]
    }

    print("\nRunning sequential processing (one item at a time)...")
    # Time sequential processing: apply the operator separately for each prompt.
    start_seq: float = perf_counter()
    sequential_results: List[Dict[str, Any]] = [
        simple_operator(inputs={"prompts": prompt})
        for prompt in batch_inputs["prompts"]
    ]
    sequential_time: float = perf_counter() - start_seq
    print(f"Sequential processing time: {sequential_time:.4f}s")

    print("\nRunning vectorized processing (all items at once)...")
    # Time vectorized processing: apply the operator once across all inputs.
    start_vec: float = perf_counter()
    vectorized_results: Dict[str, Any] = vectorized_operator(inputs=batch_inputs)
    vectorized_time: float = perf_counter() - start_vec
    print(f"Vectorized processing time: {vectorized_time:.4f}s")

    if vectorized_time > 0 and sequential_time > 0:
        speedup: float = sequential_time / vectorized_time
        print(f"Speedup: {speedup:.2f}x")
        # Highlight significant speedups
        if speedup > 1.5:
            print(f"🚀 SIGNIFICANT SPEEDUP ACHIEVED: {speedup:.2f}x faster!")
            print("Vectorized processing is efficiently handling the batch!")
        elif speedup > 1.0:
            print(f"✓ Speedup achieved: {speedup:.2f}x faster")
            print("VMAP is showing benefits for batch processing.")
        else:
            print("⚠️ No speedup detected. This can happen when:")
            print("  1. The operation has overhead that negates vectorization benefits")
            print("  2. The batch size is too small to amortize setup costs")
            print("  3. The sequential implementation is already optimized")
            print("Try with the --heavy flag or larger batch sizes.")

    # Display sample results from the vectorized operator.
    print("\nResults from vectorized operator (sample):")
    results = vectorized_results.get("results", [])
    sample_size = min(3, len(results))
    for result in results[:sample_size]:
        print(f"  {result}")
    if len(results) > sample_size:
        print(f"  ... and {len(results) - sample_size} more results")


def demonstrate_pmap() -> None:
    """Demonstrate the pmap (parallel mapping) transformation.

    This function creates a parallelized operator and compares its performance on a batch
    of inputs against the sequential execution of the operator. PMAP distributes
    computation across available devices with minimal code changes.
    """
    print("\n=== PMAP Demonstration (Parallel Mapping) ===")
    print("PMAP automatically distributes computation across available CPU cores.")
    print("It's a simple yet powerful way to parallelize computation with minimal")
    print("code changes, offering good performance for many workloads.\n")

    # In heavyweight mode, use a batch size optimized for parallelism
    cpu_count = multiprocessing.cpu_count()
    if SimpleOperator.is_heavyweight:
        # For heavy computation, use exactly one task per core
        # The optimal batch size for PMAP is typically one item per available device
        batch_size = cpu_count
    else:
        # Use more items for lightweight demo
        batch_size = cpu_count * 2

    print(
        f"Processing batch of {batch_size} items with{'out' if not SimpleOperator.is_heavyweight else ''} parallel-optimized computation..."
    )
    print(f"System has {cpu_count} CPU cores available for parallelization")

    simple_operator: SimpleOperator = SimpleOperator()
    parallel_operator: Callable[..., Any] = pmap(simple_operator)

    # Create batch inputs with appropriate size
    batch_inputs: Dict[str, List[str]] = {
        "prompts": [f"PMap item {i:03d} core{i%cpu_count}" for i in range(batch_size)]
    }

    print("\nRunning sequential processing (single-threaded)...")
    # Time sequential processing on the batch.
    sequential_time, sequential_results = _time_function_call(
        simple_operator, inputs=batch_inputs
    )
    print(f"Sequential processing time: {sequential_time:.4f}s")

    print("\nRunning parallel processing (multi-threaded with pmap)...")
    # Time parallelized processing on the batch.
    parallel_time, parallel_results = _time_function_call(
        parallel_operator, inputs=batch_inputs
    )
    print(f"Parallel processing time: {parallel_time:.4f}s")

    if parallel_time > 0 and sequential_time > 0:
        speedup: float = sequential_time / parallel_time
        print(f"Speedup: {speedup:.2f}x")
        # Highlight significant speedups
        if speedup > 1.5:
            print(f"🚀 SIGNIFICANT SPEEDUP ACHIEVED: {speedup:.2f}x faster!")
            print(f"Parallel processing is effectively using multiple CPU cores!")

            # Calculate efficiency relative to theoretical maximum
            theoretical_max = min(cpu_count, batch_size)
            efficiency = (speedup / theoretical_max) * 100
            print(
                f"Parallelization efficiency: {efficiency:.1f}% of theoretical maximum"
            )

            if efficiency > 75:
                print(
                    "Excellent efficiency! The computation is well-suited for parallelization."
                )
            elif efficiency > 50:
                print(
                    "Good efficiency. Some overhead, but still effective parallelization."
                )
            else:
                print(
                    "Moderate efficiency. Consider optimizing for better parallel scaling."
                )
        elif speedup > 1.0:
            print(f"✓ Speedup achieved: {speedup:.2f}x faster")
            print("PMAP is utilizing multiple cores, showing performance benefits.")
        else:
            print("⚠️ No speedup detected. This can happen when:")
            print("  1. The operation has high thread coordination overhead")
            print("  2. The computation is too light to benefit from parallelization")
            print("  3. System resources are already constrained")
            print("Try with the --heavy flag for more intensive computation.")
    else:
        print("Parallel processing time too small to calculate speedup")

    # Display sample results from the parallel operator.
    print("\nResults from parallel operator (sample):")
    results = parallel_results.get("results", [])
    sample_size = min(3, len(results))
    for result in results[:sample_size]:
        print(f"  {result}")
    if len(results) > sample_size:
        print(f"  ... and {len(results) - sample_size} more results")


def create_adaptive_mesh() -> Tuple[DeviceMesh, Dict[str, PartitionSpec]]:
    """Create an optimal device mesh based on available system resources.

    This function detects system capabilities and constructs a mesh configuration
    that will work efficiently on the current hardware. It handles various CPU
    counts gracefully and creates an appropriate partition specification.

    Returns:
        A tuple containing:
            - A configured DeviceMesh instance
            - An appropriate partition specification for the mesh

    Raises:
        ValueError: If unable to create a valid mesh configuration.
    """
    # Get available CPU cores, reserving at least one for system processes
    available_cores: int = max(2, multiprocessing.cpu_count() - 1)

    # Create devices list explicitly to avoid automatic device detection
    devices: List[str] = [f"cpu:{i}" for i in range(available_cores)]

    # Determine optimal mesh shape based on available cores
    if available_cores >= 4:
        # For 4+ cores, create a 2D mesh with shape that divides evenly
        # Calculate factors to find a balanced 2D shape
        for i in range(int(available_cores**0.5), 0, -1):
            if available_cores % i == 0:
                rows: int = i
                cols: int = available_cores // i
                break
        else:
            # Fallback: use a subset of cores for a clean 2x2 configuration
            rows, cols = 2, 2
            devices = devices[:4]  # Use only the first 4 devices

        mesh_shape: Tuple[int, ...] = (rows, cols)

        # Use explicit sharding along both dimensions for better efficiency
        # This optimizes the distribution of work across all mesh cells
        partition_spec: Dict[str, PartitionSpec] = {
            "prompts": PartitionSpec(0, 1)  # Shard along both dimensions
        }
    else:
        # For fewer cores, create a simple 1D mesh
        mesh_shape = (len(devices),)
        partition_spec = {"prompts": PartitionSpec(0)}  # Shard along the only dimension

    # Create the mesh with explicit devices and shape
    device_mesh: DeviceMesh = DeviceMesh(devices=devices, shape=mesh_shape)
    return device_mesh, partition_spec


def demonstrate_mesh() -> None:
    """Demonstrate the device mesh transformation.

    This function creates an adaptive device mesh based on available system resources
    and demonstrates distributed computation through mesh sharding. The implementation
    is resilient to varying hardware environments and will adapt accordingly.
    """
    print("\n=== Device Mesh Demonstration ===")

    simple_operator: SimpleOperator = SimpleOperator()

    try:
        # Create an adaptive mesh configuration
        device_mesh, partition_spec = create_adaptive_mesh()
        print(f"Created mesh: {device_mesh}")
        print(f"Partition spec: {partition_spec}")

        # Create the sharded operator
        sharded_operator: Callable[..., Any] = mesh_sharded(
            simple_operator, device_mesh, in_partition=partition_spec
        )

        # Customize batch size for mesh processing
        cpu_count = multiprocessing.cpu_count()

        # For best mesh performance, each batch item needs to be substantial
        # but we need enough items to fully utilize the mesh structure
        if SimpleOperator.is_heavyweight:
            # Create enough items to fully utilize all mesh cells with a 2x multiplier
            # to ensure each mesh cell gets multiple items for better load balancing
            mesh_size = np.prod(device_mesh.shape)
            batch_size = mesh_size * 2
        else:
            # Use more items for lightweight demo
            batch_size = cpu_count * 2

        print(
            f"Processing batch of {batch_size} items with{'out' if not SimpleOperator.is_heavyweight else ''} mesh-optimized computation..."
        )

        # Create input data with IDs that ensure even distribution
        batch_inputs: Dict[str, List[str]] = {
            "prompts": [
                f"Mesh task {i:03d} region {i % np.prod(device_mesh.shape)}"
                for i in range(batch_size)
            ]
        }

        print("\nMesh parallelization benefits explanation:")
        print("- Distributes work across a structured grid of devices")
        print("- Shards data along multiple dimensions (unlike simple pmap)")
        print("- Allows fine-grained control over work distribution")
        print("- Minimizes communication overhead between workers")
        print("- Especially beneficial for computation that can be chunked")
        print("  into independent parts with minimal coordination")
        print("")

        print("Running sequential processing...")
        # Time sequential processing on the batch
        sequential_time, sequential_results = _time_function_call(
            simple_operator, inputs=batch_inputs
        )
        print(f"Sequential processing time: {sequential_time:.4f}s")

        print("\nRunning mesh-sharded processing...")
        # Time mesh-sharded processing on the batch
        sharded_time, sharded_results = _time_function_call(
            sharded_operator, inputs=batch_inputs
        )
        print(f"Mesh-sharded processing time: {sharded_time:.4f}s")

        if sharded_time > 0 and sequential_time > 0:
            speedup: float = sequential_time / sharded_time
            print(f"Speedup: {speedup:.2f}x")
            # Highlight significant speedups
            if speedup > 1.5:
                print(f"🚀 SIGNIFICANT SPEEDUP ACHIEVED: {speedup:.2f}x faster!")
                print("The mesh parallelization is successfully distributing work!")
            elif speedup > 1.0:
                print(f"✓ Speedup achieved: {speedup:.2f}x faster")
                print("Consider increasing computation intensity for greater benefits.")
            else:
                print("⚠️ No speedup detected. This can happen when:")
                print("  1. Overhead of distribution exceeds computation benefits")
                print("  2. The computation is not well-suited for mesh parallelism")
                print("  3. System resources are already saturated")
                print("Try with the --heavy flag for more intensive computation.")
        else:
            print("Mesh-sharded processing time too small to calculate speedup")

        # Display a sample of results (limit output for large result sets)
        print("\nResults from mesh-sharded operator (sample):")
        results = sharded_results.get("results", [])
        sample_size: int = min(5, len(results))
        for result in results[:sample_size]:
            print(f"  {result}")
        if len(results) > sample_size:
            print(f"  ... and {len(results) - sample_size} more results")

    except Exception as e:
        print(f"Error in mesh demonstration: {e}")
        print(
            "Mesh demonstration could not be completed due to resource constraints or configuration issues"
        )
        print("Other transformations (vmap and pmap) should still work correctly")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Demonstrate XCS transformations with parallelizable workloads"
    )
    parser.add_argument(
        "--heavy",
        action="store_true",
        help="Enable heavyweight computation to demonstrate significant parallelization benefits",
    )
    return parser.parse_args()


def main() -> None:
    """Run all transformation demonstrations with CPU-intensive workloads.

    This example demonstrates three different parallelization strategies:
    1. vmap - Vectorized mapping for batch processing of inputs
    2. pmap - Simple parallel execution across available devices
    3. mesh - Sophisticated distribution across a logical grid of devices

    Each strategy is benchmarked against sequential processing to show
    performance benefits for CPU-intensive workloads.
    """
    # Parse command line arguments
    args = parse_args()

    # Configure heavyweight computation if requested
    SimpleOperator.is_heavyweight = args.heavy

    # Welcome message
    print("XCS Transformation Examples")
    print("==========================")
    print("This example demonstrates three parallelization strategies:")
    print("  1. vmap - Vectorized mapping across batch dimensions")
    print("  2. pmap - Simple parallel execution across devices")
    print("  3. mesh - Advanced sharding across a structured grid of devices")
    print("")

    # Report computation mode
    if args.heavy:
        print("HEAVYWEIGHT COMPUTATION MODE ENABLED")
        print("This mode performs significantly more intensive calculations")
        print("to clearly demonstrate parallelization benefits.")
        print("Expected runtime: ~1-2 minutes per transformation.")
        print("")
    else:
        print("Using lightweight computation mode (for quick demonstration)")
        print("To see significant performance gains, run with --heavy flag")
        print(
            "Example: uv run python src/ember/examples/data/transformation_example.py --heavy"
        )
        print("")

    # Configure progress tracking if in heavyweight mode
    if args.heavy:
        print("Setting up progress tracking for intensive operations...")
        SimpleOperator.progress_tracker = ProgressTracker(
            total=100, description="Processing", shared=True  # Arbitrary progress units
        )

    # Run demonstrations
    try:
        if args.heavy and SimpleOperator.progress_tracker:
            with SimpleOperator.progress_tracker:
                demonstrate_vmap()
                demonstrate_pmap()
                demonstrate_mesh()
        else:
            demonstrate_vmap()
            demonstrate_pmap()
            demonstrate_mesh()

        print("\nAll transformations demonstrated successfully!")
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")
    finally:
        # Detailed summary with transformation comparison
        print("\n============== TRANSFORMATION COMPARISON ==============")
        print("\n1. VMAP (Vectorized Mapping)")
        print("   Best for: Batch processing without communication between items")
        print("   Advantages:")
        print("   - Simplest transformation to implement and understand")
        print("   - No communication overhead between batch items")
        print("   - Perfect for identical operations on many inputs")
        print("   - Minimal framework overhead")
        print("   Ideal use cases:")
        print("   - Preprocessing many input examples")
        print("   - Applying the same function to each item in a batch")
        print("   - Independent data transformations")

        print("\n2. PMAP (Parallel Mapping)")
        print("   Best for: Simple parallelism across available devices")
        print("   Advantages:")
        print("   - Straightforward parallelization with minimal code changes")
        print("   - Automatic utilization of available compute resources")
        print("   - Good for medium-complexity workloads")
        print("   - Balanced workload distribution")
        print("   Ideal use cases:")
        print("   - CPU-bound operations that easily divide across cores")
        print("   - When computation greatly exceeds communication costs")
        print("   - Parallel inference across multiple devices")

        print("\n3. MESH (Device Mesh Sharding)")
        print(
            "   Best for: Complex distributed computation across heterogeneous hardware"
        )
        print("   Advantages:")
        print("   - Most sophisticated and flexible parallelization")
        print("   - Multi-dimensional sharding capabilities")
        print("   - Fine-grained control over work distribution")
        print("   - Scales to extremely large computations")
        print("   - Can leverage specialized hardware (GPUs/TPUs) efficiently")
        print("   Ideal use cases:")
        print("   - Training extremely large models")
        print("   - Data-parallel and model-parallel workloads")
        print("   - Sharding computation and model parameters together")
        print("   - High-performance distributed computing")

        print("\n================= SCALING INSIGHTS =================")
        print("For enterprise-scale machine learning workloads, mesh parallelism")
        print("is the transformative technology that has enabled training models")
        print("with trillions of parameters by precisely controlling how")
        print("computation and data are distributed across hardware accelerators.")
        print("")
        print("The dramatic performance differences between these techniques become")
        print("most apparent at scale, especially when handling operations with the")
        print("following characteristics:")
        print("1. High computational intensity per operation")
        print("2. Minimal inter-operation dependencies")
        print("3. Ability to chunk work into independent segments")
        print("4. Operations that match the underlying hardware capabilities")
        print("")
        print("To further explore these capabilities, run additional experiments with")
        print("the --heavy flag and varying input sizes to see how each transformation")
        print("behaves across different workload characteristics.")


if __name__ == "__main__":
    main()
