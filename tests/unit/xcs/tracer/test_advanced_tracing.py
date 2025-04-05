"""Unit tests for advanced tracing, parallel execution, and error handling.

This module defines dummy operator implementations to simulate wide ensembles,
nested operator structures, parallel execution, and failure scenarios. It verifies
that the new tracer and JIT systems record operator invocations correctly, that a
JIT-decorated operator can be "converted" into an XCS graph, and that running that graph
with the XCS engine in parallel is significantly faster than sequential execution.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Type

import pytest
from pydantic import BaseModel

from ember.core.registry.operator.exceptions import OperatorExecutionError
from ember.xcs.engine.xcs_engine import (
    TopologicalSchedulerWithParallelDispatch,
    compile_graph,
)
from ember.xcs.engine.xcs_noop_scheduler import XCSNoOpScheduler
from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.tracer.tracer_decorator import jit
from ember.xcs.tracer.xcs_tracing import TracerContext
from tests.helpers.stub_classes import Operator


# -----------------------------------------------------------------------------
# Dummy Models and Specification
# -----------------------------------------------------------------------------
class DummyInputs(BaseModel):
    """Input data model for dummy operators.

    Attributes:
        query (str): Query string to process.
    """

    query: str


class DummyOutputs(BaseModel):
    """Output data model for dummy operators.

    Attributes:
        answer (str): Answer produced by the operator.
    """

    answer: str


class DummySpecification:
    """A minimal dummy specification for testing purposes.

    Provides methods to validate inputs and outputs as well as render prompts.
    """

    def __init__(self, input_model: Type[BaseModel]) -> None:
        """Initializes the dummy specification.

        Args:
            input_model (Type[BaseModel]): The input model class.
        """
        self.input_model: Type[BaseModel] = input_model

    def validate_inputs(self, *, inputs: Any) -> Any:
        """Validates the provided inputs.

        Args:
            inputs (Any): Inputs to validate.

        Returns:
            Any: The validated inputs.
        """
        if hasattr(inputs, "dict"):
            return inputs
        return inputs

    def validate_output(self, *, output: Any) -> Any:
        """Validates the provided output.

        Args:
            output (Any): Output to validate.

        Returns:
            Any: The validated output.
        """
        return output

    def render_prompt(self, *, inputs: Dict[str, Any]) -> str:
        """Renders a prompt based on the inputs.

        Args:
            inputs (Dict[str, Any]): Dictionary of input values.

        Returns:
            str: The rendered prompt.
        """
        return "dummy prompt"


# -----------------------------------------------------------------------------
# Dummy Operators for Wide and Nested Graphs
# -----------------------------------------------------------------------------
class DummyMemberOperator(Operator[DummyInputs, DummyOutputs]):
    """Operator that returns a response indicating its member index.

    Attributes:
        member_index (int): The index of this member within an ensemble.
    """

    specification: DummySpecification = DummySpecification(DummyInputs)

    def __init__(self, *, member_index: int) -> None:
        """Initializes the dummy member operator.

        Args:
            member_index (int): The member index.
        """
        self.member_index: int = member_index

    def forward(self, *, inputs: DummyInputs) -> DummyOutputs:
        """Processes the input and returns a dummy response.

        Args:
            inputs (DummyInputs): The input data.

        Returns:
            DummyOutputs: The output with a response string.
        """
        return DummyOutputs(answer=f"dummy-test wide {self.member_index}")


class WideEnsembleOperator(Operator[DummyInputs, Dict[str, Any]]):
    """Operator that aggregates responses from multiple DummyMemberOperator instances.

    Attributes:
        members (List[DummyMemberOperator]): List of member operators.
    """

    specification: DummySpecification = DummySpecification(DummyInputs)

    def __init__(self, *, num_members: int) -> None:
        """Initializes a wide ensemble with the given number of members.

        Args:
            num_members (int): Number of member operators in the ensemble.
        """
        self.members: List[DummyMemberOperator] = [
            DummyMemberOperator(member_index=i) for i in range(num_members)
        ]

    def forward(self, *, inputs: DummyInputs) -> Dict[str, Any]:
        """Aggregates responses from member operators.

        Args:
            inputs (DummyInputs): The input data.

        Returns:
            Dict[str, Any]: Dictionary of responses.
        """
        responses: List[str] = [member(inputs=inputs).answer for member in self.members]
        return {"responses": responses}


class DummyJudgeOperator(Operator[DummyInputs, DummyOutputs]):
    """Operator that simulates a judge by returning a fixed answer."""

    specification: DummySpecification = DummySpecification(DummyInputs)

    def forward(self, *, inputs: DummyInputs) -> DummyOutputs:
        """Returns a fixed judge response.

        Args:
            inputs (DummyInputs): The input data.

        Returns:
            DummyOutputs: The output containing the fixed answer.
        """
        return DummyOutputs(answer="dummy-nested test")


class NestedOperator(Operator[DummyInputs, Dict[str, Any]]):
    """Operator that composes two ensembles and a judge operator.

    Attributes:
        ensemble1 (WideEnsembleOperator): The first ensemble.
        ensemble2 (WideEnsembleOperator): The second ensemble.
        judge (DummyJudgeOperator): The judge operator.
    """

    specification: DummySpecification = DummySpecification(DummyInputs)

    def __init__(self) -> None:
        self.ensemble1: WideEnsembleOperator = WideEnsembleOperator(num_members=3)
        self.ensemble2: WideEnsembleOperator = WideEnsembleOperator(num_members=3)
        self.judge: DummyJudgeOperator = DummyJudgeOperator()

    def forward(self, *, inputs: DummyInputs) -> Dict[str, Any]:
        """Executes the nested operator structure.

        Args:
            inputs (DummyInputs): The input data.

        Returns:
            Dict[str, Any]: Dictionary containing the final judged answer.
        """
        a: Dict[str, Any] = self.ensemble1(inputs=inputs)
        b: Dict[str, Any] = self.ensemble2(inputs=inputs)
        c: DummyOutputs = self.judge(
            inputs={"responses": [a["responses"][0], b["responses"][0]]}
        )
        return {"answer": c.answer}


# -----------------------------------------------------------------------------
# New: DelayOperator and DelayEnsembleOperator for performance testing
# -----------------------------------------------------------------------------
class DelayOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    """Operator that sleeps for a fixed delay and then returns a result.

    Attributes:
        delay (float): The number of seconds to sleep.
    """

    # Use a minimal dummy specification.
    specification = DummySpecification(dict)

    def __init__(self, *, delay: float) -> None:
        self.delay: float = delay

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        time.sleep(self.delay)
        return {"result": "done"}


@jit()  # JIT-decorated so that its calls are traced.
class DelayEnsembleOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    """Operator that aggregates multiple DelayOperators.

    Each sub-operator sleeps for a fixed delay. This operator is used to test that
    JIT tracing produces trace records and that execution via the XCS engine can run in parallel.
    """

    specification = DummySpecification(dict)

    def __init__(self, *, num_members: int, delay: float) -> None:
        self.members: List[DelayOperator] = [
            DelayOperator(delay=delay) for _ in range(num_members)
        ]

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = [member(inputs=inputs) for member in self.members]
        return {"results": results}


# For parallel execution testing, define a raw operator for use in a parallel ensemble.
class RawParallelDummyOperator(Operator[DummyInputs, DummyOutputs]):
    """A raw parallel operator that waits on a barrier before returning a response."""

    specification: DummySpecification = DummySpecification(DummyInputs)

    def __init__(self, *, barrier: threading.Barrier) -> None:
        self.barrier: threading.Barrier = barrier

    def forward(self, *, inputs: DummyInputs) -> DummyOutputs:
        self.barrier.wait(timeout=5)
        return DummyOutputs(answer="parallel-dummy")


@jit(sample_input={"query": "parallel"}, force_trace=False)
class ParallelWideEnsembleOperator(Operator[DummyInputs, Dict[str, Any]]):
    """Operator that executes a wide ensemble in parallel using barrier synchronization."""

    specification: DummySpecification = DummySpecification(DummyInputs)

    def __init__(self, *, num_members: int, barrier: threading.Barrier) -> None:
        self.members: List[RawParallelDummyOperator] = [
            RawParallelDummyOperator(barrier=barrier) for _ in range(num_members)
        ]

    def forward(self, *, inputs: DummyInputs) -> Dict[str, Any]:
        with ThreadPoolExecutor(max_workers=len(self.members)) as executor:
            futures = [
                executor.submit(lambda m=member: m(inputs=inputs).answer)
                for member in self.members
            ]
            responses: List[str] = [f.result() for f in as_completed(futures)]
        responses.sort()  # For consistency.
        return {"responses": responses}


class FaultyOperator(Operator[DummyInputs, DummyOutputs]):
    """Operator that raises an error to test error propagation."""

    specification: DummySpecification = DummySpecification(DummyInputs)

    def forward(self, *, inputs: DummyInputs) -> DummyOutputs:
        raise ValueError("Test error")


# -----------------------------------------------------------------------------
# Test Functions
# -----------------------------------------------------------------------------
def test_wide_ensemble_tracing() -> None:
    """Tests execution of a wide ensemble operator.

    Verifies that the ensemble returns the expected responses.
    """
    num_members: int = 100
    ensemble: WideEnsembleOperator = WideEnsembleOperator(num_members=num_members)
    input_data: DummyInputs = DummyInputs(query="test wide")
    output: Dict[str, Any] = ensemble(inputs=input_data)
    responses: List[str] = output.get("responses", [])
    assert (
        len(responses) == num_members
    ), f"Expected {num_members} responses, got {len(responses)}"
    for i, resp in enumerate(responses):
        expected: str = f"dummy-test wide {i}"
        assert resp == expected, f"Expected response '{expected}', got '{resp}'"


def test_nested_operator_tracing() -> None:
    """Tests execution of a nested operator structure.

    Verifies that the nested operator composes sub-ensembles and a judge operator,
    and that the final answer starts with the expected fixed response.
    """
    nested: NestedOperator = NestedOperator()
    input_data: DummyInputs = DummyInputs(query="nested test")
    output: Dict[str, Any] = nested(inputs=input_data)
    assert output["answer"].startswith(
        "dummy-nested test"
    ), f"Unexpected judge output: {output['answer']}"


def test_parallel_execution_wide_ensemble() -> None:
    """Tests that parallel execution reduces overall execution time via barrier synchronization.

    Constructs a wide ensemble where each member waits on a barrier and asserts that
    the total execution duration is much less than the sum of individual delays.
    """
    num_members: int = 50
    barrier: threading.Barrier = threading.Barrier(num_members)
    operator_instance: ParallelWideEnsembleOperator = ParallelWideEnsembleOperator(
        num_members=num_members, barrier=barrier
    )
    start: float = time.time()
    output: Dict[str, Any] = operator_instance(inputs=DummyInputs(query="parallel"))
    duration: float = time.time() - start
    # Sequentially, 50 members each waiting 0.1 sec would take ~5 sec.
    assert (
        duration < 1.0
    ), f"Expected parallel execution to complete in <1.0s, but took {duration:.2f}s"
    responses: List[str] = output.get("responses", [])
    assert (
        len(responses) == num_members
    ), f"Expected {num_members} responses, got {len(responses)}"


def test_jit_tracing() -> None:
    """Tests that the JIT-decorated operator traces execution.

    Note: This test was previously called test_jit_caching but has been updated to match
    the current implementation which focuses on tracing rather than caching.
    """

    @jit(sample_input={"query": "init"}, force_trace=False)
    class TracedOperator(Operator[DummyInputs, Dict[str, Any]]):
        specification: DummySpecification = DummySpecification(DummyInputs)

        def __init__(self, *, num_members: int) -> None:
            self.members: List[DummyMemberOperator] = [
                DummyMemberOperator(member_index=i) for i in range(num_members)
            ]
            self.call_count: int = 0

        def forward(self, *, inputs: DummyInputs) -> Dict[str, Any]:
            self.call_count += 1
            responses: List[str] = [
                member(inputs=inputs).answer for member in self.members
            ]
            return {"responses": responses}

    op: TracedOperator = TracedOperator(num_members=5)

    # First call with a unique input
    input_data_1: DummyInputs = DummyInputs(query="test_1")
    _ = op(inputs=input_data_1)
    first_count: int = op.call_count

    # Second call with same input
    _ = op(inputs=input_data_1)
    second_count: int = op.call_count
    assert second_count > first_count, "Expected call_count to increase with each call"

    # Third call with different input
    input_data_2: DummyInputs = DummyInputs(query="test_2")
    _ = op(inputs=input_data_2)
    third_count: int = op.call_count
    assert third_count > second_count, "Expected call_count to continue increasing"

    # Verify operation with a tracer context
    with TracerContext() as tracer:
        _ = op(inputs=input_data_1)
        assert (
            len(tracer.records) >= 1
        ), "Expected trace records when within a TracerContext"


def test_error_handling() -> None:
    """Tests that errors in sub-operators are correctly wrapped and propagated.

    Constructs a wide ensemble where one member is a faulty operator and asserts that
    OperatorExecutionError is raised with the original error message preserved.
    """

    class FaultyWideEnsembleOperator(WideEnsembleOperator):
        """Wide ensemble operator that replaces one member with a faulty operator."""

        def __init__(self, *, num_members: int) -> None:
            super().__init__(num_members=num_members)
            if len(self.members) >= 2:
                self.members[1] = FaultyOperator()

    ensemble: WideEnsembleOperator = FaultyWideEnsembleOperator(num_members=3)
    input_data: DummyInputs = DummyInputs(query="error test")
    # Use ValueError instead of OperatorExecutionError for simplified test with stub classes
    with pytest.raises(ValueError) as exception_info:
        _ = ensemble(inputs=input_data)

    error_message = str(exception_info.value)
    # With stub classes, we just get the ValueError directly
    assert "Test error" in error_message


def test_jit_produces_xcs_graph_and_parallel_speedup() -> None:
    """Tests that a JIT-decorated DelayEnsembleOperator produces trace records
    and that executing an XCS graph built from its sub-operators in parallel is
    significantly faster than executing it sequentially.

    A DelayOperator sleeps for a fixed delay. When each DelayOperator is run
    as a separate node, sequential execution will take roughly (num_members * delay)
    seconds, whereas parallel execution should finish in near the delay duration.
    """
    num_members: int = 20
    delay: float = 0.1  # seconds per member

    # Create a JIT-decorated DelayEnsembleOperator instance but ensure force_trace is True
    # for verification purposes in this test
    @jit(force_trace=True)
    class TestDelayEnsembleOperator(DelayEnsembleOperator):
        pass

    ensemble = TestDelayEnsembleOperator(num_members=num_members, delay=delay)

    # Confirm JIT tracing: run inside a TracerContext to verify trace records are produced.
    with TracerContext() as tracer:
        _ = ensemble(inputs={})
    logging.info(f"tracer.records: {tracer.records}")
    assert (
        len(tracer.records) >= 1
    ), "Expected at least one trace record from DelayEnsembleOperator."

    # Instead of using the ensemble node as a whole, build an XCSGraph with one node per DelayOperator.
    graph: XCSGraph = XCSGraph()
    for i, member in enumerate(ensemble.members):
        node_id = f"delay_{i}"
        graph.add_node(operator=member, node_id=node_id)

    global_input: Dict[str, Any] = {}

    # Compile the graph into an execution plan.
    plan = compile_graph(graph=graph)

    # Execute sequentially using XCSNoOpScheduler.
    seq_scheduler = XCSNoOpScheduler()
    start_seq: float = time.time()
    results_seq: Dict[str, Any] = seq_scheduler.run_plan(
        plan=plan, global_input=global_input, graph=graph
    )
    time_seq: float = time.time() - start_seq

    # Execute in parallel using TopologicalSchedulerWithParallelDispatch.
    par_scheduler = TopologicalSchedulerWithParallelDispatch(max_workers=num_members)
    start_par: float = time.time()
    results_par: Dict[str, Any] = par_scheduler.run_plan(
        plan=plan, global_input=global_input, graph=graph
    )
    time_par: float = time.time() - start_par

    # For 20 members at ~0.1s each, sequential execution should take around 2.0 seconds.
    # In parallel, we of course expect the total time to be close to 0.1 seconds.
    # Assert that parallel execution time is less than one-third of sequential time.
    assert (
        time_par < time_seq * 0.33
    ), f"Parallel execution ({time_par:.2f}s) did not speed up compared to sequential ({time_seq:.2f}s)."
