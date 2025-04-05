"""Unit tests for the autograph module.

This module contains tests for the AutoGraphBuilder class, which is responsible
for automatically building XCS graphs from execution traces.
"""

import unittest
from typing import Any, Dict, List

from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.tracer.autograph import AutoGraphBuilder
from ember.xcs.tracer.xcs_tracing import TraceRecord


class TestAutoGraphBuilder(unittest.TestCase):
    """Tests for the AutoGraphBuilder class."""

    def test_build_simple_graph(self) -> None:
        """Test building a simple graph with sequential dependencies."""
        # Create a series of trace records that form a linear chain
        records = [
            TraceRecord(
                operator_name="Op1",
                node_id="1",
                inputs={"query": "test"},
                outputs={"result": "output1"},
            ),
            TraceRecord(
                operator_name="Op2",
                node_id="2",
                inputs={"result": "output1"},
                outputs={"intermediate": "output2"},
            ),
            TraceRecord(
                operator_name="Op3",
                node_id="3",
                inputs={"intermediate": "output2"},
                outputs={"final": "output3"},
            ),
        ]

        # Build the graph
        builder = AutoGraphBuilder()
        graph = builder.build_graph(records)

        # Verify the graph structure
        self.assertEqual(len(graph.nodes), 3)
        self.assertIn("Op1_0", graph.nodes)
        self.assertIn("Op2_1", graph.nodes)
        self.assertIn("Op3_2", graph.nodes)

        # Check dependencies - Op2 should depend on Op1, Op3 should depend on Op2
        self.assertIn("Op1_0", graph.nodes["Op2_1"].inbound_edges)
        self.assertIn("Op2_1", graph.nodes["Op3_2"].inbound_edges)

    def test_build_graph_with_branches(self) -> None:
        """Test building a graph with branching dependencies."""
        # Create trace records that form a diamond pattern:
        # Op1 -> Op2a -> Op3
        #   \-> Op2b -/
        records = [
            TraceRecord(
                operator_name="Op1",
                node_id="1",
                inputs={"query": "test"},
                outputs={"result": "output1"},
            ),
            TraceRecord(
                operator_name="Op2a",
                node_id="2a",
                inputs={"result": "output1"},
                outputs={"branch_a": "output2a"},
            ),
            TraceRecord(
                operator_name="Op2b",
                node_id="2b",
                inputs={"result": "output1"},
                outputs={"branch_b": "output2b"},
            ),
            TraceRecord(
                operator_name="Op3",
                node_id="3",
                inputs={"branch_a": "output2a", "branch_b": "output2b"},
                outputs={"final": "output3"},
            ),
        ]

        # Build the graph
        builder = AutoGraphBuilder()
        graph = builder.build_graph(records)

        # Verify the graph structure
        self.assertEqual(len(graph.nodes), 4)

        # Check dependencies
        self.assertIn("Op1_0", graph.nodes["Op2a_1"].inbound_edges)
        self.assertIn("Op1_0", graph.nodes["Op2b_2"].inbound_edges)
        self.assertIn("Op2a_1", graph.nodes["Op3_3"].inbound_edges)
        self.assertIn("Op2b_2", graph.nodes["Op3_3"].inbound_edges)

    def test_dependency_detection(self) -> None:
        """Test the dependency detection logic."""
        builder = AutoGraphBuilder()

        # Test dictionary value dependency
        inputs = {"x": 1, "y": "output_from_previous"}
        outputs = {"result": "output_from_previous"}
        self.assertTrue(builder._has_dependency(inputs=inputs, outputs=outputs))

        # Test no dependency
        inputs = {"x": 1, "y": "no_match"}
        outputs = {"result": "output_from_previous"}
        self.assertFalse(builder._has_dependency(inputs=inputs, outputs=outputs))

        # Test direct value dependency
        inputs = {"x": 1, "y": "output_value"}
        outputs = "output_value"
        self.assertTrue(builder._has_dependency(inputs=inputs, outputs=outputs))


if __name__ == "__main__":
    unittest.main()
