"""Unit tests for tree utility functions.

This module verifies that the tree_flatten and tree_unflatten functions correctly handle lists,
dictionaries, and nested structures by preserving the original tree layout through a round-trip
process.
"""

from typing import Any, Dict, List

from ember.xcs.utils.tree_util import tree_flatten, tree_unflatten


def test_tree_flatten_unflatten_list() -> None:
    """Tests round-trip processing of a simple list.

    This test verifies that flattening a list of integers followed by unflattening returns the
    original list.

    Returns:
        None.
    """
    tree: List[int] = [1, 2, 3]
    leaves, aux = tree_flatten(tree=tree)
    assert leaves == [1, 2, 3]
    reconstructed: List[int] = tree_unflatten(aux=aux, children=leaves)
    assert reconstructed == tree


def test_tree_flatten_unflatten_dict() -> None:
    """Tests round-trip processing of a dictionary.

    This test confirms that flattening a dictionary and then unflattening it reconstructs the
    original dictionary. Note that keys are processed in sorted order, yielding a consistent leaf order.

    Returns:
        None.
    """
    tree: Dict[str, int] = {"a": 1, "b": 2}
    leaves, aux = tree_flatten(tree=tree)
    # Dictionary keys are sorted; hence, the expected leaf order is [1, 2].
    assert leaves == [1, 2]
    reconstructed: Dict[str, int] = tree_unflatten(aux=aux, children=leaves)
    assert reconstructed == tree


def test_tree_flatten_unflatten_nested() -> None:
    """Tests round-trip processing of a nested structure.

    This test validates that a nested combination of lists, dictionaries, and tuples is correctly
    flattened into leaves and an auxiliary representation, and then reconstructed back to its original
    form. The expected order of leaves is [1, 2, 3, 4].

    Returns:
        None.
    """
    tree: Dict[str, Any] = {"list": [1, {"nested": 2}], "tuple": (3, 4)}
    leaves, aux = tree_flatten(tree=tree)
    # Expected leaves order: [1, 2, 3, 4].
    assert leaves == [1, 2, 3, 4]
    reconstructed: Dict[str, Any] = tree_unflatten(aux=aux, children=leaves)
    assert reconstructed == tree
