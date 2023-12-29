from src.mcts.mcts import MCTS
from src.statemanager.hexstatemanager import HexStateManager

import numpy as np


def setup_node():
    state_manager = HexStateManager(4, switch_rule_allowed=True)
    nn = None

    tree = MCTS(state_manager, nn)

    return tree


def test_expand():
    tree = setup_node()

    node = tree.root
    state_manager_expand = tree.state_manager.copy_state_manager()

    tree.expand_node(node, state_manager_expand)

    assert len(node.children) == 16


def test_empty_distribution():
    tree = setup_node()
    state_manager_expand = tree.state_manager.copy_state_manager()
    tree.expand_node(tree.root, state_manager_expand)

    distribution = tree.get_visit_distribution(tree.root)
    distribution = np.squeeze(distribution)

    assert len(distribution) == 16
    assert sum(distribution) == 0


def test_non_empty_distribution():
    tree = setup_node()
    state_manager_expand = tree.state_manager.copy_state_manager()
    tree.expand_node(tree.root, state_manager_expand)

    children_keys = list(tree.root.children.keys())
    children_keys.sort()
    
    tree.prune_tree(children_keys[3])
    state_manager_expand = tree.state_manager.copy_state_manager()
    tree.expand_node(tree.root, state_manager_expand)
    
    tree.prune_tree(children_keys[3])
    state_manager_expand = tree.state_manager.copy_state_manager()
    tree.expand_node(tree.root, state_manager_expand)

    for i, node in enumerate(tree.root.children.values()):
        node.n = i + 1

    distribution = tree.get_visit_distribution(tree.root)
    distribution = np.squeeze(distribution)

    assert distribution[3] == 0
    assert len(distribution) == 16
    # Within a certain tolerance, the sum should be 1
    assert np.isclose(sum(distribution), 1, atol=1e-08)
