from src.mcts.mctsnode import MCTSNode
from src.statemanager.hexstatemanager import HexStateManager

import numpy as np


def setup_node():
    state = HexStateManager(4)
    setup_node = MCTSNode(state=state, move=None, parent=None)

    return setup_node


def test_expand():
    node = setup_node()

    node.expand()

    assert len(node.children) == 16


def test_empty_distribution():
    node = setup_node()
    node.expand()

    distribution = node.get_visit_distribution()
    distribution = np.squeeze(distribution)

    assert len(distribution) == 16
    assert sum(distribution) == 0


def test_non_empty_distribution():
    root_node = setup_node()
    root_node.state.make_move((0, 3), (1, 0))
    root_node.expand()

    for i, node in enumerate(root_node.children):
        node.nsa = i

    distribution = root_node.get_visit_distribution()
    distribution = np.squeeze(distribution)

    assert distribution[3] == 0
    assert len(distribution) == 16
    assert sum(distribution) == 1
