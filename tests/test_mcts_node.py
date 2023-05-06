from src.mcts.mctsnode import MCTSNode
from src.mcts.mcts import MCTS
from src.nn.boardgamenetann import BoardGameNetANN
from src.statemanager.hexstatemanager import HexStateManager

from src import config

import numpy as np


def setup_node():
    state = HexStateManager(4)
    nn = None

    tree = MCTS(state, nn)

    return tree


def test_expand():
    tree = setup_node()

    node = tree.root

    tree.expand_node(node)

    assert len(node.children) == 16


def test_empty_distribution():
    tree = setup_node()
    tree.expand_node(tree.root)

    distribution = tree.state_manager.get_visit_distribution(tree.root)
    distribution = np.squeeze(distribution)

    assert len(distribution) == 16
    assert sum(distribution) == 0


def test_non_empty_distribution():
    tree = setup_node()
    tree.expand_node(tree.root)

    tree.prune_tree(tree.root.children[3])
    tree.expand_node(tree.root)

    for i, node in enumerate(tree.root.children):
        node.n = i

    distribution = tree.state_manager.get_visit_distribution(tree.root)
    distribution = np.squeeze(distribution)

    assert distribution[3] == 0
    assert len(distribution) == 16
    assert sum(distribution) == 1
