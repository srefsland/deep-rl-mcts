from . import MCTSNode
import numpy as np


class MCTS:
    def __init__(self, root_state, nn, c=1.0, max_iterations=1000):
        self.c = c
        self.max_iterations = max_iterations
        self.root = MCTSNode(root_state, None, None)

    def tree_search(self):
        node = self.root

        while not node.is_leaf():
            node = self.select_best_ucb(node)

        if not node.is_terminal() and node.get_n() >= 10:
            node.expand()

    def leaf_evaluation(self, node):
        pass

    def backpropagation(self, node, reward):
        pass

    # Upper confidence bound that balances exploration (U(s,a)) and exploitation (Q(s,a))
    def get_ucb(self, node, child_node):
        return node.get_qsa(child_node) + self.get_exploration_bonus(node, child_node)

    # Exploration term
    def get_exploration_bonus(self, node, child_node):
        return self.c * np.sqrt(np.log(node.get_n()) / node.get_nsa(child_node))

    def select_best_ucb(self, node):
        return node.get_children()[np.argmax(self.get_ucb(node, node.get_children()))]
