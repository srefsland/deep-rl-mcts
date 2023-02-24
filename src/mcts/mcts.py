import numpy as np

from . import MCTSNode


class MCTS:
    def __init__(self, root_state, default_policy, c=1.0, max_iterations=1000):
        self.c = c
        self.default_policy = default_policy
        self.max_iterations = max_iterations
        self.root = MCTSNode(root_state, None)

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
        return child_node.get_qsa() + self.get_exploration_bonus(node, child_node)

    # Exploration term
    def get_exploration_bonus(self, node, child_node):
        return self.c * np.sqrt(np.log(node.get_n()) / child_node.get_nsa())

    def select_best_ucb(self, node):
        node_children = node.get_children()
        return node_children[np.argmax(self.get_ucb(node, node_children))]
