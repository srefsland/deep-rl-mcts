import numpy as np

from . import MCTSNode


class MCTS:
    def __init__(self, root_state, default_policy, c=1.0, verbose=False):
        self.c = c
        self.default_policy = default_policy
        self.root = MCTSNode(root_state, None)
        self.verbose = verbose

    def tree_search(self):
        node = self.root

        while not node.is_leaf():
            node = self.select_best_ucb(node)

        if not node.is_terminal() and node.n >= 10:
            node.expand()

            node = self.select_best_ucb(node)

        return node

    def leaf_evaluation(self, node, epsilon):
        node_state_copy = node.state.copy_state()
        board_size = node_state_copy.board_size

        # Perform rollout
        while not node_state_copy.check_winning_state():
            # Epsilon-greedy policy
            if np.random.random() < epsilon:
                node_state_copy.make_random_move()
            else:
                nn_input = node_state_copy.convert_to_nn_input()

                predictions = self.default_policy.predict(nn_input)

                move = np.argmax(predictions)
                move = (move // board_size, move % board_size)

                node_state_copy.make_move(move)

        # Winner should be the one that took the last move (the one that is not the current player)
        winner = (1, 0) if node_state_copy.player == (0, 1) else (0, 1)
        reward = node_state_copy.get_eval(winner)

        return reward

    def backpropagation(self, node, reward):
        while not node.is_root():
            node.update_values(reward)
            node = node.parent

    # Upper confidence bound that balances exploration (U(s,a)) and exploitation (Q(s,a))
    def get_ucb(self, node, child_node):
        # Player (1, 0) wants
        if node.state.player == (1, 0):
            return child_node.qsa + self.get_exploration_bonus(node, child_node)
        else:
            return child_node.qsa - self.get_exploration_bonus(node, child_node)

    # Exploration term
    def get_exploration_bonus(self, node, child_node):
        return self.c * np.sqrt(np.log(node.n) / child_node.nsa)

    def select_best_ucb(self, node):
        node_children = node.children

        if node.state.player == (1, 0):
            return node_children[np.argmax(self.get_ucb(node, node_children))]
        else:
            return node_children[np.argmin(self.get_ucb(node, node_children))]

    def prune_tree(self, node):
        self.root = node
        self.root.parent = None

    def get_visit_distribution(self):
        return self.root.get_visit_distribution()
