import numpy as np

from .mctsnode import MCTSNode


class MCTS:
    def __init__(self, root_state, default_policy, c=1.0):
        self.c = c
        self.default_policy = default_policy
        self.root = MCTSNode(root_state, None)
        
    def tree_search(self):
        node = self.root

        while not node.is_leaf_node():
            node = self.select_best_ucb(node)

        if not node.is_terminal():
            expand = node.expand()

            if expand:
                node = self.select_best_ucb(node)

        return node

    def leaf_evaluation(self, node, epsilon):
        node_state_copy = node.state.copy_state()

        # Perform rollout
        while not node_state_copy.check_winning_state():
            # Epsilon-greedy policy
            if np.random.random() < epsilon:
                node_state_copy.make_random_move()
            else:
                move = self.default_policy.predict_best_move(node_state_copy)
                node_state_copy.make_move(move)

        # Winner should be the one that took the last move (the one that is not the current player)
        winner = (1, 0) if node_state_copy.player == (0, 1) else (0, 1)
        reward = node_state_copy.get_eval(winner)

        return reward

    def backpropagation(self, node, reward):
        while not node == None:
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
        return self.c * np.sqrt(np.log(node.n + 1) / (child_node.nsa + 1))

    def select_best_ucb(self, node):
        node_children = node.children

        vectorized_get_ucb = np.vectorize(
            lambda child: self.get_ucb(node, child))
        ucb_values = vectorized_get_ucb(node_children)

        if node.state.player == (1, 0):
            return node_children[np.argmax(ucb_values)]
        else:
            return node_children[np.argmin(ucb_values)]

    def select_best_distribution(self):
        node = self.root
        node_children = node.children

        get_nsa = np.vectorize(lambda child: child.nsa)

        return node_children[np.argmax(get_nsa(node_children))]

    def select_winning_move(self, winning_move):
        node = self.root
        node_children = node.children

        has_move = np.vectorize(lambda child: child.move == winning_move)
        # Get the node child that has the winning move, childs have node attribute
        winning_child = node_children[has_move(node_children)][0]

        return winning_child

    def prune_tree(self, node):
        self.root = node
        self.root.parent = None
