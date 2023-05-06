import numpy as np

from .mctsnode import MCTSNode


class MCTS:
    def __init__(self, state_manager, default_policy, c=1.0):
        self.c = c
        self.default_policy = default_policy
        self.state_manager = state_manager
        self.root = MCTSNode(state_manager.board, state_manager.player)

    def tree_search(self):
        """Traverses the tree and picks the best node based on the UCB value.

        Returns:
            MCTSNode: the leaf node chosen.
        """
        node = self.root

        while not node.is_leaf_node():
            node = self.select_best_ucb(node)

        if not self.is_terminal(node):
            self.expand_node(node)

            node = self.select_best_ucb(node)

        return node

    def leaf_evaluation(self, node, epsilon, epsilon_critic):
        """This is the rollout function that evaluates the leaf node.

        Args:
            node (MCTSNode): the leaf node from which we simulate the game.
            epsilon (float): the epsilon value for the epsilon-greedy policy.

        Returns:
            int: the reward that the state manager calculates.
        """
        self.state_manager.update_state(node.state, node.player)

        # Call critic
        if np.random.random() > epsilon_critic:
            reward = self.default_policy.call_critic(node.state, node.player)
        else:
            # Perform rollout
            while not self.state_manager.check_winning_state():
                # Epsilon-greedy policy
                if np.random.random() < epsilon:
                    self.state_manager.make_random_move()
                else:
                    move = self.default_policy.predict_best_move(
                        self.state_manager.board, self.state_manager.player
                    )
                    self.state_manager.make_move(move)

            # Winner should be the one that took the last move (the one that is not the current player)
            winner = 1 if self.state_manager.player == -1 else -1
            reward = self.state_manager.get_eval(winner)

        self.state_manager.update_state(self.root.state, self.root.player)

        return reward

    def backpropagation(self, node, reward):
        """Passes the reward back up the parent nodes.

        Args:
            node (MCTSNode): the leaf node from which we backpropagate.
            reward (int): the reward that is backpropagated.
        """
        while not node == None:
            node.update_values(reward)
            node = node.parent

    def expand_node(self, node):
        """Expands the node (finds the child states) if a sufficient number of visits are made."""
        self.state_manager.update_state(node.state, node.player)
        node.children = np.array(
            [
                MCTSNode(state=child_state, player=player, move=move, parent=node)
                for child_state, player, move in self.state_manager.generate_child_states()
            ]
        )

        self.state_manager.update_state(self.root.state, self.root.player)

    # Upper confidence bound that balances exploration (U(s,a)) and exploitation (Q(s,a))
    def get_ucb(self, node, child_node):
        """Calculates the upper confidence bound for the given node and child node.

        Args:
            node (MCTSNode): the node for which we calculate the UCB.
            child_node (MCTSNode): the child node for which we calculate the UCB.

        Returns:
            _type_: _description_
        """
        # Player 1 wants to maximize the value, player 2 wants to minimize the value
        if node.player == 1:
            return child_node.q + self.get_exploration_bonus(node, child_node)
        else:
            return child_node.q - self.get_exploration_bonus(node, child_node)

    # Exploration term
    def get_exploration_bonus(self, node, child_node):
        """Gets the exploration bonus for the given node and child node.

        Args:
            node (MCTSNode): the node for which we calculate the exploration bonus.
            child_node (MCTSNode): the node for which we calculate the exploration bonus.

        Returns:
            float: the exploration bonus.
        """
        return self.c * np.sqrt(np.log(node.n + 1) / (child_node.n + 1))

    def select_best_ucb(self, node):
        """Selects the best ucb value for the given node. The value is minimized or maximized
        depending on the player.

        Args:
            node (MCTSNode): the node for which we select the best ucb value.

        Returns:
            MCTSNode: the best child node.
        """
        node_children = node.children

        vectorized_get_ucb = np.vectorize(lambda child: self.get_ucb(node, child))
        ucb_values = vectorized_get_ucb(node_children)

        if node.player == 1:
            return node_children[np.argmax(ucb_values)]
        else:
            return node_children[np.argmin(ucb_values)]

    def select_best_distribution(self):
        """Selects the node with the highest action visit count.

        Returns:
            MCTSNode: the best child node.
        """
        node = self.root
        node_children = node.children

        get_n = np.vectorize(lambda child: child.n)

        return node_children[np.argmax(get_n(node_children))]

    def select_winning_move(self, winning_move):
        """This disregards the visit count and picks a node that is in a winning state.

        Args:
            winning_move: the winning move.

        Returns:
            MCTSNode: the winning child node.
        """
        node = self.root
        node_children = node.children

        has_move = np.vectorize(lambda child: child.move == winning_move)
        # Get the node child that has the winning move, childs have node attribute
        winning_child = node_children[has_move(node_children)][0]

        return winning_child

    def prune_tree(self, node):
        """Prunes the tree by setting the new node to be root and
        setting the parent of the new node to None.

        Args:
            node (MCTSNode): the new root node.
        """
        self.root = node
        self.root.parent = None

    def is_terminal(self, node):
        """Checks if the node is a terminal node (game is over).

        Returns:
            bool: True if the node is a terminal node, False otherwise.
        """
        self.state_manager.update_state(node.state, node.player)

        is_terminal = self.state_manager.check_winning_state()

        self.state_manager.update_state(self.root.state, self.root.player)

        return is_terminal
