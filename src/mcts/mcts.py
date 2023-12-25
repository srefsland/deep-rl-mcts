import numpy as np

from .mctsnode import MCTSNode


class MCTS:
    def __init__(self, state_manager, c=1.0, use_critic=False):
        self.c = c
        self.state_manager = state_manager
        self.root = MCTSNode(state_manager.board, state_manager.player)
        self.use_critic = use_critic
        
    def simulation_iteration(self, actor):
        node, sim_state_manager = self.tree_search()
        reward = self.leaf_evaluation(node, sim_state_manager, actor)
        self.backpropagation(node, reward)

    def tree_search(self):
        """Traverses the tree and picks the best node based on the UCB value.

        Returns:
            MCTSNode: the leaf node chosen.
        """
        node = self.root
        sim_state_manager = self.state_manager.copy_state_manager()

        while not node.is_leaf_node():
            node = self.select_best_ucb(node)
            sim_state_manager.make_move(node.move)

        if not sim_state_manager.check_winning_state():
            self.expand_node(node, sim_state_manager)
        
        if node.children: 
            # Select a random child node
            node = np.random.choice(list(node.children.values()))
            sim_state_manager.make_move(node.move)

        return node, sim_state_manager

    def leaf_evaluation(self, node, sim_state_manager, actor):
        """This is the rollout function that evaluates the leaf node.

        Args:
            node (MCTSNode): the leaf node from which we simulate the game.
            epsilon (float): the epsilon value for the epsilon-greedy policy.

        Returns:
            int: the reward that the state manager calculates.
        """
        # Call critic
        if np.random.random() > actor.epsilon_critic and self.use_critic:
            reward = actor.predict_critic(node.state, node.player)
        else:
            # Perform rollout
            while not sim_state_manager.check_winning_state():
                # Epsilon-greedy policy
                move = actor.epsilon_greedy_policy(sim_state_manager.board, sim_state_manager.player)
                sim_state_manager.make_move(move)

            # Winner should be the one that took the last move (the one that is not the current player)
            winner = 1 if sim_state_manager.player == -1 else -1
            reward = sim_state_manager.get_eval(winner)

        return reward

    def backpropagation(self, node, reward):
        """Passes the reward back up the parent nodes.

        Args:
            node (MCTSNode): the leaf node from which we backpropagate.
            reward (int): the reward that is backpropagated.
        """
        while node is not None:
            node.update_values(reward)
            node = node.parent

    def expand_node(self, node, expand_state_manager):
        """Expands the node (finds the child states) if a sufficient number of visits are made."""   
        node.children = {
            move: MCTSNode(state=state, player=player, move=move, parent=node)
            for state, player, move in expand_state_manager.generate_child_states()
        }

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
            return child_node.get_qsa() + self.get_exploration_bonus(node, child_node)
        else:
            return child_node.get_qsa() - self.get_exploration_bonus(node, child_node)

    # Exploration term
    def get_exploration_bonus(self, node, child_node):
        """Gets the exploration bonus for the given node and child node.

        Args:
            node (MCTSNode): the node for which we calculate the exploration bonus.
            child_node (MCTSNode): the node for which we calculate the exploration bonus.

        Returns:
            float: the exploration bonus.
        """
        return self.c * np.sqrt(np.log(node.n) / (child_node.n + 1))

    def select_best_ucb(self, node):
        """Selects the best ucb value for the given node. The value is minimized or maximized
        depending on the player.

        Args:
            node (MCTSNode): the node for which we select the best ucb value.

        Returns:
            MCTSNode: the best child node.
        """
        keys, children = zip(*node.children.items())

        vectorized_get_ucb = np.vectorize(lambda child: self.get_ucb(node, child))
        ucb_values = vectorized_get_ucb(children)

        if node.player == 1:
            best_key = keys[np.argmax(ucb_values)]
            return node.children[best_key]
        else:
            best_key = keys[np.argmin(ucb_values)]
            return node.children[best_key]

    def select_best_distribution(self):
        """Selects the move with the highest action visit count.

        Returns:
            tuple[int, int]: the best child node.
        """
        node = self.root
        keys, children = zip(*node.children.items())

        get_n = np.vectorize(lambda child: child.n)
        best_key = keys[np.argmax(get_n(children))]

        return best_key
    
    def select_random_best_distribution(self):
        """Randomly selects one of the three most visited moves.

        Returns:
            _type_: _description_
        """
        node = self.root
        keys, children = zip(*node.children.items())
        
        get_n = np.vectorize(lambda child: child.n)
        visit_counts = get_n(children)
        top_three_indices = np.argsort(visit_counts)[-3:]
        top_three_counts = visit_counts[top_three_indices]
        probabilities = top_three_counts / np.sum(top_three_counts)
        
        selected_move_index = np.random.choice(top_three_indices, p=probabilities)
        selected_move = keys[selected_move_index]
        
        return selected_move

    def prune_tree(self, move):
        """Prunes the tree by setting the new node to be root and
        setting the parent of the new node to None.

        Args:
            node (MCTSNode): the new root node.
        """
        self.state_manager.make_move(move)
        self.root = self.root.children[move]
        self.root.parent = None
        
    def get_visit_distribution(self, node):
        """Gets the visit distribution of the children of the node in terms of nsa counts.

        Args:
            node (MCTSNode): the node associated with the current state.

        Returns:
            np.ndarray: the visit distribution.
        """
        visit_distribution = self.state_manager.get_distribution_shape()

        for move, child in node.children.items():
            visit_distribution[move] = child.n

        # Avoid division by zero
        if np.sum(visit_distribution) > 0:
            visit_distribution = visit_distribution / np.sum(visit_distribution)

        visit_distribution = np.expand_dims(visit_distribution.flatten(), axis=0)
        return visit_distribution

    def get_winning_distribution(self, winning_moves):
        """Gets the winning distribution if there are winning moves.

        Args:
            winning_moves (list(tuple[int, int])): the list of winning move(s).

        Returns:
            np.ndarray: the winning distribution.
        """
        visit_distribution = self.state_manager.get_distribution_shape()

        for move in winning_moves:
            visit_distribution[move] = 1

        # Normalize to 1
        visit_distribution = visit_distribution / np.sum(visit_distribution)

        visit_distribution = np.expand_dims(visit_distribution.flatten(), axis=0)
        return visit_distribution
