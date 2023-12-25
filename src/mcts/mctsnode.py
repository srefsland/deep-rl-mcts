class MCTSNode:
    def __init__(self, state, player, move=None, parent=None):
        self.parent = parent
        self.state = state
        self.player = player
        self.children = None
        self.e = 0
        self.n = 0
        # The move that led to this node
        self.move = move

    def update_values(self, reward):
        """Updates the values that are backpropagated.

        Args:
            reward (int): the reward that is backpropagated.
        """
        self.n += 1
        self.e += reward

    def is_leaf_node(self):
        """Checks if the node is a leaf node (no children).

        Returns:
            bool: if the node is a leaf node.
        """
        return self.children is None

    def is_root(self):
        """Checks if the node is root (no parent).

        Returns:
            bool: True if the node is root, False otherwise.
        """
        return self.parent is None
    
    def get_qsa(self):
        """Gets the Q(s, a) value for the node.

        Returns:
            float: the Q(s, a) value.
        """
        return self.e / self.n if self.n > 0 else 0
