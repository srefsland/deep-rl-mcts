import numpy as np


class MCTSNode:
    def __init__(self, state, move=None, parent=None):
        self.parent = parent
        self.state = state
        self.children = None
        self.qsa = 0
        self.nsa = 0
        self.n = 0
        # The move that led to this node
        self.move = move

    def expand(self):
        """Expands the node (finds the child states) if a sufficient number of visits are made.

        Returns:
            bool: True if the node was expanded, False otherwise.
        """
        if self.n > 10:
            self.children = np.array([MCTSNode(state=child_state, move=move, parent=self)
                                      for child_state, move in self.state.generate_child_states()])

            return True
        else:
            return False

    def update_values(self, reward):
        """Updates the values that are backpropagated.

        Args:
            reward (int): the reward that is backpropagated.
        """
        self.n += 1
        self.nsa += 1
        # The closer the Q(s, a) value to the reward, the less the Q(s, a) value will change
        self.qsa += (reward - self.qsa) / self.nsa

    def is_leaf_node(self):
        """Checks if the node is a leaf node (no children).

        Returns:
            bool: if the node is a leaf node.
        """
        return self.children is None

    def is_terminal(self):
        """Checks if the node is a terminal node (game is over).

        Returns:
            bool: True if the node is a terminal node, False otherwise.
        """
        return self.state.check_winning_state()

    def is_root(self):
        """Checks if the node is root (no parent).

        Returns:
            bool: True if the node is root, False otherwise.
        """
        return self.parent is None
