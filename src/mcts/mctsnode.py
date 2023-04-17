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
        if self.n > 10:
            self.children = np.array([MCTSNode(state=child_state, move=move, parent=self)
                                      for child_state, move in self.state.generate_child_states()])

            return True
        else:
            return False

    def update_values(self, reward):
        self.n += 1
        self.nsa += 1
        # The closer the Q(s, a) value to the reward, the less the Q(s, a) value will change
        self.qsa += (reward - self.qsa) / self.nsa

    def is_leaf_node(self):
        return self.children is None

    def is_terminal(self):
        return self.state.check_winning_state()

    def is_root(self):
        return self.parent is None
