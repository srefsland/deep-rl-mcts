import numpy as np


class MCTSNode:
    def __init__(self, state, parent=None):
        self.parent = parent
        self.state = state
        self.children = None
        self.qsa = 0
        self.nsa = 0
        self.n = 0

    def expand(self):
        self.children = np.array([MCTSNode(child_state, self)
                                 for child_state in self.state.generate_child_states(self.player)])

    def update_values(self, reward):
        self.n += 1
        self.nsa += 1
        # The closer the Q(s, a) value to the reward, the less the Q(s, a) value will change
        self.qsa += (reward - self.qsa) / self.nsa

    def is_leaf_node(self):
        return self.children is None

    def is_terminal(self):
        return self.state.check_winning_state()
