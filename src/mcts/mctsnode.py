import numpy as np


class MTCSNode:
    def __init__(self, state, parent=None):
        self.parent = parent
        self.state = state
        self.children = []
        self._qsa = {}
        self._nsa = {}
        self.n = 0

    def expand(self):
        for child_state in self.state.generate_child_states(self.player):
            self.children.append(
                MTCSNode(child_state, self))
            self.qsa[child_state] = 0
            self.nsa[child_state] = 0

        self.children = np.array(self.children)

    def get_qsa(self, state):
        return self.qsa[state]

    def get_nsa(self, state):
        return self.nsa[state]

    def get_n(self):
        return self.n

    def update_values(self, state, reward):
        self.n += 1
        self.nsa[state] += 1
        # The closer the Q(s, a) value to the reward, the less the Q(s, a) value will change
        self.qsa[state] += (reward - self.qsa[state]) / self.nsa[state]

    def is_leaf_node(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.state.check_winning_state()
