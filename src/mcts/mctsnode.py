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

    def get_visit_distribution(self):
        visit_distribution = np.zeros(
            (self.state.board_size, self.state.board_size))

        for child in self.children:
            visit_distribution[child.move[0],
                               child.move[1]] = child.nsa

        # Avoid division by zero
        if np.sum(visit_distribution) > 0:
            visit_distribution = visit_distribution / \
                np.sum(visit_distribution)

        visit_distribution = np.expand_dims(
            visit_distribution.flatten(), axis=0)
        return visit_distribution

    def get_winning_distribution(self, move):
        visit_distribution = np.zeros(
            (self.state.board_size, self.state.board_size))

        visit_distribution[move[0], move[1]] = 1

        visit_distribution = np.expand_dims(
            visit_distribution.flatten(), axis=0)
        return visit_distribution

    def is_root(self):
        return self.parent is None
