import numpy as np
import copy
from .hexboardcell import HexBoardCell
from .statemanager import StateManager

# Notes (for square board representation)
# Player 1 (red): top to bottom
# Player 2 (black): left to right


class HexStateManager(StateManager):
    def __init__(self, board_size=6, player=(1, 0), board=None):
        if board is None:
            self.board = self.initialize_state(board_size)
            self.board_size = board_size
        else:
            self.board = board
            self.board_size = len(board)

        # Whose turn it is
        self.player = player

    def initialize_state(self, board_size):
        board = np.array([[HexBoardCell(row, col) for col in range(board_size)]
                         for row in range(board_size)])

        return board

    def copy_state(self):
        return HexStateManager(player=self.player, board=copy.deepcopy(self.board))

    # NOTE: only passes player as parameter to be able to generalize for all types of state manager in 2v2 board games.
    # In Hex, the available moves are the same for both players.
    def get_moves_legal(self, player=None):
        moves = []

        for row in self.board:
            for node in row:
                if self.is_move_legal(node.get_position()):
                    moves.append(node.get_position())

        return moves

    def is_move_legal(self, move):
        return self.board[move[0]][move[1]].get_owner() == (0, 0)

    def make_random_move(self, player=None):
        if player is None:
            player = self.player

        moves = self.get_moves_legal()

        if len(moves) == 0:
            return

        move = self.make_move(moves[np.random.randint(0, len(moves))], player)

        return move

    def make_move(self, move, player=None):
        if player is None:
            player = self.player

        if self.is_within_bounds(move[0], move[1]) and self.is_move_legal(move):
            self.board[move[0]][move[1]].set_owner(player)
        else:
            raise Exception("Illegal move")

        self.player = (0, 1) if player == (1, 0) else (1, 0)

        return move

    def convert_to_1D_array(self):
        return self.board.flatten()

    def convert_to_diamond_shape(self):
        diamond_array = []
        for i in range(-self.board_size + 1, self.board_size):
            diamond_array.append(np.diagonal(
                np.flipud(self.board), i).tolist())

        return diamond_array

    def is_within_bounds(self, row, col):
        return row >= 0 and row < self.board_size and col >= 0 and col < self.board_size

    def expand_neighbors(self, node, player=None):
        if player is None:
            player = self.player

        row, col = node.get_position()

        neighbors_coords = [(row - 1, col), (row + 1, col), (row, col - 1),
                            (row, col + 1), (row + 1, col - 1), (row - 1, col + 1)]

        neighbors = []

        for neighbor in neighbors_coords:
            if self.is_within_bounds(neighbor[0], neighbor[1]) and self.board[neighbor[0]][neighbor[1]].get_owner() == player:
                    neighbors.append(self.board[neighbor[0]][neighbor[1]])

        return neighbors

    def generate_child_states(self, player=None):
        if player is None:
            player = self.player

        child_states = []
        moves = self.get_moves_legal()

        for move in moves:
            child_board = self.copy_state()
            child_board.make_move(move, player)
            child_states.append(child_board)

        return child_states

    def check_winning_state(self, player=None):
        if player is None:
            player = self.player

        if player == (1, 0):
            return self.check_winning_state_player1()
        elif player == (0, 1):
            return self.check_winning_state_player2()
        else:
            return self.check_winning_state_player1() or self.check_winning_state_player2()

    # Player 1 (red) is top to bottom
    def check_winning_state_player1(self):
        nodes_to_visit = []
        nodes_visited = []

        for col in self.board[0]:
            if col.get_owner() == (1, 0):
                nodes_to_visit.append(col)

        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop(0)
            nodes_visited.append(node)

            if node.get_position()[0] == self.board_size - 1:
                return True

            neighbors = self.expand_neighbors(node, (1, 0))

            for neighbor in neighbors:
                if neighbor not in nodes_to_visit and neighbor not in nodes_visited:
                    nodes_to_visit.append(neighbor)

        return False

    # Player 2 (black) is left to right
    def check_winning_state_player2(self):
        nodes_to_visit = []
        nodes_visited = []

        for row in self.board:
            if row[0].get_owner() == (0, 1):
                nodes_to_visit.append(row[0])

        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop(0)
            nodes_visited.append(node)

            if node.get_position()[1] == self.board_size - 1:
                return True

            neighbors = self.expand_neighbors(node, (0, 1))

            for neighbor in neighbors:
                if neighbor not in nodes_to_visit and neighbor not in nodes_visited:
                    nodes_to_visit.append(neighbor)

        return False

    def find_immediate_winning_move(self, player=None):
        if player is None:
            player = self.player
        # Check that at least one edge is populated
        if not self._has_one_edge_populated(player):
            return None

        for move in self.get_moves_legal():
            child_state = self.copy_state()
            child_state.make_move(move, player)

            if child_state.check_winning_state(player):
                return move

        return None

    def _has_one_edge_populated(self, player):
        if player == (1, 0):
            return True if len([col for col in np.concatenate([self.board[0], self.board[self.board_size - 1]]) if col.get_owner() == (1, 0)]) > 0 else False
        elif player == (0, 1):
            return True if len([row for row in np.concatenate([self.board[:, 0], self.board[:, self.board_size - 1]]) if row.get_owner() == (0, 1)]) > 0 else False
        else:
            return False
