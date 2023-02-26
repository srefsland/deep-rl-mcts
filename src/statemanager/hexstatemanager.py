import copy

import numpy as np

from .hexboardcell import HexBoardCell
from .statemanager import StateManager

# Notes (for square board representation)
# Player 1 (red): top to bottom
# Player 2 (black): left to right


class HexStateManager(StateManager):
    def __init__(self, board_size=6, player=(1, 0), board=None):
        if board is None:
            self.board = self._initialize_state(board_size)
            self.board_size = board_size
        else:
            self.board = board
            self.board_size = len(board)

        # Whose turn it is
        self.player = player

    def _initialize_state(self, board_size):
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
                if self.is_move_legal(node.position):
                    moves.append(node.position)

        return moves

    def is_move_legal(self, move):
        return self.board[move[0]][move[1]].is_empty()

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

        if self._is_within_bounds(move[0], move[1]) and self.is_move_legal(move):
            self.board[move[0]][move[1]].occupant = player
        else:
            raise Exception("Illegal move")

        self.player = (0, 1) if player == (1, 0) else (1, 0)

        return move

    # Inspired by the article here: https://www.idi.ntnu.no/emner/it3105/materials/neural/gao-2017.pdf
    # Should make it possible to feed to convolutional neural network with 5 channels, 3 for occupancy
    # and 2 for each player's turn
    def convert_to_nn_input(self):
        nn_input = np.zeros(
            shape=(self.board_size, self.board_size, 5), dtype=np.int8)

        is_occupied_player_1 = np.vectorize(
            lambda x: 1 if x.occupant == (1, 0) else 0)
        is_occupied_player_2 = np.vectorize(
            lambda x: 1 if x.occupant == (0, 1) else 0)
        is_occupied_empty = np.vectorize(
            lambda x: 1 if x.is_empty() else 0)

        nn_input[:, :, 0] = is_occupied_player_1(self.board)
        nn_input[:, :, 1] = is_occupied_player_2(self.board)
        nn_input[:, :, 2] = is_occupied_empty(self.board)
        nn_input[:, :, 3] = 1 if self.player == (1, 0) else 0
        nn_input[:, :, 4] = 1 if self.player == (0, 1) else 0

        return nn_input

    def convert_to_diamond_shape(self):
        diamond_array = []
        for i in range(-self.board_size + 1, self.board_size):
            diamond_array.append(np.diagonal(
                np.flipud(self.board), i).tolist())

        return diamond_array

    def _is_within_bounds(self, row, col):
        return row >= 0 and row < self.board_size and col >= 0 and col < self.board_size

    def _expand_neighbors(self, node, player=None):
        if player is None:
            player = self.player

        row, col = node.position

        neighbors_coords = [(row - 1, col), (row + 1, col), (row, col - 1),
                            (row, col + 1), (row + 1, col - 1), (row - 1, col + 1)]

        neighbors = []

        for neighbor in neighbors_coords:
            if self._is_within_bounds(neighbor[0], neighbor[1]) and self.board[neighbor[0]][neighbor[1]].occupant == player:
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

        # Just makes it easier to keep track of what move was made to get to the child state
        return zip(child_states, moves)

    def check_winning_state(self, player=None):
        if player is None:
            player = self.player

        if player == (1, 0):
            return self._check_winning_state_player1()
        elif player == (0, 1):
            return self._check_winning_state_player2()
        else:
            return self._check_winning_state_player1() or self._check_winning_state_player2()

    # Player 1 (red) is top to bottom
    def _check_winning_state_player1(self):
        nodes_to_visit = []
        nodes_visited = []

        for col in self.board[0]:
            if col.occupant == (1, 0):
                nodes_to_visit.append(col)

        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop(0)
            nodes_visited.append(node)

            if node.position[0] == self.board_size - 1:
                return True

            neighbors = self._expand_neighbors(node, (1, 0))

            for neighbor in neighbors:
                if neighbor not in nodes_to_visit and neighbor not in nodes_visited:
                    nodes_to_visit.append(neighbor)

        return False

    # Player 2 (black) is left to right
    def _check_winning_state_player2(self):
        nodes_to_visit = []
        nodes_visited = []

        for row in self.board:
            if row[0].occupant == (0, 1):
                nodes_to_visit.append(row[0])

        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop(0)
            nodes_visited.append(node)

            if node.position[1] == self.board_size - 1:
                return True

            neighbors = self._expand_neighbors(node, (0, 1))

            for neighbor in neighbors:
                if neighbor not in nodes_to_visit and neighbor not in nodes_visited:
                    nodes_to_visit.append(neighbor)

        return False

    def get_eval(self, winner=(1, 0)):
        return 1 if winner == (1, 0) else -1 if winner == (0, 1) else 0
