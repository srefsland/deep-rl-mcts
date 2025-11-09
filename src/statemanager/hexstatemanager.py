import pickle

import numpy as np
from disjoint_set import DisjointSet

from .exceptions import IllegalMoveException

from .statemanager import StateManager


class MoveEntry:
    def __init__(self, move, player_moved, switch_move_type):
        self.move = move
        self.player_moved = player_moved
        self.switch_move_type = switch_move_type


class HexStateManager(StateManager):
    def __init__(self, board_size=6):
        self.board_size = board_size
        self._initialize_state(board_size)

    def copy_state_manager(self):
        return pickle.loads(pickle.dumps(self))

    # NOTE: only passes player as parameter to be able to generalize for all types of state manager in 2v2 board games.
    # In Hex, the available moves are the same for both players.
    def get_legal_moves(self, player_to_move=None):
        return self.legal_moves

    def make_move(self, move, player_to_move=None):
        if player_to_move is None:
            player_to_move = self.player_to_move

        if move not in self.legal_moves:
            raise IllegalMoveException()

        is_first_move = self.first_move is None
        is_second_move = len(self.move_history) == 1

        if is_first_move:
            self.first_move = move
            self.board[move] = player_to_move
        elif is_second_move:
            if move == self.first_move:
                self.switched = True

                mirrored_move = (move[1], move[0])
                
                if mirrored_move != move:
                    self.board[move] = 0
                    self.board[mirrored_move] = player_to_move
                else:
                    self.board[move] = player_to_move

                self.legal_moves.remove(mirrored_move)
                self._union_move(mirrored_move, player_to_move)
                
                move = mirrored_move
            else:
                self.board[move] = player_to_move
                self.legal_moves.remove(move)
                self.legal_moves.remove(self.first_move)

                self._union_move(move, player_to_move)
                self._union_move(self.first_move, 1)
        else:
            self.board[move] = player_to_move
            self.legal_moves.remove(move)
            self._union_move(move, player_to_move)

        self.move_history.append(
            MoveEntry(
                move, player_to_move, self.switched and len(self.move_history) == 1
            )
        )
        self._change_player_to_move(player_to_move)

        return move

    def make_random_move(self, player_to_move=None):
        if player_to_move is None:
            player_to_move = self.player_to_move

        moves = self.get_legal_moves()

        if len(moves) == 0:
            return

        move = self.make_move(
            list(moves)[np.random.randint(0, len(moves))], player_to_move
        )

        return move

    def generate_child_states(self, player_to_move=None):
        if player_to_move is None:
            player_to_move = self.player_to_move

        for move in self.get_legal_moves():
            state_manager = self.copy_state_manager()
            state_manager.make_move(move, player_to_move)

            yield state_manager.board, state_manager.player_to_move, move

    def check_winning_state(self, player_moved=None):
        if player_moved == 1:
            return self._check_winning_state_player1()
        elif player_moved == -1:
            return self._check_winning_state_player2()
        else:
            return (
                self._check_winning_state_player1()
                or self._check_winning_state_player2()
            )

    def reset(self):
        self._initialize_state(self.board_size)

    def get_eval(self, winner=1):
        return winner if not self.switched else -winner

    def get_board_shape(self):
        return np.zeros((self.board_size, self.board_size))

    def _initialize_state(self, board_size):
        self.board = np.zeros((board_size, board_size))
        self.switched = False
        self.legal_moves = set(
            [(i, j) for j in range(board_size) for i in range(board_size)]
        )
        self.first_move = None
        self.move_history = []
        self.player_to_move = 1

        self.top_node = (-1, 0)
        self.bottom_node = (board_size, 0)
        self.left_node = (0, -1)
        self.right_node = (0, board_size)

        cells = [(i, j) for j in range(board_size) for i in range(board_size)]
        self.disjoint_set_red = DisjointSet(cells + [self.top_node, self.bottom_node])
        self.disjoint_set_blue = DisjointSet(cells + [self.left_node, self.right_node])

        for i in range(board_size):
            self.disjoint_set_red.union((0, i), self.top_node)
            self.disjoint_set_red.union((board_size - 1, i), self.bottom_node)
            self.disjoint_set_blue.union((i, 0), self.left_node)
            self.disjoint_set_blue.union((i, board_size - 1), self.right_node)

    def _check_winning_state_player1(self):
        return self.disjoint_set_red.find(self.top_node) == self.disjoint_set_red.find(
            self.bottom_node
        )

    def _check_winning_state_player2(self):
        return self.disjoint_set_blue.find(
            self.left_node
        ) == self.disjoint_set_blue.find(self.right_node)

    def _union_move(self, move, player):
        neighbors = self._expand_neighbors(move, player)
        if player == 1:
            for neighbor in neighbors:
                self.disjoint_set_red.union(move, neighbor)
        else:
            for neighbor in neighbors:
                self.disjoint_set_blue.union(move, neighbor)

    def _expand_neighbors(self, cell, player_to_move=None):
        if player_to_move is None:
            player_to_move = self.player_to_move

        row, col = cell

        neighbors_coords = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
            (row + 1, col - 1),
            (row - 1, col + 1),
        ]

        neighbors = []

        for neighbor in neighbors_coords:
            if (
                self._is_within_bounds(neighbor)
                and self.board[neighbor] == player_to_move
            ):
                neighbors.append(neighbor)

        return neighbors

    def _is_within_bounds(self, cell):
        row, col = cell
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def _change_player_to_move(self, player_to_move):
        self.player_to_move = -player_to_move
