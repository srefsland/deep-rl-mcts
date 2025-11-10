import pickle

import numpy as np

from .exceptions import IllegalMoveException
from .statemanager import StateManager


class HexStateManager(StateManager):
    def __init__(self, board_size=6, initialize=True):
        self.board_size = board_size
        if initialize:
            self._initialize_state(board_size)

    def copy_state_manager2(self):
        return pickle.loads(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    def copy_state_manager(self):
        copied_state_manager = HexStateManager(self.board_size, initialize=False)
        copied_state_manager.board = self.board.copy()

        copied_state_manager.switched = self.switched
        copied_state_manager.legal_moves = set(self.legal_moves)
        copied_state_manager.first_move = self.first_move
        copied_state_manager.move_count = self.move_count
        copied_state_manager.player_to_move = self.player_to_move

        return copied_state_manager

    # NOTE: only passes player as parameter to be able to generalize for all types of state manager in 2v2 board games.
    # In Hex, the available moves are the same for both players.
    def get_legal_moves(self, player_to_move=None):
        return self.legal_moves

    def make_move(self, move, player_to_move=None):
        if player_to_move is None:
            player_to_move = self.player_to_move

        if move not in self.legal_moves:
            raise IllegalMoveException()

        is_first_move = self.move_count == 0
        is_second_move = self.move_count == 1

        if not is_first_move and not is_second_move:
            self.board[move] = player_to_move
            self.legal_moves.remove(move)
        elif is_first_move:
            self.first_move = move
            self.board[move] = player_to_move
        else:
            if move == self.first_move:
                self.switched = True

                mirrored_move = (move[1], move[0])

                if mirrored_move != move:
                    self.board[move] = 0
                    self.board[mirrored_move] = player_to_move
                else:
                    self.board[move] = player_to_move

                self.legal_moves.remove(mirrored_move)

                move = mirrored_move
            else:
                self.board[move] = player_to_move
                self.legal_moves.remove(move)
                self.legal_moves.remove(self.first_move)

        self.move_count += 1
        self._change_player_to_move(player_to_move)

        return move

    def undo_move(self, move, player_to_move=None):
        if player_to_move is None:
            player_to_move = -self.player_to_move  # The player who made the move

        self.board[move] = 0
        self.legal_moves.add(move)
        self.move_count -= 1
        self._change_player_to_move(player_to_move)

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
            child_state_manager = self.copy_state_manager()
            
            child_state_manager.make_move(move, player_to_move)
            yield child_state_manager.board, child_state_manager.player_to_move, move

    def generate_child_states2(self, player_to_move=None):
        if player_to_move is None:
            player_to_move = self.player_to_move

        move_count = self.move_count
        first_move = self.first_move

        for move in self.get_legal_moves():
            child_board = self.board.copy()

            if move_count == 1:
                if move == first_move:
                    mirrored_move = (move[1], move[0])

                    if mirrored_move != move:
                        child_board[move] = 0
                        child_board[mirrored_move] = player_to_move
                    else:
                        child_board[move] = player_to_move
                else:
                    child_board[move] = player_to_move
            else:
                child_board[move] = player_to_move

            player_to_move_next = -player_to_move

            yield child_board, player_to_move_next, move

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
        return winner

    def get_board_shape(self):
        return np.zeros((self.board_size, self.board_size))

    def _initialize_state(self, board_size):
        self.board = np.zeros((board_size, board_size))
        self.switched = False
        self.legal_moves = set(
            [(i, j) for j in range(board_size) for i in range(board_size)]
        )
        self.first_move = None
        self.move_count = 0
        self.player_to_move = 1

    def _check_winning_state_player1(self):
        return self._dfs(1)

    def _check_winning_state_player2(self):
        return self._dfs(-1)

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

    def _dfs(self, player):
        visited = set()
        stack = []

        if player == 1:
            for i in range(self.board_size):
                if self.board[0, i] == 1:
                    stack.append((0, i))
                    visited.add((0, i))

            while stack:
                cell = stack.pop()
                if cell[0] == self.board_size - 1:
                    return True

                for neighbor in self._expand_neighbors(cell, player):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

        else:
            for i in range(self.board_size):
                if self.board[i, 0] == -1:
                    stack.append((i, 0))
                    visited.add((i, 0))

            while stack:
                cell = stack.pop()
                if cell[1] == self.board_size - 1:
                    return True

                for neighbor in self._expand_neighbors(cell, player):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

        return False

    def _is_within_bounds(self, cell):
        row, col = cell
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def _change_player_to_move(self, player_to_move):
        self.player_to_move = -player_to_move
