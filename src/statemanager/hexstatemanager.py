from typing import Any, Generator, Optional, Set, Tuple

import numpy as np

from utils.disjoint_set import DisjointSet

from .exceptions import IllegalMoveException
from .statemanager import StateManager


class HexStateManager(StateManager):

    def __init__(self, board_size: int = 7, initialize: bool = True) -> None:
        self.board_size = board_size
        if initialize:
            self._initialize_state(board_size)

    def copy_state_manager(self) -> "HexStateManager":
        state_manager_copy = HexStateManager(self.board_size, initialize=False)
        state_manager_copy.board = self.board.copy()

        state_manager_copy.switched = self.switched
        state_manager_copy.legal_moves = set(self.legal_moves)
        state_manager_copy.first_move = self.first_move
        state_manager_copy.move_count = self.move_count
        state_manager_copy.player_to_move = self.player_to_move

        state_manager_copy.top_node = self.top_node
        state_manager_copy.bottom_node = self.bottom_node
        state_manager_copy.left_node = self.left_node
        state_manager_copy.right_node = self.right_node

        state_manager_copy.ds_player1 = self.ds_player1.copy()
        state_manager_copy.ds_player2 = self.ds_player2.copy()

        return state_manager_copy

    def get_legal_moves(
        self, player_to_move: Optional[int] = None
    ) -> Set[Tuple[int, int]]:
        return self.legal_moves

    def make_move(
        self, move: Tuple[int, int], player_to_move: Optional[int] = None
    ) -> Tuple[int, int]:
        if player_to_move is None:
            player_to_move = self.player_to_move

        if move not in self.legal_moves:
            raise IllegalMoveException()

        is_first_move = self.move_count == 0
        is_second_move = self.move_count == 1

        if not is_first_move and not is_second_move:
            self.board[move] = player_to_move
            self.legal_moves.remove(move)
            self._union_move(move, player_to_move)
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
                self._union_move(mirrored_move, player_to_move)

                move = mirrored_move
            else:
                self.board[move] = player_to_move
                self.legal_moves.remove(move)
                self.legal_moves.remove(self.first_move)

                self._union_move(move, player_to_move)
                self._union_move(self.first_move, 1)

        self.move_count += 1
        self._change_player_to_move(player_to_move)

        return move

    def undo_move(
        self, move: Tuple[int, int], player_to_move: Optional[int] = None
    ) -> None:
        if player_to_move is None:
            player_to_move = -self.player_to_move  # The player who made the move

        self.board[move] = 0
        self.legal_moves.add(move)
        self.move_count -= 1
        self._change_player_to_move(player_to_move)

    def make_random_move(
        self, player_to_move: Optional[int] = None
    ) -> Optional[Tuple[int, int]]:
        if player_to_move is None:
            player_to_move = self.player_to_move

        moves = self.get_legal_moves()

        if len(moves) == 0:
            return

        move = self.make_move(
            list(moves)[np.random.randint(0, len(moves))], player_to_move
        )

        return move

    def generate_child_states(
        self, player_to_move: Optional[int] = None
    ) -> Generator[Tuple[Tuple[int, int], int], None, None]:
        for move in self.get_legal_moves():
            yield move, -self.player_to_move

    def check_winning_state(self, player_moved: Optional[int] = None) -> bool:
        if player_moved == 1:
            return self._check_winning_state_player1()
        elif player_moved == -1:
            return self._check_winning_state_player2()
        else:
            return (
                self._check_winning_state_player1()
                or self._check_winning_state_player2()
            )

    def reset(self) -> None:
        self._initialize_state(self.board_size)

    def get_eval(self, winner: int = 1) -> int:
        return winner

    def get_board_shape(self) -> Any:
        return np.zeros((self.board_size, self.board_size))

    def _initialize_state(self, board_size: int) -> None:
        self.board = np.zeros((board_size, board_size))
        self.switched = False
        self.legal_moves = set(
            [(i, j) for j in range(board_size) for i in range(board_size)]
        )
        self.first_move = None
        self.move_count = 0
        self.player_to_move = 1

        self.top_node = (-1, 0)
        self.bottom_node = (board_size, 0)
        self.left_node = (0, -1)
        self.right_node = (0, board_size)

        all_board_positions = [
            (i, j) for j in range(board_size) for i in range(board_size)
        ]

        self.ds_player1 = DisjointSet(
            all_board_positions + [self.top_node, self.bottom_node]
        )
        self.ds_player2 = DisjointSet(
            all_board_positions + [self.left_node, self.right_node]
        )

        for i in range(board_size):
            self.ds_player1.union(self.top_node, (0, i))
            self.ds_player1.union(self.bottom_node, (board_size - 1, i))
            self.ds_player2.union(self.left_node, (i, 0))
            self.ds_player2.union(self.right_node, (i, board_size - 1))

    def _check_winning_state_player1(self) -> bool:
        return self.ds_player1.find(self.top_node) == self.ds_player1.find(
            self.bottom_node
        )

    def _check_winning_state_player2(self) -> bool:
        return self.ds_player2.find(self.left_node) == self.ds_player2.find(
            self.right_node
        )

    def _expand_neighbors(
        self, cell: Tuple[int, int], player_to_move: Optional[int] = None
    ) -> list[Tuple[int, int]]:
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

        neighbors: list[Tuple[int, int]] = []

        for neighbor in neighbors_coords:
            if (
                self._is_within_bounds(neighbor)
                and self.board[neighbor] == player_to_move
            ):
                neighbors.append(neighbor)

        return neighbors

    def _union_move(
        self, move: Tuple[int, int], player_to_move: Optional[int] = None
    ) -> None:
        if player_to_move is None:
            player_to_move = self.player_to_move

        neighbors = self._expand_neighbors(move, player_to_move)

        if player_to_move == 1:
            for neighbor in neighbors:
                self.ds_player1.union(move, neighbor)
        else:
            for neighbor in neighbors:
                self.ds_player2.union(move, neighbor)

    def _is_within_bounds(self, cell: Tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def _change_player_to_move(self, player_to_move: int) -> None:
        self.player_to_move = -player_to_move
