
from abc import ABC, abstractmethod
from typing import Any, Set, Tuple, Generator, Optional


class StateManager(ABC):
    @abstractmethod
    def copy_state_manager(self) -> 'StateManager':
        pass

    @abstractmethod
    def get_legal_moves(self, player_to_move: Optional[int]) -> Set[Tuple[int, int]]:
        pass

    @abstractmethod
    def make_move(self, move: Tuple[int, int], player_to_move: Optional[int]) -> Tuple[int, int]:
        pass

    @abstractmethod
    def make_random_move(self, player_to_move: Optional[int]) -> Optional[Tuple[int, int]]:
        pass

    @abstractmethod
    def generate_child_states(self, player_to_move: Optional[int]) -> Generator[Tuple[Tuple[int, int], int], None, None]:
        pass

    @abstractmethod
    def check_winning_state(self, player_moved: Optional[int]) -> bool:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_eval(self, winner: int) -> int:
        pass

    @abstractmethod
    def get_board_shape(self) -> Any:
        pass

