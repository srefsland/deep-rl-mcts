from abc import ABC, abstractmethod


class GameBoardDisplay(ABC):
    @abstractmethod
    def display_board(self, board, delay=0, winner=None, newest_move=None):
        pass
