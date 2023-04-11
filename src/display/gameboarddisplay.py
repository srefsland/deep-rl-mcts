from abc import ABC, abstractmethod


# Abstract class for game board dispalys. Ensures that visualization is not specific to Hex.
class GameBoardDisplay(ABC):
    @abstractmethod
    def display_board(self, board, delay=0, winner=None, newest_move=None):
        pass
