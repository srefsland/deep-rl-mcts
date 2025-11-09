from abc import ABC, abstractmethod


class StateManager(ABC):
    @abstractmethod
    def copy_state_manager(self):
        pass

    @abstractmethod
    def get_legal_moves(self, player_to_move):
        pass

    @abstractmethod
    def make_move(self, move, player_to_move):
        pass

    @abstractmethod
    def make_random_move(self, player_to_move):
        pass

    @abstractmethod
    def generate_child_states(self, player_to_move):
        pass

    @abstractmethod
    def check_winning_state(self, player_moved):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_eval(self, winner):
        pass

    @abstractmethod
    def get_board_shape(self):
        pass

