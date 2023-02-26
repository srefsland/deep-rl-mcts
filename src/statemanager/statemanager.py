from abc import ABC, abstractmethod


class StateManager(ABC):
    @abstractmethod
    def check_winning_state(self, player):
        pass

    @abstractmethod
    def generate_child_states(self, player):
        pass

    @abstractmethod
    def get_moves_legal(self, player):
        pass

    @abstractmethod
    def make_move(self, move, player):
        pass

    @abstractmethod
    def copy_state(self):
        pass

    @abstractmethod
    def get_eval(self, winner):
        pass

    @abstractmethod
    def convert_to_nn_input(self):
        pass
