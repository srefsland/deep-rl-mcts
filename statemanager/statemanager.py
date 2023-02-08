from abc import ABC, abstractmethod

class StateManager(ABC):
    @abstractmethod
    def check_winning_state(self, player):
        pass
    
    @abstractmethod
    def generate_child_states(self, player):
        pass