import numpy as np


class Actor:
    """The actor class is used to make moves from trained neural networks.
    """

    def __init__(self, name, model, board_size):
        self.name = name
        self.model = model
        self.board_size = board_size

    def predict_move(self, state, temperature=1.0):
        """Predicts what move is the best to take given the model input.

        Args:
            state (StateManager): the state with the current board and current player.
            temperature (float, optional): probability to take best or probabilistic move. Defaults to 1.0.

        Returns:
            tuple[int, int]: the move taken.
        """
        if np.random.random() < temperature:
            move = self.model.predict_best_move(state=state)
        else:
            move = self.model.predict_probabilistic_move(state=state)

        return move

    def predict_move_model_input(self, model_input, temperature=1.0):
        """Predicts what move is the best to take given the model input.

        Args:
            model_input (np.array): the model input.
            temperature (float, optional): probability to take best or probabilistic move. Defaults to 1.0.

        Returns:
            tuple[int, int]: the move taken.
        """
        if np.random.random() < temperature:
            move = self.model.predict_best_move(model_input=model_input)
        else:
            move = self.model.predict_probabilistic_move(
                model_input=model_input)

        return move
