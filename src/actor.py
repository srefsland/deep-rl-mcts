import numpy as np


class Actor:
    """The actor class is used to make moves from trained neural networks.
    """

    def __init__(self, name, model, board_size):
        self.name = name
        self.model = model
        self.board_size = board_size

    def convert_state_to_nn_input(self, state):
        """Converts the game state to the input format of the convolutional neural network. 
        The 5 channels represent the game state as bitboards:
        Channel 1 is the cells occupied by player 1.
        Channel 2 is the cells occupied by player 2.
        Channel 3 is the cells currently unoccupied.
        Channel 4 are all 1's if the current player is 1.
        Channel 5 are all 1's if the current player is 2.

        Args:
            state (np.ndarray): the player ID along with the board state.

        Returns:
            np.ndarray: the nn input of shape (0, board_size, board_size, 5),
        """
        player_to_move = state[0]
        board = np.array(state[1:]).reshape((self.board_size, self.board_size))

        nn_input = np.zeros(
            shape=(self.board_size, self.board_size, 5), dtype=np.int8)

        is_occupied_player_1 = np.vectorize(lambda x: 1 if x == 1 else 0)
        is_occupied_player_2 = np.vectorize(lambda x: 1 if x == 2 else 0)
        is_occupied_empty = np.vectorize(lambda x: 1 if x == 0 else 0)

        nn_input[:, :, 0] = is_occupied_player_1(board)
        nn_input[:, :, 1] = is_occupied_player_2(board)
        nn_input[:, :, 2] = is_occupied_empty(board)
        nn_input[:, :, 3] = 1 if player_to_move == 1 else 0
        nn_input[:, :, 4] = 1 if player_to_move == 2 else 0

        nn_input = np.expand_dims(nn_input, axis=0)

        return nn_input

    def predict_move(self, model_input, temperature=1.0):
        """Predicts what move is the best to take given the model input.

        Args:
            model_input (np.ndarray): the state as model input.
            temperature (float, optional): probability to take best or probabilistic move. Defaults to 1.0.

        Returns:
            tuple[int, int]: the move taken.
        """
        if np.random.random() < temperature:
            moves = self.model.call(model_input).reshape(-1,)

            indices = np.arange(len(moves))

            move = np.random.choice(indices, p=moves)
        else:
            moves = self.model.call(model_input).reshape(-1,)

            move = np.argmax(moves)

        move = (move // self.board_size, move % self.board_size)

        return move
