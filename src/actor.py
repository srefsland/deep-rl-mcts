import numpy as np


class Actor():
    def __init__(self, name, model, board_size):
        self.name = name
        self.model = model
        self.board_size = board_size

    def convert_state_to_nn_input(self, state):
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
        if np.random.random() < temperature:
            moves = self.model.predict(model_input).reshape(-1,)

            indices = np.arange(len(moves))

            move = np.random.choice(indices, p=moves)
        else:
            moves = self.model.predict(model_input).reshape(-1,)

            move = np.argmax(moves)

        move = (move // self.board_size, move % self.board_size)

        return move
