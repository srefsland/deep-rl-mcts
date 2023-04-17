from oht.ActorClient import ActorClient
from actor import Actor
from nn.boardgamenetcnn import BoardGameNetCNN
import numpy as np

board_size = 7
model_dir = "models"
model_eps = 160
model = BoardGameNetCNN(
    saved_model=f"{model_dir}/model_{board_size}x{board_size}_{model_eps}", board_size=board_size)
hex_actor = Actor('hex_actor', model, board_size=board_size)


class MyClient(ActorClient):
    def handle_series_start(
        self, unique_id, series_id, player_map, num_games, game_params
    ):
        """Called at the start of each set of games against an opponent

        Args:
            unique_id (int): your unique id within the tournament
            series_id (int): whether you are player 1 or player 2
            player_map (list): (inique_id, series_id) touples for both players
            num_games (int): number of games that will be played
            game_params (list): game-specific parameters.

        Note:
            > For the qualifiers, your player_id should always be "-200",
              but this can change later
            > For Hex, game params will be a 1-length list containing
              the size of the game board ([board_size])
        """
        self.logger.info(
            'Series start: unique_id=%s series_id=%s player_map=%s num_games=%s'
            ', game_params=%s',
            unique_id, series_id, player_map, num_games, game_params,
        )

    def handle_game_start(self, start_player):
        """Called at the beginning of of each game

        Args:
            start_player (int): the series_id of the starting player (1 or 2)
        """
        self.logger.info('Game start: start_player=%s', start_player)

    def handle_get_action(self, state):
        """Called whenever it's your turn to pick an action

        Args:
            state (list): board configuration as a list of board_size^2 + 1 ints

        Returns:
            tuple: action with board coordinates (row, col) (a list is ok too)

        Note:
            > Given the following state for a 5x5 Hex game
                state = [
                    1,              # Current player (you) is 1
                    0, 0, 0, 0, 0,  # First row
                    0, 2, 1, 0, 0,  # Second row
                    0, 0, 1, 0, 0,  # ...
                    2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                ]
            > Player 1 goes "top-down" and player 2 goes "left-right"
            > Returning (3, 2) would put a "1" at the free (0) position
              below the two vertically aligned ones.
            > The neighborhood around a cell is connected like
                  |/
                --0--
                 /|
        """
        nn_input = self.convert_state_to_nn_input_cnn(state)

        move = hex_actor.predict_move_model_input(model_input=nn_input)
        move = (int(move[0]), int(move[1]))
        self.logger.info(f'Get action: state={state}')
        self.logger.info(f'Picked deliberate move: {move[0]} {move[1]}')
        return move[0], move[1]

    def handle_game_over(self, winner, end_state):
        """Called after each game

        Args:
            winner (int): the winning player (1 or 2)
            end_stats (tuple): final board configuration

        Note:
            > Given the following end state for a 5x5 Hex game
            state = [
                2,              # Current player is 2 (doesn't matter)
                0, 2, 0, 1, 2,  # First row
                0, 2, 1, 0, 0,  # Second row
                0, 0, 1, 0, 0,  # ...
                2, 2, 1, 0, 0,
                0, 1, 0, 0, 0
            ]
            > Player 1 has won here since there is a continuous
              path of ones from the top to the bottom following the
              neighborhood description given in `handle_get_action`
        """
        self.logger.info('Game over: winner=%s end_state=%s',
                         winner, end_state)

    def handle_series_over(self, stats):
        """Called after each set of games against an opponent is finished

        Args:
            stats (list): a list of lists with stats for the series players

        Example stats (suppose you have ID=-200, and playing against ID=999):
            [
                [-200, 1, 7, 3],  # id=-200 is player 1 with 7 wins and 3 losses
                [ 999, 2, 3, 7],  # id=+999 is player 2 with 3 wins and 7 losses
            ]
        """
        self.logger.info('Series over: stats=%s', stats)

    def handle_tournament_over(self, score):
        """Called after all series have finished

        Args:
            score (float): Your score (your win %) for the tournament
        """
        self.logger.info('Tournament over: score=%s', score)
        
    def convert_state_to_nn_input_cnn(self, state):
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
        board = np.array(state[1:]).reshape((board_size, board_size))

        nn_input = np.zeros(
            shape=(board_size, board_size, 5), dtype=np.int8)

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


if __name__ == '__main__':
    client = MyClient()
    client.run()
