import logging
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

import config
import replay_buffer
from actor import Actor
from display.hexboarddisplay import HexBoardDisplay
from display.hexboarddisplayclassic import HexBoardDisplayClassic
from mcts.mcts import MCTS
from nn.boardgamenetcnn import BoardGameNetCNN
from statemanager.hexstatemanager import HexStateManager


def rl_algorithm(actor, state_manager, mcts_state_manager, display):
    """The reinforcement learning algorithm.

    Args:
        actor: the actor to use.
        state_manager (StateManager): the state manager class to use.
        display (GameBoardDisplay): game board display class to use.
    """
    # Configure logging level and format for console output
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    replay_buf = replay_buffer.ReplayBuffer(maxlen=config.REPLAY_BUFFER_SIZE)
    i_s = config.SAVE_INTERVAL
    time_stamp = datetime.now()
    replay_buf.clear()

    for g_a in tqdm(range(config.NUM_EPISODES + 1)):
        state_manager.reset()
        mcts_state_manager.reset()
        actor.create_lite_model()
        
        logging.info(f"Episode {g_a}: current epsilon: {actor.epsilon:.2f}, current epsilon critic: {actor.epsilon_critic:.2f}")

        mcts_tree = MCTS(
            state_manager=mcts_state_manager, c=config.MTCS_C, use_critic=config.USE_CRITIC
        )

        moves = 0
        while not state_manager.check_winning_state():
            logging.info(f"Move {moves}")
    
            start_time = time.time()
            i = 0

            while (
                time.time() - start_time < config.MCTS_DYNAMIC_SIMS_TIME
                or i < config.MCTS_MIN_SIMULATIONS
            ):
                i += 1
                mcts_tree.simulation_iteration(actor)
            logging.info(f"Number of simulations: {i}, time: {(time.time() - start_time):.2f} seconds")

            moves += 1

            distribution = mcts_tree.get_visit_distribution(mcts_tree.root)

            replay_buf.add_case(
              (
                  nn.convert_to_nn_input(mcts_tree.root.state, mcts_tree.root.player),
                  distribution,
                  np.array([mcts_tree.root.get_qsa()]),
              )
            )

            s_move = (
                mcts_tree.select_best_distribution()
                if config.SELECT_BEST_MOVE_RL
                else mcts_tree.select_random_best_distribution()
            )

            state_manager.make_move(s_move)
            mcts_tree.prune_tree(s_move)

            if config.DISPLAY_GAME_RL and g_a % config.DISPLAY_GAME_RL_INTERVAL == 0:
                display.display_board(
                    state_manager, delay=0.1, newest_move=s_move
                )

        X, y_actor, y_critic = replay_buf.get_random_minibatch(config.MINI_BATCH_SIZE)

        actor.train_model(X, y_actor, y_critic, epochs=config.NUM_EPOCHS)
        actor.decrease_epsilon()
    
        if g_a % i_s == 0:
            nn.save_model(f"models/{time_stamp}/model_{config.BOARD_SIZE}x{config.BOARD_SIZE}_{g_a}")

            if g_a != 0:
                nn.save_losses()


if __name__ == "__main__":
    nn = BoardGameNetCNN(
        convolutional_layers=config.CNN_FILTERS,
        neural_network_dimensions=config.NEURAL_NETWORK_DIMENSIONS,
        lr=config.LEARNING_RATE,
        activation=config.ACTIVATION_FUNCTION,
        output_activation_actor=config.OUTPUT_ACTIVATION_FUNCTION_ACTOR,
        output_activation_critic=config.OUTPUT_ACTIVATION_FUNCTION_CRITIC,
        loss_actor=config.LOSS_FUNCTION_ACTOR,
        loss_critic=config.LOSS_FUNCTION_CRITIC,
        optimizer=config.ANN_OPTIMIZER,
        board_size=config.BOARD_SIZE,
    )
    state_manager = HexStateManager(config.BOARD_SIZE)
    mcts_state_manager = HexStateManager(config.BOARD_SIZE)
    display = None if not config.DISPLAY_GAME_RL else HexBoardDisplayClassic() if config.CLASSIC_DISPLAY else HexBoardDisplay()
    actor = Actor(
        name="actor_rl",
        nn=nn,
        board_size=config.BOARD_SIZE,
        epsilon=config.EPSILON,
        epsilon_decay=config.EPSILON_DECAY,
        epsilon_critic=config.EPSILON_CRITIC,
        epsilon_decay_critic=config.EPSILON_DECAY_CRITIC,
        litemodel=None,
    )
    rl_algorithm(actor=actor, state_manager=state_manager, mcts_state_manager=mcts_state_manager, display=display)
