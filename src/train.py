import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from actor import Actor
from display.gameboarddisplay import GameBoardDisplay
from display.hexboarddisplay import HexBoardDisplay
from display.hexboarddisplayclassic import HexBoardDisplayClassic
from mcts.mcts import MCTS
from nn.batch_inference import BatchActorInferenceWorker
from nn.board_encoder import convert_board_state_to_tensor
from nn.hexresnet import HexResNet
from nn.options_torch import loss_functions, optimizers
from nn.train import train
from replay_buffer import ReplayBuffer, ReplayCase
from statemanager.hexstatemanager import HexStateManager
from statemanager.statemanager import StateManager

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def parallel_mcts_tree_search(
    mcts_tree: MCTS, actor: Actor, num_simulations: list, worker_num: int
):
    start_time = time.time()

    while (
        time.time() - start_time < config.MCTS_DYNAMIC_SIMS_TIME
        or num_simulations.sum() < config.MCTS_MIN_SIMULATIONS
    ):
        mcts_tree.simulation_iteration(actor)

        num_simulations[worker_num] += 1


def mcts_tree_search(mcts_tree: MCTS, actor: Actor, num_simulations: list):
    start_time = time.time()

    while (
        time.time() - start_time < config.MCTS_DYNAMIC_SIMS_TIME
        or num_simulations[0] < config.MCTS_MIN_SIMULATIONS
    ):
        mcts_tree.simulation_iteration(actor)

        num_simulations[0] += 1


def rl_algorithm(
    actor: Actor,
    state_manager: StateManager,
    mcts_state_manager: StateManager,
    display: GameBoardDisplay,
):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info(f"Using device: {actor.nn.device}")

    replay_buf = ReplayBuffer(maxlen=config.REPLAY_BUFFER_SIZE)
    i_s = config.SAVE_INTERVAL

    replay_buf.clear()

    for g_a in tqdm(range(config.NUM_EPISODES + 1)):
        state_manager.reset()
        mcts_state_manager.reset()

        logging.info(
            f"Episode {g_a}: current epsilon: {actor.epsilon:.2f}, current epsilon critic: {actor.epsilon_critic:.2f}"
        )

        mcts_tree = MCTS(
            state_manager=mcts_state_manager,
            c=config.MTCS_C,
            use_critic=config.USE_CRITIC,
            use_locks=config.RL_MULTI_THREADING,
        )

        moves = 0
        while not state_manager.check_winning_state():
            logging.info(f"Move {moves}")

            start_time = time.time()

            num_workers = 32

            num_simulations = np.zeros(num_workers, dtype=int)

            if config.RL_MULTI_THREADING:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            parallel_mcts_tree_search,
                            mcts_tree,
                            actor,
                            num_simulations,
                            i,
                        )
                        for i in range(num_workers)
                    ]

                    for future in as_completed(futures):
                        future.result()
            else:
                mcts_tree_search(mcts_tree, actor, num_simulations)

            logging.info(
                f"Number of simulations: {num_simulations.sum()}, time: {(time.time() - start_time):.2f} seconds"
            )

            moves += 1

            distribution = mcts_tree.get_visit_distribution(mcts_tree.root)

            board_state = (
                convert_board_state_to_tensor(
                    state_manager.board.copy(), state_manager.player_to_move
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            replay_buf.add_case(
                ReplayCase(
                    board_state=board_state,
                    target_distribution=distribution,
                    target_value=np.array([mcts_tree.root.get_qsa()]),
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
                display.display_board(state_manager, delay=0.1, newest_move=s_move)

        minibatch = replay_buf.get_random_minibatch(config.MINI_BATCH_SIZE)

        train_dataset = TensorDataset(
            torch.tensor(minibatch.X).float(),
            torch.tensor(minibatch.y_actor).float(),
            torch.tensor(minibatch.y_critic).float(),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config.MINI_BATCH_SIZE, shuffle=True
        )

        checkpoint_dir = None
        if g_a % i_s != 0:
            date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_dir = (
                f"{config.MODEL_DIR}/episode_{g_a}_size_{config.BOARD_SIZE}_{date_str}"
            )

        checkpoint_dir = None

        if g_a % i_s == 0:
            date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_dir = (
                f"{config.MODEL_DIR}/episode_{g_a}_size_{config.BOARD_SIZE}_{date_str}"
            )

        train(
            model=actor.nn,
            train_loader=train_loader,
            optimizer=optimizers[config.ANN_OPTIMIZER](
                actor.nn.parameters(), lr=config.LEARNING_RATE
            ),
            device=actor.nn.device,
            epochs=config.NUM_EPOCHS,
            loss_actor_fn=loss_functions[config.LOSS_FUNCTION_ACTOR](),
            loss_critic_fn=loss_functions[config.LOSS_FUNCTION_CRITIC](),
            checkpoint_dir=checkpoint_dir,
        )

        actor.decrease_epsilon()


if __name__ == "__main__":
    cnn_filter_size = config.CNN_NUM_FILTERS
    num_res_blocks = config.NUM_RES_BLOCKS

    nn = HexResNet(
        board_size=config.BOARD_SIZE,
        in_channels=5,
        num_filters=cnn_filter_size,
        num_res_blocks=num_res_blocks,
    )

    nn.to(device)
    nn.device = device

    state_manager = HexStateManager(config.BOARD_SIZE)
    mcts_state_manager = HexStateManager(config.BOARD_SIZE)

    display = (
        None
        if not config.DISPLAY_GAME_RL
        else HexBoardDisplayClassic() if config.CLASSIC_DISPLAY else HexBoardDisplay()
    )

    num_workers = 32

    batch_worker = BatchActorInferenceWorker(model=nn, max_batch_size=num_workers)

    actor = Actor(
        name="actor_rl",
        nn=nn,
        board_size=config.BOARD_SIZE,
        epsilon=config.EPSILON,
        epsilon_decay=config.EPSILON_DECAY,
        epsilon_critic=config.EPSILON_CRITIC,
        epsilon_decay_critic=config.EPSILON_DECAY_CRITIC,
        batch_worker=batch_worker if config.RL_MULTI_THREADING else None,
    )

    rl_algorithm(
        actor=actor,
        state_manager=state_manager,
        mcts_state_manager=mcts_state_manager,
        display=display,
    )
