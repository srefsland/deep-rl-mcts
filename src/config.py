import yaml
from pathlib import Path

# Read config.yml and expose top-level constants expected by the project.
_CFG_PATH = Path(__file__).resolve().parent / "config.yml"
_cfg = yaml.safe_load(_CFG_PATH.read_text())

# Board
BOARD_SIZE = _cfg["board"]["size"]
CLASSIC_DISPLAY = _cfg["board"]["classic_display"]

# MCTS
MCTS_DYNAMIC_SIMS_TIME = _cfg["mcts"]["dynamic_sims_time"]
MCTS_MIN_SIMULATIONS = _cfg["mcts"]["min_simulations"]
MTCS_C = _cfg["mcts"]["c"]
EPSILON = _cfg["mcts"]["epsilon"]
EPSILON_DECAY = _cfg["mcts"]["epsilon_decay"]
EPSILON_CRITIC = _cfg["mcts"]["epsilon_critic"]
EPSILON_DECAY_CRITIC = _cfg["mcts"]["epsilon_decay_critic"]

# RL
NUM_EPISODES = _cfg["rl"]["num_episodes"]
DISPLAY_GAME_RL = _cfg["rl"]["display_game_rl"]
DISPLAY_GAME_RL_INTERVAL = _cfg["rl"]["display_game_rl_interval"]
REPLAY_BUFFER_SIZE = _cfg["rl"]["replay_buffer_size"]
MINI_BATCH_SIZE = _cfg["rl"]["mini_batch_size"]
SAVE_INTERVAL = _cfg["rl"]["save_interval"]
SELECT_BEST_MOVE_RL = _cfg["rl"]["select_best_move_rl"]

# ANN
LEARNING_RATE = _cfg["ann"]["learning_rate"]
CNN_NUM_FILTERS = _cfg["ann"]["cnn_num_filters"]
NUM_RES_BLOCKS = _cfg["ann"]["num_res_blocks"]
ACTIVATION_FUNCTION = _cfg["ann"]["activation_function"]
OUTPUT_ACTIVATION_FUNCTION_ACTOR = _cfg["ann"]["output_activation_actor"]
OUTPUT_ACTIVATION_FUNCTION_CRITIC = _cfg["ann"]["output_activation_critic"]
ANN_OPTIMIZER = _cfg["ann"]["optimizer"]
LOSS_FUNCTION_ACTOR = _cfg["ann"]["loss_function_actor"]
LOSS_FUNCTION_CRITIC = _cfg["ann"]["loss_function_critic"]
NUM_EPOCHS = _cfg["ann"]["num_epochs"]
BRIDGE_FEATURES = _cfg["ann"]["bridge_features"]
USE_CRITIC = _cfg["ann"]["use_critic"]
MODEL_DIR = _cfg["ann"]["model_dir"]

