import config
from actor import Actor
from display.hexboarddisplay import HexBoardDisplay
from display.hexboarddisplayclassic import HexBoardDisplayClassic
from nn.boardgamenetcnn import BoardGameNetCNN
from statemanager.hexstatemanager import HexStateManager


def play_versus_actor(actor, board_display, board_size=4, best_move=True, player1=True):
    board = HexStateManager(board_size=board_size, )

    new_move = None
    is_terminal = False
    while not is_terminal:
        board_display.display_board(board, delay=0.5, newest_move=new_move)
        current_player = board.player

        # If switch player, then we need to switch the player for the actor as well
        if current_player == 1 and player1 and not board.switched:
            if config.CLASSIC_DISPLAY:
                x = input("Enter position: ")
                
                j = ord(x[0].upper()) - ord("A")
                i = int(x[1]) - 1
                move = (i, j)
                
            else:
                x = input("Enter x: ")
                y = input("Enter y: ")

                move = (int(x), int(y))
        else:
            if best_move:
                move = actor.predict_best_move(board.board, board.player, board.legal_moves)
            else:
                move = actor.predict_probabilistic_move(board.board, board.player, board.legal_moves)

        new_move = board.make_move(move)
        is_terminal = board.check_winning_state(current_player)

    board_display.display_board(board, delay=0.5, winner=current_player)

if __name__ == "__main__":
    actor_episodes = 200

    saved_model = f"{config.MODEL_DIR}/model_{config.BOARD_SIZE}x{config.BOARD_SIZE}_{actor_episodes}"
    model = BoardGameNetCNN(board_size=config.BOARD_SIZE, bridge_features=config.BRIDGE_FEATURES, saved_model=saved_model)
    actor = Actor("actor1", model, board_size=config.BOARD_SIZE)
    
    display = HexBoardDisplayClassic() if config.CLASSIC_DISPLAY else HexBoardDisplay()

    continue_playing = True
    while continue_playing:
        player_choice = input("Play as player 1 or 2? (1/2) ") == "1"
        play_versus_actor(
            actor, display, board_size=config.BOARD_SIZE, best_move=True, player1=player_choice
        )
        continue_playing = input("Play again? (y/n) ") == "y"
