from src.statemanager.hexstatemanager import HexStateManager
import pytest


def setup_board(player=1):
    setup_board = HexStateManager(4)

    setup_board.make_move((0, 0), player)
    setup_board.make_move((1, 0), player)
    setup_board.make_move((2, 0), player)
    setup_board.make_move((1, 1), player)
    setup_board.make_move((2, 1), player)
    setup_board.make_move((1, 2), player)

    return setup_board


def test_win_state_player1():
    setup_board_win_state_check = setup_board(1)

    setup_board_win_state_check.make_move((1, 3), 1)

    is_win = setup_board_win_state_check._check_winning_state_player1()
    assert is_win is False

    setup_board_win_state_check = setup_board(1)

    setup_board_win_state_check.make_move((2, 2), 1)

    is_win = setup_board_win_state_check._check_winning_state_player1()
    assert is_win is False

    setup_board_win_state_check = setup_board(1)

    setup_board_win_state_check.make_move((2, 2), 1)
    setup_board_win_state_check.make_move((3, 1), 1)

    is_win = setup_board_win_state_check._check_winning_state_player1()
    assert is_win is True


def test_win_state_player2():
    setup_board_win_state_check = setup_board(-1)

    setup_board_win_state_check.make_move((1, 3), -1)
    setup_board_win_state_check.make_move((0, 3), -1)

    is_win = setup_board_win_state_check._check_winning_state_player2()
    assert is_win is True

    setup_board_win_state_check = setup_board(-1)

    setup_board_win_state_check.make_move((2, 2), -1)
    setup_board_win_state_check.make_move((3, 2), -1)

    is_win = setup_board_win_state_check._check_winning_state_player2()
    assert is_win is False


def test_move_making():
    board = HexStateManager(4)

    board.make_move((0, 0), -1)
    assert board.board[0][0] == -1
    assert board.player == 1

    board.make_move((0, 1), 1)
    assert board.board[0][1] == 1
    assert board.player == -1
    
def test_switch_rule():
    board = HexStateManager(4, switch_rule_allowed=True)
    
    board.make_move((0, 0))
    assert board.player == -1
    assert board.switched == False
    
    assert len([_ for _ in board.generate_child_states()]) == 16
    
    board.make_move((0, 0))
    assert board.player == -1
    assert board.switched == True
    
    assert len([_ for _ in board.generate_child_states()]) == 15
    
    with pytest.raises(Exception):
        board.make_move((0, 0))
    
    board.make_move((0, 1))
    assert board.player == 1
