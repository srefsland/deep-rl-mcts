from src.statemanager.hexstatemanager import HexStateManager
import pytest


def setup_board(player=(1, 0)):
    setup_board = HexStateManager(4)

    setup_board.make_move((0, 0), player)
    setup_board.make_move((1, 0), player)
    setup_board.make_move((2, 0), player)
    setup_board.make_move((1, 1), player)
    setup_board.make_move((2, 1), player)
    setup_board.make_move((1, 2), player)

    return setup_board


def test_win_state_player1():
    setup_board_win_state_check = setup_board((1, 0))

    setup_board_win_state_check.make_move((1, 3), (1, 0))

    is_win = setup_board_win_state_check._check_winning_state_player1()
    assert is_win is False

    setup_board_win_state_check = setup_board((1, 0))

    setup_board_win_state_check.make_move((2, 2), (1, 0))

    is_win = setup_board_win_state_check._check_winning_state_player1()
    assert is_win is False

    setup_board_win_state_check = setup_board((1, 0))

    setup_board_win_state_check.make_move((2, 2), (1, 0))
    setup_board_win_state_check.make_move((3, 1), (1, 0))

    is_win = setup_board_win_state_check._check_winning_state_player1()
    assert is_win is True


def test_win_state_player2():
    setup_board_win_state_check = setup_board((0, 1))

    setup_board_win_state_check.make_move((1, 3), (0, 1))
    setup_board_win_state_check.make_move((0, 3), (0, 1))

    is_win = setup_board_win_state_check._check_winning_state_player2()
    assert is_win is True

    setup_board_win_state_check = setup_board((0, 1))

    setup_board_win_state_check.make_move((2, 2), (0, 1))
    setup_board_win_state_check.make_move((3, 2), (0, 1))

    is_win = setup_board_win_state_check._check_winning_state_player2()
    assert is_win is False


def test_conversion_to_diamond_shape():
    setup_board_win_state_check = HexStateManager(4)

    diamond_array = setup_board_win_state_check.convert_to_diamond_shape()

    assert diamond_array[0][0].position == (0, 0)

    assert diamond_array[1][0].position == (1, 0)
    assert diamond_array[1][1].position == (0, 1)

    assert diamond_array[2][0].position == (2, 0)
    assert diamond_array[2][1].position == (1, 1)
    assert diamond_array[2][2].position == (0, 2)

    assert diamond_array[3][0].position == (3, 0)
    assert diamond_array[3][1].position == (2, 1)
    assert diamond_array[3][2].position == (1, 2)
    assert diamond_array[3][3].position == (0, 3)

    assert diamond_array[4][0].position == (3, 1)
    assert diamond_array[4][1].position == (2, 2)
    assert diamond_array[4][2].position == (1, 3)

    assert diamond_array[5][0].position == (3, 2)
    assert diamond_array[5][1].position == (2, 3)

    assert diamond_array[6][0].position == (3, 3)


def test_move_making():
    board = HexStateManager(4)

    board.make_move((0, 0), (0, 1))
    assert board.board[0][0].occupant == (0, 1)
    assert board.player == (1, 0)

    with pytest.raises(Exception):
        board.make_move((0, 0), (0, 1))

    board.make_move((0, 1), (1, 0))
    assert board.board[0][1].occupant == (1, 0)
    assert board.player == (0, 1)