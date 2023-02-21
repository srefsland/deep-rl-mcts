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

    is_win = setup_board_win_state_check.check_winning_state_player1()

    assert is_win is False

    setup_board_win_state_check = setup_board((1, 0))

    setup_board_win_state_check.make_move((2, 2), (1, 0))

    is_win = setup_board_win_state_check.check_winning_state_player1()

    assert is_win is False

    setup_board_win_state_check = setup_board((1, 0))

    setup_board_win_state_check.make_move((2, 2), (1, 0))
    setup_board_win_state_check.make_move((3, 1), (1, 0))

    is_win = setup_board_win_state_check.check_winning_state_player1()

    assert is_win is True


def test_win_state_player2():
    setup_board_win_state_check = setup_board((0, 1))

    setup_board_win_state_check.make_move((1, 3), (0, 1))
    setup_board_win_state_check.make_move((0, 3), (0, 1))

    is_win = setup_board_win_state_check.check_winning_state_player2()

    assert is_win is True

    setup_board_win_state_check = setup_board((0, 1))

    setup_board_win_state_check.make_move((2, 2), (0, 1))
    setup_board_win_state_check.make_move((3, 2), (0, 1))

    is_win = setup_board_win_state_check.check_winning_state_player2()

    assert is_win is False


def test_conversion_to_diamond_shape():
    setup_board_win_state_check = HexStateManager(4)

    diamond_array = setup_board_win_state_check.convert_to_diamond_shape()

    assert diamond_array[0][0].get_position() == (0, 0)

    assert diamond_array[1][0].get_position() == (1, 0)
    assert diamond_array[1][1].get_position() == (0, 1)

    assert diamond_array[2][0].get_position() == (2, 0)
    assert diamond_array[2][1].get_position() == (1, 1)
    assert diamond_array[2][2].get_position() == (0, 2)

    assert diamond_array[3][0].get_position() == (3, 0)
    assert diamond_array[3][1].get_position() == (2, 1)
    assert diamond_array[3][2].get_position() == (1, 2)
    assert diamond_array[3][3].get_position() == (0, 3)

    assert diamond_array[4][0].get_position() == (3, 1)
    assert diamond_array[4][1].get_position() == (2, 2)
    assert diamond_array[4][2].get_position() == (1, 3)

    assert diamond_array[5][0].get_position() == (3, 2)
    assert diamond_array[5][1].get_position() == (2, 3)

    assert diamond_array[6][0].get_position() == (3, 3)


def test_conversion_to_1D_array():
    board = HexStateManager(4)

    new_board = board.convert_to_1D_array()

    assert new_board.shape == (16,)

    assert new_board[0].get_position() == (0, 0)
    assert new_board[1].get_position() == (0, 1)
    assert new_board[2].get_position() == (0, 2)
    assert new_board[3].get_position() == (0, 3)
    assert new_board[4].get_position() == (1, 0)
    assert new_board[5].get_position() == (1, 1)
    assert new_board[6].get_position() == (1, 2)
    assert new_board[7].get_position() == (1, 3)
    assert new_board[8].get_position() == (2, 0)
    assert new_board[9].get_position() == (2, 1)
    assert new_board[10].get_position() == (2, 2)
    assert new_board[11].get_position() == (2, 3)
    assert new_board[12].get_position() == (3, 0)
    assert new_board[13].get_position() == (3, 1)
    assert new_board[14].get_position() == (3, 2)
    assert new_board[15].get_position() == (3, 3)


def test_move_making():
    board = HexStateManager(4)

    board.make_move((0, 0), (0, 1))
    assert board.board[0][0].get_owner() == (0, 1)
    assert board.player == (1, 0)

    with pytest.raises(Exception):
        board.make_move((0, 0), (0, 1))

    board.make_move((0, 1), (1, 0))
    assert board.board[0][1].get_owner() == (1, 0)
    assert board.player == (0, 1)


def test_immediate_winning_move_player1():
    setup_board_win_state_check = HexStateManager(4)
    setup_board_win_state_check.make_move((0, 0), (1, 0))
    winning_move = setup_board_win_state_check.find_immediate_winning_move(
        (1, 0))

    assert winning_move is None

    setup_board_win_state_check = HexStateManager(4)
    setup_board_win_state_check.make_move((3, 0), (1, 0))
    winning_move = setup_board_win_state_check.find_immediate_winning_move(
        (1, 0))

    assert winning_move is None

    setup_board_win_state_check = HexStateManager(4)

    setup_board_win_state_check.make_move((1, 0), (1, 0))
    setup_board_win_state_check.make_move((1, 3), (1, 0))

    winning_move = setup_board_win_state_check.find_immediate_winning_move(
        (1, 0))

    assert winning_move is None

    setup_board_win_state_check = setup_board((1, 0))

    winning_move = setup_board_win_state_check.find_immediate_winning_move(
        (1, 0))

    assert winning_move is not None


def test_immediate_winning_move_player2():
    setup_board_win_state_check = HexStateManager(4)
    setup_board_win_state_check.make_move((0, 0), (0, 1))
    winning_move = setup_board_win_state_check.find_immediate_winning_move(
        (0, 1))

    assert winning_move is None

    setup_board_win_state_check = HexStateManager(4)
    setup_board_win_state_check.make_move((0, 3), (0, 1))
    winning_move = setup_board_win_state_check.find_immediate_winning_move(
        (0, 1))

    assert winning_move is None

    setup_board_win_state_check = HexStateManager(4)
    setup_board_win_state_check.make_move((1, 1), (0, 1))
    setup_board_win_state_check.make_move((1, 2), (0, 1))

    winning_move = setup_board_win_state_check.find_immediate_winning_move(
        (0, 1))

    assert winning_move is None

    setup_board_win_state_check = setup_board((0, 1))

    winning_move = setup_board_win_state_check.find_immediate_winning_move(
        (0, 1))

    assert winning_move is not None
