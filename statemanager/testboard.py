from board import Board
import unittest


class BoardTest(unittest.TestCase):
    def test_win_state_player1(self):
        setup_board_win_state_check = Board(4)

        setup_board_win_state_check.board[0][0].set_owner((1, 0))
        setup_board_win_state_check.board[1][0].set_owner((1, 0))
        setup_board_win_state_check.board[2][0].set_owner((1, 0))
        setup_board_win_state_check.board[1][1].set_owner((1, 0))
        setup_board_win_state_check.board[2][1].set_owner((1, 0))
        setup_board_win_state_check.board[1][2].set_owner((1, 0))
        setup_board_win_state_check.board[1][3].set_owner((1, 0))

        is_win = setup_board_win_state_check.check_winning_state_player1()

        self.assertFalse(is_win)

        setup_board_win_state_check = Board(4)

        setup_board_win_state_check.board[0][0].set_owner((1, 0))
        setup_board_win_state_check.board[1][0].set_owner((1, 0))
        setup_board_win_state_check.board[2][0].set_owner((1, 0))
        setup_board_win_state_check.board[1][1].set_owner((1, 0))
        setup_board_win_state_check.board[2][1].set_owner((1, 0))
        setup_board_win_state_check.board[1][2].set_owner((1, 0))
        setup_board_win_state_check.board[2][2].set_owner((1, 0))

        is_win = setup_board_win_state_check.check_winning_state_player1()

        self.assertFalse(is_win)

        setup_board_win_state_check = Board(4)

        setup_board_win_state_check.board[0][0].set_owner((1, 0))
        setup_board_win_state_check.board[1][0].set_owner((1, 0))
        setup_board_win_state_check.board[2][0].set_owner((1, 0))
        setup_board_win_state_check.board[1][1].set_owner((1, 0))
        setup_board_win_state_check.board[2][1].set_owner((1, 0))
        setup_board_win_state_check.board[1][2].set_owner((1, 0))
        setup_board_win_state_check.board[2][2].set_owner((1, 0))
        setup_board_win_state_check.board[3][1].set_owner((1, 0))

        is_win = setup_board_win_state_check.check_winning_state_player1()

        self.assertTrue(is_win)

    def test_win_state_player2(self):
        setup_board_win_state_check = Board(4)

        setup_board_win_state_check.board[0][0].set_owner((0, 1))
        setup_board_win_state_check.board[1][0].set_owner((0, 1))
        setup_board_win_state_check.board[2][0].set_owner((0, 1))
        setup_board_win_state_check.board[1][1].set_owner((0, 1))
        setup_board_win_state_check.board[2][1].set_owner((0, 1))
        setup_board_win_state_check.board[1][2].set_owner((0, 1))
        setup_board_win_state_check.board[1][3].set_owner((0, 1))
        setup_board_win_state_check.board[0][3].set_owner((0, 1))

        is_win = setup_board_win_state_check.check_winning_state_player2()

        self.assertTrue(is_win)

        setup_board_win_state_check = Board(4)

        setup_board_win_state_check.board[0][0].set_owner((0, 1))
        setup_board_win_state_check.board[1][0].set_owner((0, 1))
        setup_board_win_state_check.board[2][0].set_owner((0, 1))
        setup_board_win_state_check.board[1][1].set_owner((0, 1))
        setup_board_win_state_check.board[2][1].set_owner((0, 1))
        setup_board_win_state_check.board[1][2].set_owner((0, 1))
        setup_board_win_state_check.board[2][2].set_owner((0, 1))
        setup_board_win_state_check.board[3][2].set_owner((0, 1))

        is_win = setup_board_win_state_check.check_winning_state_player2()

        self.assertFalse(is_win)

    def test_conversion_to_diamond_shape(self):
        setup_board_win_state_check = Board(4)

        diamond_array = setup_board_win_state_check.convert_to_diamond_shape()

        self.assertEqual(diamond_array[0][0].get_position(), (0, 0))

        self.assertEqual(diamond_array[1][0].get_position(), (1, 0))
        self.assertEqual(diamond_array[1][1].get_position(), (0, 1))

        self.assertEqual(diamond_array[2][0].get_position(), (2, 0))
        self.assertEqual(diamond_array[2][1].get_position(), (1, 1))
        self.assertEqual(diamond_array[2][2].get_position(), (0, 2))

        self.assertEqual(diamond_array[3][0].get_position(), (3, 0))
        self.assertEqual(diamond_array[3][1].get_position(), (2, 1))
        self.assertEqual(diamond_array[3][2].get_position(), (1, 2))
        self.assertEqual(diamond_array[3][3].get_position(), (0, 3))

        self.assertEqual(diamond_array[4][0].get_position(), (3, 1))
        self.assertEqual(diamond_array[4][1].get_position(), (2, 2))
        self.assertEqual(diamond_array[4][2].get_position(), (1, 3))

        self.assertEqual(diamond_array[5][0].get_position(), (3, 2))
        self.assertEqual(diamond_array[5][1].get_position(), (2, 3))

        self.assertEqual(diamond_array[6][0].get_position(), (3, 3))

    def test_conversion_to_1D_array(self):
        board = Board(4)

        new_board = board.convert_to_1D_array()

        self.assertEqual(new_board.shape, (16,))

        self.assertEqual(new_board[0].get_position(), (0, 0))
        self.assertEqual(new_board[1].get_position(), (0, 1))
        self.assertEqual(new_board[2].get_position(), (0, 2))
        self.assertEqual(new_board[3].get_position(), (0, 3))
        self.assertEqual(new_board[4].get_position(), (1, 0))
        self.assertEqual(new_board[5].get_position(), (1, 1))
        self.assertEqual(new_board[6].get_position(), (1, 2))
        self.assertEqual(new_board[7].get_position(), (1, 3))
        self.assertEqual(new_board[8].get_position(), (2, 0))
        self.assertEqual(new_board[9].get_position(), (2, 1))
        self.assertEqual(new_board[10].get_position(), (2, 2))
        self.assertEqual(new_board[11].get_position(), (2, 3))
        self.assertEqual(new_board[12].get_position(), (3, 0))
        self.assertEqual(new_board[13].get_position(), (3, 1))
        self.assertEqual(new_board[14].get_position(), (3, 2))
        self.assertEqual(new_board[15].get_position(), (3, 3))


if __name__ == '__main__':
    unittest.main()
