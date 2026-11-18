import unittest

from tic_tac_toe_ai import Board, O, X, rollout, best_mcts_move
from ttt_minimax import best_move


class FixedChoice:
    def __init__(self):
        self.calls = []

    def choice(self, values):
        self.calls.append(tuple(values))
        return values[0]


class GameRulesTest(unittest.TestCase):
    def test_five_in_a_row_in_all_directions(self):
        for positions in (
            [(2, c) for c in range(5)],
            [(r, 2) for r in range(5)],
            [(r, r) for r in range(5)],
            [(r, 4 - r) for r in range(5)],
        ):
            board = Board(5)
            for row, col in positions:
                board.cells[row][col] = X
            self.assertEqual(board.winner(), X)

    def test_heuristic_rollout_blocks_immediate_threat(self):
        board = Board(5, player=X)
        board.cells[2][0:4] = [O] * 4
        board.cells[0][0] = X
        rng = FixedChoice()
        rollout(board, heuristic=True, rng=rng)
        self.assertEqual(rng.calls[0], ((2, 4),))

    def test_mcts_visits_root_budget(self):
        _, root = best_mcts_move(Board(5), 25, 1.4, False, seed=4)
        self.assertEqual(root.visits, 25)


class ExactTicTacToeTest(unittest.TestCase):
    def test_minimax_and_alpha_beta_agree(self):
        minimax_move, value, minimax_stats = best_move(".........", X, "minimax")
        alpha_move, alpha_value, alpha_stats = best_move(".........", X, "alphabeta")
        self.assertEqual(value, alpha_value)
        self.assertEqual(value, 0)
        self.assertEqual(minimax_move, alpha_move)
        self.assertLess(alpha_stats["nodes"], minimax_stats["nodes"])
        self.assertGreater(alpha_stats["cutoffs"], 0)


if __name__ == "__main__":
    unittest.main()
