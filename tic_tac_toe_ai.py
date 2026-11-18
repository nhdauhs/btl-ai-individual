"""Caro 5 quân: Minimax-Alpha-Beta và MCTS-UCT.

Chạy thử:
    python tic_tac_toe_ai.py --mode mcts --size 9 --iterations 1000
    python tic_tac_toe_ai.py --mode minimax --size 5 --depth 2

Quy ước: X là máy, O là đối thủ ngẫu nhiên. Có thể đổi chế độ để so sánh.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

EMPTY = "."
X = "X"
O = "O"
WIN_SCORE = 1_000_000
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass
class Board:
    size: int = 9
    cells: list[list[str]] = field(default_factory=list)
    player: str = X

    def __post_init__(self) -> None:
        if not self.cells:
            self.cells = [[EMPTY for _ in range(self.size)] for _ in range(self.size)]

    def copy(self) -> "Board":
        return Board(self.size, [row[:] for row in self.cells], self.player)

    def legal_moves(self) -> list[tuple[int, int]]:
        occupied = [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if self.cells[row][col] != EMPTY
        ]
        if not occupied:
            center = self.size // 2
            return [(center, center)]

        candidates: set[tuple[int, int]] = set()
        for row, col in occupied:
            for d_row in range(-2, 3):
                for d_col in range(-2, 3):
                    next_row, next_col = row + d_row, col + d_col
                    if (
                        0 <= next_row < self.size
                        and 0 <= next_col < self.size
                        and self.cells[next_row][next_col] == EMPTY
                    ):
                        candidates.add((next_row, next_col))
        return list(candidates)

    def play(self, move: tuple[int, int]) -> "Board":
        row, col = move
        if self.cells[row][col] != EMPTY:
            raise ValueError("Nước đi không hợp lệ")
        result = self.copy()
        result.cells[row][col] = self.player
        result.player = O if self.player == X else X
        return result

    def winner(self, last_move: Optional[tuple[int, int]] = None) -> Optional[str]:
        moves = [last_move] if last_move is not None else [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if self.cells[row][col] != EMPTY
        ]
        for row, col in moves:
            symbol = self.cells[row][col]
            if symbol == EMPTY:
                continue
            for d_row, d_col in DIRECTIONS:
                count = 1
                for sign in (-1, 1):
                    next_row = row + sign * d_row
                    next_col = col + sign * d_col
                    while (
                        0 <= next_row < self.size
                        and 0 <= next_col < self.size
                        and self.cells[next_row][next_col] == symbol
                    ):
                        count += 1
                        next_row += sign * d_row
                        next_col += sign * d_col
                if count >= 5:
                    return symbol
        return None

    def full(self) -> bool:
        return all(cell != EMPTY for row in self.cells for cell in row)

    def terminal(self, last_move: Optional[tuple[int, int]] = None) -> bool:
        return self.winner(last_move) is not None or self.full()

    def display(self) -> None:
        print("   " + " ".join(f"{col:2}" for col in range(self.size)))
        for row, values in enumerate(self.cells):
            print(f"{row:2} " + "  ".join(values))


def line_score(own: int, opponent: int) -> int:
    if own and opponent:
        return 0
    table = {0: 0, 1: 1, 2: 10, 3: 100, 4: 10_000, 5: WIN_SCORE}
    score = table.get(own or opponent, WIN_SCORE * 2)
    return score if own else -score


def evaluate(board: Board, maximizer: str) -> int:
    minimizer = O if maximizer == X else X
    winner = board.winner()
    if winner == maximizer:
        return WIN_SCORE
    if winner == minimizer:
        return -WIN_SCORE

    total = 0
    for row in range(board.size):
        for col in range(board.size):
            for d_row, d_col in DIRECTIONS:
                end_row, end_col = row + 4 * d_row, col + 4 * d_col
                if not (0 <= end_row < board.size and 0 <= end_col < board.size):
                    continue
                own = opponent = 0
                for offset in range(5):
                    cell = board.cells[row + offset * d_row][col + offset * d_col]
                    own += cell == maximizer
                    opponent += cell == minimizer
                total += line_score(own, opponent)
    return total


def minimax(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    maximizer: str,
    statistics: dict[str, int],
) -> float:
    statistics["nodes"] += 1
    if depth == 0 or board.terminal():
        return evaluate(board, maximizer)

    moves = board.legal_moves()
    if board.player == maximizer:
        value = -math.inf
        for move in moves:
            value = max(value, minimax(board.play(move), depth - 1, alpha, beta,
                                       maximizer, statistics))
            alpha = max(alpha, value)
            if alpha >= beta:
                statistics["cutoffs"] += 1
                break
        return value

    value = math.inf
    for move in moves:
        value = min(value, minimax(board.play(move), depth - 1, alpha, beta,
                                   maximizer, statistics))
        beta = min(beta, value)
        if alpha >= beta:
            statistics["cutoffs"] += 1
            break
    return value


def best_minimax_move(board: Board, depth: int, player: str) -> tuple[tuple[int, int], dict[str, int]]:
    statistics = {"nodes": 0, "cutoffs": 0}
    best_move = None
    best_value = -math.inf if board.player == player else math.inf
    for move in board.legal_moves():
        value = minimax(board.play(move), depth - 1, -math.inf, math.inf,
                        player, statistics)
        if (board.player == player and value > best_value) or (
            board.player != player and value < best_value
        ):
            best_value, best_move = value, move
    if best_move is None:
        raise RuntimeError("Không còn nước đi hợp lệ")
    return best_move, statistics


@dataclass
class MCTSNode:
    board: Board
    parent: Optional["MCTSNode"] = None
    move: Optional[tuple[int, int]] = None
    player_just_moved: Optional[str] = None
    children: list["MCTSNode"] = field(default_factory=list)
    untried_moves: list[tuple[int, int]] = field(default_factory=list)
    visits: int = 0
    wins: float = 0.0

    def __post_init__(self) -> None:
        if not self.untried_moves:
            self.untried_moves = self.board.legal_moves()

    def uct(self, exploration: float) -> float:
        if self.visits == 0:
            return math.inf
        parent_visits = max(self.parent.visits if self.parent else 1, 1)
        return self.wins / self.visits + exploration * math.sqrt(
            math.log(parent_visits) / self.visits
        )


def rollout(board: Board, heuristic: bool, rng: random.Random) -> Optional[str]:
    current = board
    while not current.terminal():
        moves = current.legal_moves()
        if heuristic:
            winning = []
            for move in moves:
                candidate = current.play(move)
                if candidate.winner(move) == current.player:
                    winning.append(move)
            if winning:
                moves = winning
            else:
                opponent = O if current.player == X else X
                blocks = []
                for move in moves:
                    # Test the threat on a temporary board where the opponent
                    # is the player to move.  Playing ``move`` as the current
                    # player can only detect our own win, so it never detects
                    # an opponent's immediate winning square.
                    threat = Board(
                        current.size,
                        [row[:] for row in current.cells],
                        opponent,
                    ).play(move)
                    if threat.winner(move) == opponent:
                        blocks.append(move)
                if blocks:
                    moves = blocks
        current = current.play(rng.choice(moves))
    return current.winner()


def best_mcts_move(
    board: Board,
    iterations: int,
    exploration: float,
    heuristic_rollout: bool,
    seed: Optional[int] = None,
) -> tuple[tuple[int, int], MCTSNode]:
    rng = random.Random(seed)
    root = MCTSNode(board)
    for _ in range(iterations):
        node = root
        while not node.untried_moves and node.children and not node.board.terminal():
            node = max(node.children, key=lambda child: child.uct(exploration))

        if node.untried_moves and not node.board.terminal():
            move = rng.choice(node.untried_moves)
            node.untried_moves.remove(move)
            child_board = node.board.play(move)
            node = MCTSNode(child_board, node, move, node.board.player)
            node.parent.children.append(node)

        winner = rollout(node.board, heuristic_rollout, rng)
        while node is not None:
            node.visits += 1
            if winner is None:
                node.wins += 0.5
            elif winner == node.player_just_moved:
                node.wins += 1
            node = node.parent

    if not root.children:
        raise RuntimeError("MCTS không tạo được nút con")
    return max(root.children, key=lambda child: child.visits).move, root


def random_move(board: Board, seed: Optional[int] = None) -> tuple[int, int]:
    return random.Random(seed).choice(board.legal_moves())


def play_game(args: argparse.Namespace) -> None:
    board = Board(args.size)
    while not board.terminal():
        board.display()
        print(f"\nLượt của {board.player}")
        start = time.perf_counter()
        if board.player == X:
            if args.mode == "minimax":
                move, stats = best_minimax_move(board, args.depth, X)
                print(f"Minimax: nodes={stats['nodes']}, cutoffs={stats['cutoffs']}")
            else:
                move, root = best_mcts_move(
                    board, args.iterations, args.exploration, args.heuristic, args.seed
                )
                print(f"MCTS: root visits={root.visits}, children={len(root.children)}")
        else:
            move = random_move(board, args.seed)
        print(f"Chọn: {move}, thời gian={time.perf_counter() - start:.3f}s")
        board = board.play(move)

    board.display()
    print("Kết quả:", board.winner() or "Hòa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caro 5 quân với Minimax hoặc MCTS")
    parser.add_argument("--mode", choices=("minimax", "mcts"), default="mcts")
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--exploration", type=float, default=math.sqrt(2))
    parser.add_argument("--heuristic", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.size < 5:
        parser.error("--size phải lớn hơn hoặc bằng 5")
    return args


if __name__ == "__main__":
    play_game(parse_args())
