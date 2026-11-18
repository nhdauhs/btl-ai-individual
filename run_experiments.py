"""Reproducible experiments for the Caro MCTS implementation.

Examples:
    python run_experiments.py --games 50 --iterations 500
    python run_experiments.py --games 50 --iterations 100 --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

from tic_tac_toe_ai import Board, O, X, best_mcts_move


def play_match(size: int, iterations: int, exploration: float,
               heuristic: bool, seed: int) -> tuple[str, float, int]:
    board = Board(size)
    rng = random.Random(seed)
    elapsed = 0.0
    simulations = 0
    while not board.terminal():
        if board.player == X:
            start = time.perf_counter()
            move, root = best_mcts_move(
                board, iterations, exploration, heuristic, rng.randrange(2**32)
            )
            elapsed += time.perf_counter() - start
            simulations += root.visits
        else:
            move = rng.choice(board.legal_moves())
        board = board.play(move)
    return board.winner() or "draw", elapsed, simulations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--iterations", type=int, nargs="+", default=[100, 500])
    parser.add_argument("--c", type=float, nargs="+", default=[0.5, 1.0, 1.4, 2.0])
    parser.add_argument("--rollouts", choices=("random", "heuristic", "both"), default="both")
    parser.add_argument("--csv", type=Path, default=Path("experiment_results.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for iterations in args.iterations:
        for c in args.c:
            heuristic_values = {
                "random": (False,), "heuristic": (True,), "both": (False, True)
            }[args.rollouts]
            for heuristic in heuristic_values:
                counts = {X: 0, O: 0, "draw": 0}
                elapsed = simulations = 0.0
                for game in range(args.games):
                    result, seconds, sims = play_match(
                        args.size, iterations, c, heuristic, 20261118 + game
                    )
                    counts[result] += 1
                    elapsed += seconds
                    simulations += sims
                rows.append({
                    "size": args.size, "games": args.games,
                    "iterations": iterations, "c": c,
                    "rollout": "heuristic" if heuristic else "random",
                    "wins": counts[X], "draws": counts["draw"],
                    "losses": counts[O],
                    "win_rate": counts[X] / args.games,
                    "mean_seconds_per_game": elapsed / args.games,
                    "mean_simulations": simulations / args.games,
                })
                print(rows[-1])
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
