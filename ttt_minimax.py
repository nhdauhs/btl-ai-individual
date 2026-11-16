"""Tic-Tac-Toe tối ưu bằng Minimax và Alpha-Beta pruning.
Chạy: python ttt_minimax.py --algorithm alphabeta --show-search
"""
from __future__ import annotations
import argparse
from functools import lru_cache
from math import inf
from typing import Optional

WIN_LINES = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))

def winner(s: str) -> Optional[str]:
    for a,b,c in WIN_LINES:
        if s[a] != '.' and s[a] == s[b] == s[c]: return s[a]
    return None

def terminal(s: str) -> bool: return winner(s) is not None or '.' not in s

def utility(s: str, root: str) -> int:
    w = winner(s)
    return 0 if w is None else (1 if w == root else -1)

def moves(s: str): return (i for i, cell in enumerate(s) if cell == '.')

def put(s: str, i: int, p: str) -> str: return s[:i] + p + s[i+1:]

def minimax(s: str, player: str, root: str, stats: dict) -> int:
    stats['nodes'] += 1
    if terminal(s): return utility(s, root)
    vals = [minimax(put(s, i, player), 'O' if player == 'X' else 'X', root, stats) for i in moves(s)]
    return (max(vals) if player == root else min(vals))

def alphabeta(s: str, player: str, root: str, alpha: int, beta: int, stats: dict) -> int:
    stats['nodes'] += 1
    if terminal(s): return utility(s, root)
    if player == root:
        value = -inf
        for i in moves(s):
            value = max(value, alphabeta(put(s,i,player), 'O' if player=='X' else 'X', root, alpha, beta, stats))
            alpha = max(alpha, value)
            if alpha >= beta:
                stats['cutoffs'] += 1; break
        return value
    value = inf
    for i in moves(s):
        value = min(value, alphabeta(put(s,i,player), 'O' if player=='X' else 'X', root, alpha, beta, stats))
        beta = min(beta, value)
        if alpha >= beta:
            stats['cutoffs'] += 1; break
    return value

def best_move(s: str, root: str, algorithm: str, trace: bool = False):
    stats = {'nodes': 0, 'cutoffs': 0}; best = None; best_value = -inf
    for i in moves(s):
        child = put(s, i, root)
        value = (minimax(child, 'O' if root=='X' else 'X', root, stats)
                 if algorithm == 'minimax' else alphabeta(child, 'O' if root=='X' else 'X', root, -1, 1, stats))
        if trace: print(f'  thử ô {i+1}: giá trị {value}')
        if value > best_value: best_value, best = value, i
    return best, best_value, stats

def print_board(s):
    print('\n'.join(' | '.join(s[3*r:3*r+3]) for r in range(3)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--algorithm', choices=['minimax','alphabeta'], default='alphabeta'); ap.add_argument('--show-search', action='store_true'); args = ap.parse_args()
    board = '.........'; player = 'X'
    print('Ô được đánh số 1..9 theo thứ tự trái-phải, trên-xuống. Máy là X.')
    while not terminal(board):
        print_board(board)
        if player == 'X':
            move, value, stats = best_move(board, player, args.algorithm, args.show_search)
            print(f'Máy chọn ô {move+1}; giá trị minimax={value}; nút={stats["nodes"]}; cắt={stats["cutoffs"]}')
        else:
            raw = input('Bạn (O), chọn ô: '); move = int(raw)-1
            if move not in list(moves(board)): print('Ô không hợp lệ.'); continue
        board = put(board, move, player); player = 'O' if player == 'X' else 'X'
    print_board(board); print('Kết quả:', winner(board) or 'Hòa')

if __name__ == '__main__': main()
