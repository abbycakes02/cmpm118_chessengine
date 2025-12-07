#!/usr/bin/env python3
"""
Debug test to see WHY engine doesn't capture free queen
"""
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "backend_engine"))

from engines.minimax_engine import MinimaxEngine
import chess

engine = MinimaxEngine(use_nn=False, max_depth=2)

# Position: White to move, Black queen on e5 undefended
board = chess.Board("rnb1kbnr/pppppppp/8/4q3/8/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 1")

print("Position:")
print(board)
print()

# Manually evaluate a few moves
test_moves = [
    "d1e5",  # Capture queen - should be BEST (+900)
    "f1c4",  # Develop bishop - OK but misses queen
    "g1h3",  # Develop knight - OK but misses queen
    "e3e4",  # Push pawn - neutral
]

print("Manual evaluation of moves:")
for move_str in test_moves:
    move = chess.Move.from_uci(move_str)
    if move in board.legal_moves:
        board.push(move)
        score = engine.evaluate_board(board)
        print(f"{move_str:6s}: eval = {score:6d}  {'(TAKES QUEEN!)' if 'e5' in move_str and move_str[0] == 'd' else ''}")
        board.pop()

print("\nNow let's see what the minimax search finds...")
print("=" * 60)

# Run depth 1 search manually
legal_moves = list(board.legal_moves)
print(f"\nDepth 1 search (White maximizing):")
max_eval = -float('inf')
best_move = None

for move in legal_moves[:5]:  # Just first 5 for brevity
    board.push(move)
    # After White moves, it's Black's turn (minimizing = False)
    eval_score = engine.minimax(board, 0, -float('inf'), float('inf'), False, None)
    board.pop()
    
    print(f"  {move.uci():6s}: {eval_score:6d}")
    
    if eval_score > max_eval:
        max_eval = eval_score
        best_move = move

print(f"\nBest move at depth 1: {best_move.uci() if best_move else 'None'} (score: {max_eval})")
