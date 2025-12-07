#!/usr/bin/env python3
"""
Quick test to see if MinimaxEngine makes reasonable moves
"""
import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "backend_engine"))

from engines.minimax_engine import MinimaxEngine
import chess

print("=== MINIMAX ENGINE SANITY CHECK ===\n")

# Initialize engine
engine = MinimaxEngine(use_nn=False, max_depth=3)

# Test 1: Can it capture a free piece?
print("TEST 1: Free Queen")
print("Can the engine capture a free queen?")
# Position: White to move, Black queen on e5 undefended
board = chess.Board("rnb1kbnr/pppppppp/8/4q3/8/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
print(f"Position: {board.fen()}")
print(f"Visual:")
print(board)
move = engine.get_move(board.fen(), time_limit=5)
print(f"Engine move: {move}")
expected = chess.Move.from_uci("d1e5") # Queen takes queen
if move == expected.uci():
    print("✓ PASS: Captured the free queen!\n")
else:
    print(f"✗ FAIL: Should capture queen with Qxe5, but played {move}\n")

# Test 2: Can it avoid hanging its own queen?
print("TEST 2: Don't Hang Queen")
print("Will the engine avoid moving queen to a square where it can be captured?")
# Starting position - queen should not move to e5 where it can be taken by pawn
board = chess.Board()
move = engine.get_move(board.fen(), time_limit=5)
move_obj = chess.Move.from_uci(move)
print(f"Engine move: {move}")
# Check if queen moved to e5 (bad move)
if move == "d1e5":
    print("✗ FAIL: Engine hung its queen on e5!\n")
else:
    print(f"✓ PASS: Played a reasonable move ({move})\n")

# Test 3: Checkmate in 1
print("TEST 3: Checkmate in 1")
print("Can the engine find mate in 1 move?")
# Back rank mate: Qd8# is mate
board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1")
print(f"Visual:")
print(board)
move = engine.get_move(board.fen(), time_limit=5)
print(f"Engine move: {move}")
if move == "e1e8":
    print("✓ PASS: Found checkmate!\n")
else:
    print(f"✗ FAIL: Should play Qe8# but played {move}\n")

# Test 4: Material evaluation
print("TEST 4: Material Evaluation")
print("Testing evaluation function...")
board = chess.Board()  # Starting position
score = engine.evaluate_board(board)
print(f"Starting position score: {score}")
if score == 0:
    print("✓ PASS: Equal material = 0 score\n")
else:
    print(f"✗ FAIL: Should be 0 but got {score}\n")

# Test 5: White up a queen
board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
score = engine.evaluate_board(board)
print(f"White up a queen: {score}")
if score > 800:  # Should be +900 for queen
    print("✓ PASS: Positive score when White is winning\n")
else:
    print(f"✗ FAIL: Should be ~900 but got {score}\n")

# Test 6: Black up a queen  
board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
score = engine.evaluate_board(board)
print(f"Black up a queen: {score}")
if score < -800:  # Should be -900 for Black's advantage
    print("✓ PASS: Negative score when Black is winning\n")
else:
    print(f"✗ FAIL: Should be ~-900 but got {score}\n")

print("=== TESTS COMPLETE ===")
