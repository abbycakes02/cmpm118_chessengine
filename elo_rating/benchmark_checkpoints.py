#!/usr/bin/env python3
"""
Benchmark multiple model checkpoints against Stockfish.

This script tests different model checkpoints by having the minimax engine
(configured with each checkpoint) play against Stockfish at various skill levels.
Results are saved and can be used to calculate ELO ratings.

Usage:
    python benchmark_checkpoints.py
"""

import os
import sys
import json
import chess
import chess.engine
from pathlib import Path
from datetime import datetime

# Add backend_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend_engine"))

from engines.minimax_engine import get_best_move


def find_checkpoint_files(base_path: Path):
    """Find all .pth checkpoint files in the models directory."""
    checkpoints = []
    models_dir = base_path / "backend_engine" / "ml" / "models"

    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return checkpoints

    for checkpoint_file in models_dir.rglob("*.pth"):
        # Skip non-epoch checkpoints or other utility files
        if checkpoint_file.stem.startswith("epoch") or "ch_" in checkpoint_file.stem:
            checkpoints.append(checkpoint_file)

    return sorted(checkpoints)


def play_game(engine_checkpoint: Path, stockfish_level: int,
              stockfish_path: str, time_per_move: float = 1.0,
              engine_plays_white: bool = True) -> str:
    """
    Play a single game between the minimax engine and Stockfish.

    Args:
        engine_checkpoint: Path to the model checkpoint file
        stockfish_level: Stockfish skill level (1-20)
        stockfish_path: Path to Stockfish binary
        time_per_move: Time limit per move in seconds
        engine_plays_white: If True, minimax plays white; otherwise black

    Returns:
        Result string: "1-0" (white wins), "0-1" (black wins), or "1/2-1/2" (draw)
    """
    board = chess.Board()
    stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Configure Stockfish skill level
    stockfish.configure({"Skill Level": stockfish_level})

    move_count = 0
    max_moves = 200  # Prevent infinite games

    try:
        while not board.is_game_over() and move_count < max_moves:
            if (board.turn == chess.WHITE and engine_plays_white) or \
               (board.turn == chess.BLACK and not engine_plays_white):
                # Minimax engine's turn
                try:
                    move = get_best_move(
                        board.fen(),
                        time_limit=time_per_move,
                        model_path=str(engine_checkpoint)
                    )
                    board.push_uci(move)
                except Exception as e:
                    print(f"  Error in minimax engine: {e}")
                    # If engine fails, it loses
                    return "0-1" if engine_plays_white else "1-0"
            else:
                # Stockfish's turn
                result = stockfish.play(board, chess.engine.Limit(time=time_per_move))
                board.push(result.move)

            move_count += 1

        # Determine result
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return "0-1"  # Black wins
            else:
                return "1-0"  # White wins
        elif board.is_game_over():
            return "1/2-1/2"  # Draw
        else:
            # Max moves reached
            return "1/2-1/2"

    finally:
        stockfish.quit()


def benchmark_checkpoint(checkpoint: Path, stockfish_path: str,
                        stockfish_levels: list[int], games_per_level: int = 5,
                        time_per_move: float = 1.0) -> dict:
    """
    Benchmark a single checkpoint against multiple Stockfish levels.

    Args:
        checkpoint: Path to model checkpoint
        stockfish_path: Path to Stockfish binary
        stockfish_levels: List of Stockfish skill levels to test against
        games_per_level: Number of games to play at each level
        time_per_move: Time limit per move

    Returns:
        Dictionary with benchmark results
    """
    results = {
        "checkpoint": str(checkpoint),
        "timestamp": datetime.now().isoformat(),
        "levels": {}
    }

    print(f"\nBenchmarking: {checkpoint.name}")
    print("=" * 60)

    for level in stockfish_levels:
        print(f"\n  Testing against Stockfish Level {level}...")
        wins = 0
        losses = 0
        draws = 0

        for game_num in range(games_per_level):
            # Alternate colors
            engine_plays_white = (game_num % 2 == 0)
            color = "White" if engine_plays_white else "Black"

            print(f"    Game {game_num + 1}/{games_per_level} (Engine plays {color})...", end=" ")

            result = play_game(
                checkpoint, level, stockfish_path,
                time_per_move, engine_plays_white
            )

            if result == "1-0":
                if engine_plays_white:
                    wins += 1
                    print("WIN")
                else:
                    losses += 1
                    print("LOSS")
            elif result == "0-1":
                if engine_plays_white:
                    losses += 1
                    print("LOSS")
                else:
                    wins += 1
                    print("WIN")
            else:
                draws += 1
                print("DRAW")

        results["levels"][level] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "games": games_per_level
        }

        score_pct = (wins + 0.5 * draws) / games_per_level * 100
        print(f"    Result: {wins}W-{losses}L-{draws}D ({score_pct:.1f}%)")

    return results


def save_results(results: list[dict], output_file: Path):
    """Save benchmark results to JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


def main():
    """Main benchmarking function."""
    project_root = Path(__file__).parent.parent

    print("=" * 60)
    print("Model Checkpoint Benchmarking Tool")
    print("=" * 60)

    # Find checkpoints
    checkpoints = find_checkpoint_files(project_root)

    if not checkpoints:
        print("\nNo checkpoint files found!")
        print(f"Expected location: {project_root / 'backend_engine' / 'ml' / 'models'}")
        return

    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    for i, cp in enumerate(checkpoints, 1):
        print(f"  {i}. {cp.relative_to(project_root)}")

    # Configuration
    stockfish_path = input("\nEnter path to Stockfish binary (or 'stockfish' if in PATH): ").strip()
    if not stockfish_path:
        stockfish_path = "stockfish"

    print("\nStockfish levels to test (comma-separated, e.g., '1,2,3'):")
    levels_input = input("Levels: ").strip()
    stockfish_levels = [int(x.strip()) for x in levels_input.split(",")]

    games_per_level = int(input("Games per level (default 5): ").strip() or "5")
    time_per_move = float(input("Time per move in seconds (default 1.0): ").strip() or "1.0")

    print(f"\nConfiguration:")
    print(f"  Stockfish: {stockfish_path}")
    print(f"  Levels: {stockfish_levels}")
    print(f"  Games per level: {games_per_level}")
    print(f"  Time per move: {time_per_move}s")

    proceed = input("\nProceed with benchmarking? (y/n): ").strip().lower()
    if proceed != "y":
        print("Benchmarking cancelled.")
        return

    # Run benchmarks
    all_results = []

    for checkpoint in checkpoints:
        try:
            results = benchmark_checkpoint(
                checkpoint, stockfish_path, stockfish_levels,
                games_per_level, time_per_move
            )
            all_results.append(results)
        except Exception as e:
            print(f"\nError benchmarking {checkpoint.name}: {e}")
            continue

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = project_root / "elo_rating" / f"benchmark_results_{timestamp}.json"
    save_results(all_results, output_file)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for result in all_results:
        checkpoint_name = Path(result["checkpoint"]).name
        print(f"\n{checkpoint_name}:")

        for level, stats in result["levels"].items():
            wins = stats["wins"]
            losses = stats["losses"]
            draws = stats["draws"]
            games = stats["games"]
            score_pct = (wins + 0.5 * draws) / games * 100

            print(f"  Level {level}: {wins}W-{losses}L-{draws}D ({score_pct:.1f}%)")

    print("\n" + "=" * 60)
    print("Next step: Use calculate_elo.py to compute ELO ratings from these results")
    print("=" * 60)


if __name__ == "__main__":
    main()
