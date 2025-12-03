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
import chess.pgn
from pathlib import Path
from datetime import datetime

# Add backend_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend_engine"))

from engines.minimax_engine import MinimaxEngine


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
              stockfish_path: str, base_time: float = 300.0,
              increment: float = 5.0, engine_plays_white: bool = True) -> tuple[str, chess.pgn.Game]:
    """
    Play a single game between the minimax engine and Stockfish with tournament time controls.

    Args:
        engine_checkpoint: Path to the model checkpoint file
        stockfish_level: Stockfish skill level (1-20)
        stockfish_path: Path to Stockfish binary
        base_time: Base time per player in seconds (e.g., 300 = 5 minutes)
        increment: Increment per move in seconds (e.g., 5)
        engine_plays_white: If True, minimax plays white; otherwise black

    Returns:
        Tuple of (result string, pgn.Game object)
        Result: "1-0" (white wins), "0-1" (black wins), or "1/2-1/2" (draw)
    """
    import time

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Checkpoint Benchmark"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = f"MinimaxBot-{engine_checkpoint.stem}" if engine_plays_white else f"Stockfish-L{stockfish_level}"
    game.headers["Black"] = f"Stockfish-L{stockfish_level}" if engine_plays_white else f"MinimaxBot-{engine_checkpoint.stem}"
    game.headers["TimeControl"] = f"{int(base_time)}+{int(increment)}"

    node = game
    stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Configure Stockfish skill level
    stockfish.configure({"Skill Level": stockfish_level})

    # Initialize minimax engine with neural network
    minimax = MinimaxEngine(
        use_nn=True,
        model_path=str(engine_checkpoint),
        channels=64 if "64ch" in engine_checkpoint.stem else 32,
        blocks=3,
        max_depth=3,
        max_time=None
    )

    # Time tracking
    white_time = base_time
    black_time = base_time

    move_count = 0
    max_moves = 300  # Prevent infinite games
    game_start = time.time()

    try:
        while not board.is_game_over() and move_count < max_moves:
            move_start = time.time()
            current_time = white_time if board.turn == chess.WHITE else black_time

            if (board.turn == chess.WHITE and engine_plays_white) or \
               (board.turn == chess.BLACK and not engine_plays_white):
                # Minimax engine's turn
                try:
                    # Check if engine has time
                    if current_time <= 0:
                        result = "0-1" if engine_plays_white else "1-0"
                        game.headers["Result"] = result
                        game.headers["Termination"] = "time forfeit"
                        return result, game

                    # Allocate time smartly: use 1/30th of remaining time per move
                    # This ensures we don't run out of time in long games
                    move_time_alloc = max(0.5, min(current_time / 30.0, current_time - 5.0))

                    move_uci = minimax.get_move(
                        board.fen(),
                        time_limit=move_time_alloc,
                        max_depth=3
                    )
                    move = chess.Move.from_uci(move_uci)
                    board.push(move)
                    node = node.add_variation(move)

                except Exception as e:
                    print(f"    Error in minimax engine: {e}")
                    result = "0-1" if engine_plays_white else "1-0"
                    game.headers["Result"] = result
                    game.headers["Termination"] = "engine error"
                    return result, game
            else:
                # Stockfish's turn
                result = stockfish.play(board, chess.engine.Limit(time=current_time))
                board.push(result.move)
                node = node.add_variation(result.move)

            # Update time
            move_time = time.time() - move_start
            if board.turn == chess.BLACK:  # White just moved
                white_time = white_time - move_time + increment
            else:  # Black just moved
                black_time = black_time - move_time + increment

            move_count += 1

        # Determine result
        if board.is_checkmate():
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            game.headers["Termination"] = "checkmate"
        elif board.is_stalemate():
            result = "1/2-1/2"
            game.headers["Termination"] = "stalemate"
        elif board.is_insufficient_material():
            result = "1/2-1/2"
            game.headers["Termination"] = "insufficient material"
        elif board.can_claim_draw():
            result = "1/2-1/2"
            game.headers["Termination"] = "draw by repetition/50-move"
        elif move_count >= max_moves:
            result = "1/2-1/2"
            game.headers["Termination"] = "max moves"
        else:
            result = "1/2-1/2"
            game.headers["Termination"] = "unknown"

        game.headers["Result"] = result
        game.headers["PlyCount"] = str(move_count)

        return result, game

    finally:
        stockfish.quit()


def benchmark_checkpoint(checkpoint: Path, stockfish_path: str,
                        stockfish_levels: list[int], games_per_level: int = 5,
                        base_time: float = 300.0, increment: float = 5.0,
                        output_dir: Path = None) -> dict:
    """
    Benchmark a single checkpoint against multiple Stockfish levels.

    Args:
        checkpoint: Path to model checkpoint
        stockfish_path: Path to Stockfish binary
        stockfish_levels: List of Stockfish skill levels to test against
        games_per_level: Number of games to play at each level
        base_time: Base time per side in seconds
        increment: Increment per move in seconds
        output_dir: Directory to save PGN files

    Returns:
        Dictionary with benchmark results
    """
    results = {
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.stem,
        "timestamp": datetime.now().isoformat(),
        "time_control": f"{int(base_time)}+{int(increment)}",
        "levels": {}
    }

    print(f"\nBenchmarking: {checkpoint.name}")
    print("=" * 60)

    for level in stockfish_levels:
        print(f"\n  Testing against Stockfish Level {level}...")
        wins = 0
        losses = 0
        draws = 0
        games = []

        for game_num in range(games_per_level):
            # Alternate colors
            engine_plays_white = (game_num % 2 == 0)
            color = "White" if engine_plays_white else "Black"

            print(f"    Game {game_num + 1}/{games_per_level} (Engine plays {color})...", end=" ")

            result, pgn_game = play_game(
                checkpoint, level, stockfish_path,
                base_time, increment, engine_plays_white
            )

            pgn_game.headers["Round"] = str(game_num + 1)
            games.append(pgn_game)

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

        # Save PGN file for this level
        if output_dir:
            pgn_filename = f"{checkpoint.stem}_vs_stockfish_l{level}.pgn"
            pgn_path = output_dir / pgn_filename
            with open(pgn_path, "w") as f:
                for game in games:
                    print(game, file=f, end="\n\n")
            print(f"    PGN saved: {pgn_filename}")

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
    pgn_output_dir = project_root / "elo_rating"

    print("=" * 60)
    print("Model Checkpoint Benchmarking Tool")
    print("Time Controls: 300+5 (5min + 5sec increment)")
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

    # Checkpoint selection
    print("\nSelect checkpoints to test:")
    print("  Enter numbers (comma-separated, e.g., '1,3,8')")
    print("  Or press Enter to use recommended: 8,3,1")
    print("    (64ch best, 32ch best, baseline)")
    selection_input = input("  Selection: ").strip()

    if not selection_input:
        # Default to recommended: 64ch best first, then 32ch, then baseline
        selected_indices = [8, 3, 1]  # 64ch best, 32ch best, baseline
    else:
        selected_indices = [int(x.strip()) for x in selection_input.split(",")]

    checkpoints = [checkpoints[i-1] for i in selected_indices if 1 <= i <= len(checkpoints)]

    if not checkpoints:
        print("No valid checkpoints selected!")
        return

    print(f"\nSelected {len(checkpoints)} checkpoint(s):")
    for i, cp in enumerate(checkpoints, 1):
        print(f"  {i}. {cp.name}")

    # Configuration
    print("\nConfiguration Options:")
    stockfish_path = input("  Stockfish path (default: /opt/homebrew/bin/stockfish): ").strip()
    if not stockfish_path:
        stockfish_path = "/opt/homebrew/bin/stockfish"

    print("\n  Stockfish levels to test against:")
    print("    Your existing benchmarks used levels 1, 2, and 3")
    levels_input = input("  Levels (comma-separated, default: 1,2,3): ").strip()
    if not levels_input:
        stockfish_levels = [1, 2, 3]
    else:
        stockfish_levels = [int(x.strip()) for x in levels_input.split(",")]

    games_per_level_input = input("  Games per level (default: 6, must be even): ").strip()
    games_per_level = int(games_per_level_input) if games_per_level_input else 6
    if games_per_level % 2 != 0:
        games_per_level += 1
        print(f"    Adjusted to {games_per_level} for equal color distribution")

    # Time controls
    print("\n  Time control (default: 300+5):")
    base_time_input = input("    Base time per side in seconds (default: 300): ").strip()
    base_time = float(base_time_input) if base_time_input else 300.0

    increment_input = input("    Increment per move in seconds (default: 5): ").strip()
    increment = float(increment_input) if increment_input else 5.0

    print(f"\n{'='*60}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Stockfish: {stockfish_path}")
    print(f"  Levels: {stockfish_levels}")
    print(f"  Games per level: {games_per_level}")
    print(f"  Time control: {int(base_time)}+{int(increment)}")
    print(f"  Checkpoints to test: {len(checkpoints)}")
    print(f"  Total games: {len(checkpoints) * len(stockfish_levels) * games_per_level}")
    print(f"  PGN files will be saved to: {pgn_output_dir}")

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
                games_per_level, base_time, increment, pgn_output_dir
            )
            all_results.append(results)
        except Exception as e:
            print(f"\nError benchmarking {checkpoint.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = pgn_output_dir / f"benchmark_results_{timestamp}.json"
    save_results(all_results, output_file)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for result in all_results:
        checkpoint_name = result["checkpoint_name"]
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
