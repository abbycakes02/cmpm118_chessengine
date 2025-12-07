import sys
import os
import chess
import chess.engine
import chess.pgn
import math
import shutil
import time
import json
from datetime import datetime

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "backend_engine"))

from engines.minimax_engine import MinimaxEngine


# --- COLAB-SPECIFIC CONFIGURATION ---
# Save results to Google Drive for persistence
SAVE_TO_DRIVE = True
DRIVE_SAVE_DIR = "/content/drive/MyDrive/chess_results"
CHECKPOINT_FILE = "tournament_checkpoint.json"

# --- ENGINE CONFIGURATION ---
STOCKFISH_PATH = shutil.which("stockfish")
USE_NN = True
MODEL_PATH = os.path.join(project_root, "backend_engine", "ml", "models", "session_1764582902", "chess_valuenet_32ch_3resblocks.pth")
MAX_DEPTH = 5   # Deeper search possible with GPU
TIME_LIMIT = 2  # Seconds per move
BATCH_SIZE = None  # Auto-detect (will use 64 on Colab GPU)

# --- TOURNAMENT CONFIGURATION ---
GAMES_PER_LEVEL = 5
STOCKFISH_LEVELS = [1, 3, 4]
SAVE_PGN = True
PGN_OUTPUT = os.path.join(current_dir, "minimax_vs_stockfish_colab.pgn")

# ELO mapping for Stockfish levels
STOCKFISH_ELO_MAP = {
    0: 1320, 1: 1400, 2: 1500, 3: 1600, 4: 1700,
    5: 1850, 6: 2000, 7: 2150, 8: 2300, 9: 2400, 10: 2500
}


def save_checkpoint(checkpoint_data, filepath):
    """Save tournament checkpoint to file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"  Checkpoint saved: {filepath}")
    except Exception as e:
        print(f"  Warning: Could not save checkpoint: {e}")


def load_checkpoint(filepath):
    """Load tournament checkpoint from file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load checkpoint: {e}")
    return None


def get_stockfish_engine(skill_level):
    """Initialize Stockfish engine at specified skill level"""
    if not STOCKFISH_PATH:
        raise FileNotFoundError("Stockfish not found. Install: apt-get install stockfish")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Skill Level": skill_level})
    return engine


def play_game(my_engine, sf_engine, i_am_white, time_limit):
    """Play a single game between engines"""
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "MinimaxEngine" if i_am_white else f"Stockfish-L{sf_engine.options.get('Skill Level', 0)}"
    game.headers["Black"] = f"Stockfish-L{sf_engine.options.get('Skill Level', 0)}" if i_am_white else "MinimaxEngine"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    node = game

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            if i_am_white:
                try:
                    move_uci = my_engine.get_move(board.fen(), time_limit=time_limit, max_depth=MAX_DEPTH)
                    move = chess.Move.from_uci(move_uci)
                    if move not in board.legal_moves:
                        print(f"Error: Engine returned illegal move {move_uci}")
                        return 0.0, None
                    board.push(move)
                    node = node.add_variation(move)
                except Exception as e:
                    print(f"Engine error: {e}")
                    return 0.0, None
            else:
                result = sf_engine.play(board, chess.engine.Limit(time=time_limit))
                board.push(result.move)
                node = node.add_variation(result.move)
        else:
            if not i_am_white:
                try:
                    move_uci = my_engine.get_move(board.fen(), time_limit=time_limit, max_depth=MAX_DEPTH)
                    move = chess.Move.from_uci(move_uci)
                    if move not in board.legal_moves:
                        print(f"Error: Engine returned illegal move {move_uci}")
                        return 0.0, None
                    board.push(move)
                    node = node.add_variation(move)
                except Exception as e:
                    print(f"Engine error: {e}")
                    return 0.0, None
            else:
                result = sf_engine.play(board, chess.engine.Limit(time=time_limit))
                board.push(result.move)
                node = node.add_variation(result.move)

    # Game over
    result = board.result()
    game.headers["Result"] = result

    if result == "1-0":
        score = 1.0 if i_am_white else 0.0
    elif result == "0-1":
        score = 0.0 if i_am_white else 1.0
    else:
        score = 0.5

    return score, game


def calculate_performance_rating(score, total_games, opponent_elo):
    """Calculate performance rating from score"""
    percentage = score / total_games

    if percentage >= 0.99:
        return opponent_elo + 400
    if percentage <= 0.01:
        return opponent_elo - 400

    diff = -400 * math.log10(1 / percentage - 1)
    return opponent_elo + diff


def calculate_elo_error_margin(games_played, win_rate=0.5):
    """Calculate 95% confidence interval for ELO rating"""
    std_error = math.sqrt(win_rate * (1 - win_rate) / games_played)
    elo_uncertainty = 1.96 * 173.7 * std_error
    return elo_uncertainty


def main():
    print("="*70)
    print("CHESS ENGINE ELO TOURNAMENT - GOOGLE COLAB VERSION")
    print("="*70)

    # Setup save directory
    if SAVE_TO_DRIVE and os.path.exists("/content/drive"):
        os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
        checkpoint_path = os.path.join(DRIVE_SAVE_DIR, CHECKPOINT_FILE)
        print(f"📁 Saving to Google Drive: {DRIVE_SAVE_DIR}")
    else:
        checkpoint_path = os.path.join(current_dir, CHECKPOINT_FILE)
        print(f"📁 Saving locally: {current_dir}")

    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        print(f"✅ Resuming from checkpoint: {len(checkpoint['results'])} games completed")
        overall_results = checkpoint['results']
        overall_opponent_elos = checkpoint['opponent_elos']
        all_games_pgn = []  # PGN not saved in checkpoint
        completed_levels = checkpoint.get('completed_levels', [])
    else:
        overall_results = []
        overall_opponent_elos = []
        all_games_pgn = []
        completed_levels = []

    # Initialize engine
    print(f"\n🚀 Initializing MinimaxEngine (NN={USE_NN}, Depth={MAX_DEPTH}, Batch Size={BATCH_SIZE})")
    try:
        if USE_NN:
            my_engine = MinimaxEngine(
                use_nn=True,
                model_path=MODEL_PATH,
                channels=32,
                blocks=3,
                max_depth=MAX_DEPTH,
                batch_size=BATCH_SIZE  # Auto-detects 64 on Colab GPU
            )
        else:
            my_engine = MinimaxEngine(use_nn=False, max_depth=MAX_DEPTH)
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return

    if not STOCKFISH_PATH:
        print("❌ Stockfish not found. Install: apt-get install stockfish")
        return

    print(f"\n⚔️  Starting Tournament vs Stockfish")
    print(f"⏱️  Time per move: {TIME_LIMIT}s")
    print(f"{'-'*70}")
    print(f"| {'Level':<5} | {'SF ELO':<6} | {'Score':<10} | {'Perf Rating':<12} |")
    print(f"{'-'*70}")

    # Run tournament
    for level in STOCKFISH_LEVELS:
        # Skip if already completed
        if level in completed_levels:
            print(f"⏭️  Skipping Level {level} (already completed)")
            continue

        sf_elo = STOCKFISH_ELO_MAP.get(level, 1350 + level * 100)
        level_score = 0.0

        try:
            sf_engine = get_stockfish_engine(level)
        except Exception as e:
            print(f"❌ Could not start Stockfish: {e}")
            break

        print(f"\n📊 Level {level} (ELO {sf_elo}):")

        for i in range(GAMES_PER_LEVEL):
            i_am_white = (i % 2 == 0)

            print(f"  Game {i+1}/{GAMES_PER_LEVEL} ({'White' if i_am_white else 'Black'})...", end=" ", flush=True)

            game_score, pgn_game = play_game(my_engine, sf_engine, i_am_white, TIME_LIMIT)
            level_score += game_score

            # Print result
            if game_score == 1.0:
                print("✅ WIN")
            elif game_score == 0.0:
                print("❌ LOSS")
            else:
                print("🤝 DRAW")

            if pgn_game and SAVE_PGN:
                all_games_pgn.append(pgn_game)

            overall_results.append(game_score)
            overall_opponent_elos.append(sf_elo)

            # Save checkpoint after each game
            checkpoint_data = {
                'results': overall_results,
                'opponent_elos': overall_opponent_elos,
                'completed_levels': completed_levels,
                'last_updated': datetime.now().isoformat()
            }
            save_checkpoint(checkpoint_data, checkpoint_path)

        sf_engine.quit()

        # Mark level as completed
        completed_levels.append(level)

        # Calculate level performance
        perf = calculate_performance_rating(level_score, GAMES_PER_LEVEL, sf_elo)
        print(f"| {level:<5} | {sf_elo:<6} | {level_score}/{GAMES_PER_LEVEL:<8} | {int(perf):<12} |")

    # Save PGN file
    if SAVE_PGN and all_games_pgn:
        with open(PGN_OUTPUT, "w") as pgn_file:
            for game in all_games_pgn:
                print(game, file=pgn_file, end="\n\n")
        print(f"\n✅ Saved {len(all_games_pgn)} games to {PGN_OUTPUT}")

        # Also save to Drive if available
        if SAVE_TO_DRIVE and os.path.exists("/content/drive"):
            drive_pgn = os.path.join(DRIVE_SAVE_DIR, "minimax_vs_stockfish_colab.pgn")
            shutil.copy(PGN_OUTPUT, drive_pgn)
            print(f"✅ Saved to Drive: {drive_pgn}")

    # Calculate overall performance
    if overall_results:
        avg_opp_elo = sum(overall_opponent_elos) / len(overall_opponent_elos)
        total_score = sum(overall_results)
        total_games = len(overall_results)
        overall_perf = calculate_performance_rating(total_score, total_games, avg_opp_elo)
        win_rate = total_score / total_games
        error_margin = calculate_elo_error_margin(total_games, win_rate)

        print(f"\n{'-'*70}")
        print(f"🏆 FINAL RESULTS")
        print(f"{'-'*70}")
        print(f"Performance Rating: {int(overall_perf)} ± {int(error_margin)} ELO")
        print(f"Total Score: {total_score}/{total_games} ({100*win_rate:.1f}%)")
        print(f"95% Confidence: [{int(overall_perf - error_margin)}, {int(overall_perf + error_margin)}]")
        print(f"\n💡 Interpretation:")
        print(f"   Your engine's true strength is between {int(overall_perf - error_margin)}")
        print(f"   and {int(overall_perf + error_margin)} ELO (95% confidence)")

        # Save final results to text file
        results_file = os.path.join(DRIVE_SAVE_DIR if SAVE_TO_DRIVE and os.path.exists("/content/drive") else current_dir, "tournament_results.txt")
        with open(results_file, 'w') as f:
            f.write(f"Tournament Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"="*70 + "\n")
            f.write(f"Performance Rating: {int(overall_perf)} ± {int(error_margin)} ELO\n")
            f.write(f"Total Score: {total_score}/{total_games} ({100*win_rate:.1f}%)\n")
            f.write(f"95% Confidence: [{int(overall_perf - error_margin)}, {int(overall_perf + error_margin)}]\n")
        print(f"\n✅ Results saved to {results_file}")

        # Print cache statistics
        if USE_NN and hasattr(my_engine, 'evaluator') and my_engine.evaluator:
            my_engine.evaluator.print_stats()


if __name__ == "__main__":
    main()
