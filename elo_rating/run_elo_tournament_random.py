import sys
import os
import chess
import chess.engine
import chess.pgn
import math
import shutil
import time
# Add the project root to sys.path to allow imports from backend_engine
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
# Add backend_engine to path so internal imports (like 'from ml...') work
sys.path.append(os.path.join(project_root, "backend_engine"))

from engines.random_engine import RandomEngine


# --- CONFIGURATION ---
# Path to Stockfish - try to find it in PATH, otherwise set manually
STOCKFISH_PATH = shutil.which("stockfish")
# If not found automatically, uncomment and set your path:
# STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # macOS Homebrew default
# STOCKFISH_PATH = "/usr/local/bin/stockfish"

# Engine Settings - RANDOM ENGINE
ENGINE_NAME = "RandomEngine"

# Tournament Settings
GAMES_PER_LEVEL = 10  # More games for better accuracy with random
STOCKFISH_LEVELS = [0]  # Just test against easiest level for now
SAVE_PGN = True  # Save games to PGN file for analysis
PGN_OUTPUT = os.path.join(current_dir, "random_vs_stockfish.pgn")

# Approximate ELOs for Stockfish levels (UCI Skill Level 0-20)
# NOTE: These are estimates. Stockfish Level 0 is still much stronger than random play.
# Random engine is expected to score poorly even against Level 0.
STOCKFISH_ELO_MAP = {
    0: 800,   # Beginner level
    1: 1400,   # Novice 
    2: 1500,   # Class D
    3: 1600,   # Class C
    4: 1700,   # Class B
    5: 1850,   # Class A
    6: 2000,   # Expert
    7: 2150,   # Candidate Master
    8: 2300,   # Master
    9: 2400,   # FIDE Master
    10: 2500   # International Master
}

def get_stockfish_engine(skill_level):
    if not STOCKFISH_PATH:
        raise FileNotFoundError("Stockfish executable not found. Please install Stockfish or set STOCKFISH_PATH.")

    # Create a simple engine instance
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Skill Level": skill_level})
    return engine

def play_game(my_engine, sf_engine, i_am_white, sf_level):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = ENGINE_NAME if i_am_white else f"Stockfish-L{sf_level}"
    game.headers["Black"] = f"Stockfish-L{sf_level}" if i_am_white else ENGINE_NAME
    node = game

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            if i_am_white:
                # My Engine (White)
                try:
                    move_uci = my_engine.get_move(board.fen())
                    move = chess.Move.from_uci(move_uci)
                    if move not in board.legal_moves:
                        print(f"Error: Engine returned illegal move {move_uci}")
                        return 0.0, None # Loss
                    board.push(move)
                    node = node.add_variation(move)
                except Exception as e:
                    print(f"Engine error: {e}")
                    return 0.0, None
            else:
                # Stockfish (White)
                result = sf_engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)
                node = node.add_variation(result.move)
        else:
            if not i_am_white:
                # My Engine (Black)
                try:
                    move_uci = my_engine.get_move(board.fen())
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
                # Stockfish (Black)
                result = sf_engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)
                node = node.add_variation(result.move)

    # Game Over
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
    percentage = score / total_games

    if percentage >= 0.99:
        return opponent_elo + 400
    if percentage <= 0.01:
        return opponent_elo - 400

    # Standard logistic distribution formula derived from ELO expectation
    diff = -400 * math.log10(1 / percentage - 1)
    return opponent_elo + diff

def calculate_elo_error_margin(games_played, win_rate=0.5):
    """
    Calculate the 95% confidence interval for ELO rating.

    Typical error margins:
    - 10 games: ±100 ELO
    - 20 games: ±70 ELO
    - 50 games: ±45 ELO
    - 100 games: ±32 ELO
    """
    import math
    std_error = math.sqrt(win_rate * (1 - win_rate) / games_played)
    elo_uncertainty = 1.96 * 173.7 * std_error
    return elo_uncertainty

def main():
    print("="*70)
    print(f"RANDOM ENGINE ELO TOURNAMENT")
    print("="*70)
    print(f"Testing: {ENGINE_NAME}")
    print(f"Opponent: Stockfish (various levels)")
    print(f"Games per level: {GAMES_PER_LEVEL}")
    print("="*70)

    # Initialize Random Engine
    print(f"\nInitializing {ENGINE_NAME}...")
    try:
        my_engine = RandomEngine()
        print("✅ Random engine initialized")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return

    if not STOCKFISH_PATH:
        print("❌ ERROR: Stockfish not found. Please install it (e.g. 'brew install stockfish')")
        return

    print(f"\n🏁 Starting Tournament")
    print(f"{'-'*70}")
    print(f"| {'Level':<5} | {'SF ELO':<7} | {'Score':<12} | {'Perf Rating':<12} |")
    print(f"{'-'*70}")

    overall_results = []
    overall_opponent_elos = []
    all_games = []

    for level in STOCKFISH_LEVELS:
        sf_elo = STOCKFISH_ELO_MAP.get(level, 1350 + level * 100)
        score = 0.0

        try:
            sf_engine = get_stockfish_engine(level)
        except Exception as e:
            print(f"❌ Could not start Stockfish: {e}")
            break

        print(f"\n📊 Level {level} (ELO {sf_elo}):")

        for i in range(GAMES_PER_LEVEL):
            i_am_white = (i % 2 == 0)

            print(f"  Game {i+1}/{GAMES_PER_LEVEL} ({'W' if i_am_white else 'B'})...", end=" ", flush=True)

            game_score, pgn_game = play_game(my_engine, sf_engine, i_am_white, level)
            score += game_score

            # Print result immediately
            if game_score == 1.0:
                print("✅ WIN")
            elif game_score == 0.0:
                print("❌ LOSS")
            else:
                print("🤝 DRAW")

            if pgn_game and SAVE_PGN:
                all_games.append(pgn_game)

            overall_results.append(game_score)
            overall_opponent_elos.append(sf_elo)

        sf_engine.quit()

        perf = calculate_performance_rating(score, GAMES_PER_LEVEL, sf_elo)
        print(f"| {level:<5} | {sf_elo:<7} | {score}/{GAMES_PER_LEVEL:<10} | {int(perf):<12} |")

    # Save PGN file
    if SAVE_PGN and all_games:
        with open(PGN_OUTPUT, "w") as pgn_file:
            for game in all_games:
                print(game, file=pgn_file, end="\n\n")
        print(f"\n✅ Saved {len(all_games)} games to {PGN_OUTPUT}")

    # Calculate overall performance
    if overall_results:
        avg_opp_elo = sum(overall_opponent_elos) / len(overall_opponent_elos)
        total_score = sum(overall_results)
        total_games = len(overall_results)
        overall_perf = calculate_performance_rating(total_score, total_games, avg_opp_elo)
        win_rate = total_score / total_games
        error_margin = calculate_elo_error_margin(total_games, win_rate)

        print(f"\n{'='*70}")
        print(f"🏆 FINAL RESULTS - {ENGINE_NAME}")
        print(f"{'='*70}")
        print(f"Performance Rating: {int(overall_perf)} ± {int(error_margin)} ELO")
        print(f"Total Score: {total_score}/{total_games} ({100*win_rate:.1f}%)")
        print(f"95% Confidence: [{int(overall_perf - error_margin)}, {int(overall_perf + error_margin)}]")
        print(f"\n💡 Interpretation:")
        print(f"   The {ENGINE_NAME} strength is between {int(overall_perf - error_margin)}")
        print(f"   and {int(overall_perf + error_margin)} ELO (95% confidence)")

        # Comparison context
        print(f"\n📊 Context:")
        print(f"   Random moves: ~500-800 ELO (expected)")
        print(f"   Beginner player: ~800-1200 ELO")
        print(f"   Your NN engine: ~1400-1600 ELO (estimated from previous tests)")

if __name__ == "__main__":
    main()
