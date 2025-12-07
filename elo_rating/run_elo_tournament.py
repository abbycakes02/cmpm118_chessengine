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

from engines.minimax_engine import MinimaxEngine


# --- CONFIGURATION ---
# Path to Stockfish - try to find it in PATH, otherwise set manually
STOCKFISH_PATH = shutil.which("stockfish")
# If not found automatically, uncomment and set your path:
# STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # macOS Homebrew default
# STOCKFISH_PATH = "/usr/local/bin/stockfish"

# Engine Settings
USE_NN = True  # Set to True to test the Neural Network version
MODEL_PATH = os.path.join(project_root, "backend_engine", "ml", "models", "session_1764582902", "chess_valuenet_32ch_3resblocks.pth") # 32 channels, 3 blocks
MAX_DEPTH = 2   # Depth for your engine (reduced for NN - it's slower)
TIME_LIMIT = 10 # Seconds per move for BOTH engines (fair comparison)

# Tournament Settings
GAMES_PER_LEVEL = 4  # More games = more accurate rating (10-20 recommended)
STOCKFISH_LEVELS = [0, 1]  # Broader range to find competitive zone
SAVE_PGN = True  # Save games to PGN file for analysis
PGN_OUTPUT = os.path.join(current_dir, "minimax_vs_stockfish.pgn")

# Approximate ELOs for Stockfish levels (UCI Skill Level 0-20)
# Based on empirical testing and CCRL data for limited-strength Stockfish
# Source: ChessProgramming Wiki, Stockfish testing, and engine rating lists
STOCKFISH_ELO_MAP = {
    0: 1320,   # Beginner level
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

def play_game(my_engine, sf_engine, i_am_white, time_limit):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "MinimaxEngine" if i_am_white else f"Stockfish-L{sf_engine.options.get('Skill Level', 0)}"
    game.headers["Black"] = f"Stockfish-L{sf_engine.options.get('Skill Level', 0)}" if i_am_white else "MinimaxEngine"
    node = game
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            if i_am_white:
                # My Engine (White)
                try:
                    move_uci = my_engine.get_move(board.fen(), time_limit=time_limit)
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
                result = sf_engine.play(board, chess.engine.Limit(time=time_limit))
                board.push(result.move)
                node = node.add_variation(result.move)
        else:
            if not i_am_white:
                # My Engine (Black)
                try:
                    move_uci = my_engine.get_move(board.fen(), time_limit=time_limit)
                    move = chess.Move.from_uci(move_uci)
                    if move not in board.legal_moves:
                        print(f"Error: Engine returned illegal move {move_uci}")
                        return 0.0, None # Loss (fixed from 1.0)
                    board.push(move)
                    node = node.add_variation(move)
                except Exception as e:
                    print(f"Engine error: {e}")
                    return 0.0, None # Loss (fixed from 1.0)
            else:
                # Stockfish (Black)
                result = sf_engine.play(board, chess.engine.Limit(time=time_limit))
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
    # Expected_Score = 1 / (1 + 10^((Rb - Ra) / 400))
    # We solve for Ra (our rating) given Expected_Score (our actual percentage) and Rb (opponent rating)
    diff = -400 * math.log10(1 / percentage - 1)
    return opponent_elo + diff

def calculate_elo_error_margin(games_played, win_rate=0.5):
    """
    Calculate the 95% confidence interval for ELO rating.
    Based on statistical analysis from ChessProgramming Wiki.
    
    Typical error margins:
    - 10 games: ±100 ELO
    - 20 games: ±70 ELO
    - 50 games: ±45 ELO
    - 100 games: ±32 ELO
    """
    import math
    # Standard error formula for binomial proportion
    std_error = math.sqrt(win_rate * (1 - win_rate) / games_played)
    # 1.96 is the z-score for 95% confidence
    # 400/ln(10) ≈ 173.7 converts probability to ELO
    elo_uncertainty = 1.96 * 173.7 * std_error
    return elo_uncertainty

def main():
    print(f"Initializing MinimaxEngine (NN={USE_NN}, Depth={MAX_DEPTH})...")
    try:
        if USE_NN:
            my_engine = MinimaxEngine(
                use_nn=True, 
                model_path=MODEL_PATH, 
                channels=32,  # Must match the model architecture
                blocks=3,     # Must match the model architecture
                max_depth=MAX_DEPTH
            )
        else:
            my_engine = MinimaxEngine(
                use_nn=False, 
                max_depth=MAX_DEPTH
            )
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    if not STOCKFISH_PATH:
        print("ERROR: Stockfish not found. Please install it (e.g. 'brew install stockfish') or update STOCKFISH_PATH in the script.")
        return

    print(f"Starting Gauntlet against Stockfish (Path: {STOCKFISH_PATH})")
    print(f"Time per move: {TIME_LIMIT}s (equal for both engines)")
    print(f"{'-'*60}")
    print(f"| {'Level':<5} | {'SF ELO':<6} | {'Score':<10} | {'Perf Rating':<12} |")
    print(f"{'-'*60}")

    overall_results = []
    overall_opponent_elos = []
    all_games = []

    for level in STOCKFISH_LEVELS:
        sf_elo = STOCKFISH_ELO_MAP.get(level, 1350 + level * 100)
        score = 0.0
        
        try:
            sf_engine = get_stockfish_engine(level)
        except Exception as e:
            print(f"Could not start Stockfish: {e}")
            break

        for i in range(GAMES_PER_LEVEL):
            i_am_white = (i % 2 == 0)
            
            game_score, pgn_game = play_game(my_engine, sf_engine, i_am_white, TIME_LIMIT)
            score += game_score
            
            if pgn_game and SAVE_PGN:
                all_games.append(pgn_game)
            
            overall_results.append(game_score)
            overall_opponent_elos.append(sf_elo)

        sf_engine.quit()
        
        perf = calculate_performance_rating(score, GAMES_PER_LEVEL, sf_elo)
        print(f"| {level:<5} | {sf_elo:<6} | {score}/{GAMES_PER_LEVEL:<8} | {int(perf):<12} |")

    # Save PGN file
    if SAVE_PGN and all_games:
        with open(PGN_OUTPUT, "w") as pgn_file:
            for game in all_games:
                print(game, file=pgn_file, end="\n\n")
        print(f"\n✓ Saved {len(all_games)} games to {PGN_OUTPUT}")

    # Calculate overall performance
    if overall_results:
        avg_opp_elo = sum(overall_opponent_elos) / len(overall_opponent_elos)
        total_score = sum(overall_results)
        total_games = len(overall_results)
        overall_perf = calculate_performance_rating(total_score, total_games, avg_opp_elo)
        win_rate = total_score / total_games
        error_margin = calculate_elo_error_margin(total_games, win_rate)
        
        print(f"{'-'*60}")
        print(f"Overall Performance Rating: {int(overall_perf)} ± {int(error_margin)} ELO")
        print(f"Total Score: {total_score}/{total_games} ({100*win_rate:.1f}%)")
        print(f"95% Confidence Interval: [{int(overall_perf - error_margin)}, {int(overall_perf + error_margin)}]")
        print(f"\nInterpretation:")
        print(f"  With {total_games} games played, we are 95% confident")
        print(f"  your engine's true strength is between {int(overall_perf - error_margin)} and {int(overall_perf + error_margin)} ELO.")
        print(f"\nNote: Play {int(100/GAMES_PER_LEVEL * 10)} games per level (total ~{int(100/GAMES_PER_LEVEL * 10 * len(STOCKFISH_LEVELS))}) for ±32 ELO accuracy.")

if __name__ == "__main__":
    main()
