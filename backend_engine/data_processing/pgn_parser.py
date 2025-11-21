import chess.pgn
import pandas as pd
import os
import time
import sys

# ================= CONFIGURATION =================
# Output directory for the small chunk files
OUTPUT_DIR = "../../data/processed/"

# How many games to process before saving (keeps RAM low)
CHUNK_SIZE = 5000

# Set to None to process everything, or a number (e.g., 10000) for a quick test
MAX_GAMES = None
# =================================================


def parse_result(result_str):
    """
    Converts PGN result string to an integer label.
    1: White Wins, -1: Black Wins, 0: Draw
    """
    if result_str == "1-0":
        return 1
    if result_str == "0-1":
        return -1
    if result_str == "1/2-1/2":
        return 0
    return None


def process_pgn():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # We read from sys.stdin so we can pipe the decompressed stream directly
    print("Waiting for PGN data from stdin...")
    pgn_input = sys.stdin

    games_processed = 0
    chunk_index = 0
    data_buffer = []
    start_time = time.time()

    while True:
        try:
            # Read one game from the stream
            game = chess.pgn.read_game(pgn_input)
        except Exception:
            # Skip decoding errors common in massive streams
            continue

        if game is None:
            break  # End of stream

        # Extract Result
        result = parse_result(game.headers.get("Result", "*"))

        # Optional: Filter by ELO (if you want only high quality games)
        # try:
        #     if int(game.headers.get("WhiteElo", 0)) < 2000: continue
        # except: continue

        if result is not None:
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)

                # Data we save: The Board FEN for each move, and the Final Result
                data_buffer.append({
                    "fen": board.fen(),
                    "result": result
                })

        games_processed += 1

        # Check Buffer Size (Approx 50 moves per game * 5000 games = 250k rows)
        if len(data_buffer) >= (CHUNK_SIZE * 50):
            save_chunk(data_buffer, chunk_index)
            data_buffer = []  # Clear RAM
            chunk_index += 1

            elapsed = time.time() - start_time
            print(f"Processed {games_processed} games in {elapsed:.1f}s...")

        # Stop if we hit the limit (for testing)
        if MAX_GAMES and games_processed >= MAX_GAMES:
            print(f"Limit of {MAX_GAMES} games reached.")
            break

    # Save whatever is left
    if data_buffer:
        save_chunk(data_buffer, chunk_index)

    print("Done! Processing complete.")


def save_chunk(data, index):
    df = pd.DataFrame(data)

    # Save as Parquet (10x smaller than CSV)
    filename = os.path.join(OUTPUT_DIR, f"chunk_{index}.parquet")
    df.to_parquet(filename, index=False)
    print(f"Saved {filename} ({len(df)} positions)")


if __name__ == "__main__":
    process_pgn()
