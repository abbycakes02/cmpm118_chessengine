import chess.pgn
import pandas as pd
import os
import time
import glob

# ================= CONFIGURATION =================
# Output directory for the small chunk files
OUTPUT_DIR = "/Users/mratcliff/Documents/GitHub/cmpm118_chessengine/data/processed/"
RAW_DATA_DIR = "/Users/mratcliff/Documents/GitHub/cmpm118_chessengine/data/raw/"

# How many games to process before saving (keeps RAM low)
CHUNK_SIZE = 200000

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

    # Find all .pgn files in the raw directory
    pgn_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.pgn"))
    pgn_files.sort()  # Process in order of date

    print(f"Found {len(pgn_files)} PGN files to process.")

    games_processed = 0
    chunk_index = 0
    data_buffer = []
    start_time = time.time()

    for pgn_file_path in pgn_files:

        print(f"Processing file: {os.path.basename(pgn_file_path)}...")

        with open(pgn_file_path, encoding="utf-8") as pgn:
            while True:
                try:
                    game = chess.pgn.read_game(pgn)
                except Exception:
                    continue  # Skip broken games

                if game is None:
                    break  # End of this file

                #  Extract Result
                result = parse_result(game.headers.get("Result", "*"))

                if result is not None:
                    board = game.board()
                    for move in game.mainline_moves():
                        board.push(move)
                        data_buffer.append({
                            "fen": board.fen(),
                            "result": result
                        })

                games_processed += 1

                # Save Chunk if buffer is full roughly 50 moves per game
                if len(data_buffer) >= (CHUNK_SIZE * 50):
                    save_chunk(data_buffer, chunk_index)
                    data_buffer = []
                    chunk_index += 1
                    print(f"{games_processed} games processed, in {time.time() - start_time:.2f} seconds.")

                if MAX_GAMES and games_processed >= MAX_GAMES:
                    print("Max games reached.")
                    return

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
