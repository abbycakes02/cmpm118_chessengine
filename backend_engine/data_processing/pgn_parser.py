import chess.pgn
import pandas as pd
import os
import time
import glob

# ================= CONFIGURATION =================
# Output directory for the small chunk files
OUTPUT_DIR = "/Users/mratcliff/Documents/GitHub/cmpm118_chessengine/data/processed/"
RAW_DATA_DIR = "/Users/mratcliff/Documents/GitHub/cmpm118_chessengine/data/raw/"

# How many moves to process before saving (keeps RAM low)
CHUNK_SIZE = 2000000

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

    # count of which game we're on which is important for passing in game history
    game_num = 0
    files_processed = 0
    for pgn_file_path in pgn_files:
        files_processed += 1
        print(f"Processing file {files_processed}/{len(pgn_files)}: {os.path.basename(pgn_file_path)}...")

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
                    game_num += 1
                    # track moves per game
                    move_num = 0

                    for move in game.mainline_moves():
                        board.push(move)
                        move_num += 1

                        if board.turn == chess.BLACK:
                            is_black_turn = 1
                        else:
                            is_black_turn = 0

                        data_buffer.append({
                            "game_num": game_num,
                            "move_num": move_num,
                            "fen": board.fen(),
                            "is_black": is_black_turn,
                            "result": result
                        })

                games_processed += 1

                # Save Chunk if buffer is full
                if len(data_buffer) >= CHUNK_SIZE:
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

    # explcitly cast integer columns to reduce size
    df["game_num"] = df["game_num"].astype("int32")
    df["move_num"] = df["move_num"].astype("int16")
    df["is_black"] = df["is_black"].astype("int8")
    df["result"] = df["result"].astype("int8")

    # Save as Parquet (10x smaller than CSV)
    filename = os.path.join(OUTPUT_DIR, f"chunk_{index}.parquet")
    df.to_parquet(filename, index=False)
    print(f"Saved chunk_{index}.parquet ({len(df)} positions)")


if __name__ == "__main__":
    process_pgn()
