import chess.pgn
import pandas as pd
import os
import time
import glob
import multiprocessing
from tqdm import tqdm

# ================= CONFIGURATION =================
# Output directory for the small chunk files
OUTPUT_DIR = "/Users/mratcliff/Documents/GitHub/cmpm118_chessengine/data/processed/"
RAW_DATA_DIR = "/Users/mratcliff/Documents/GitHub/cmpm118_chessengine/data/raw/"

# How many moves to process before saving (keeps RAM low)
CHUNK_SIZE = 100000
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


def save_chunk(data, worker_id, index):
    df = pd.DataFrame(data)

    # explcitly cast integer columns to reduce size
    df["game_num"] = df["game_num"].astype("string")
    df["move_num"] = df["move_num"].astype("int16")
    df["is_black"] = df["is_black"].astype("int8")
    df["result"] = df["result"].astype("int8")
    df["move_uci"] = df["move_uci"].astype("string")

    # Save as Parquet (10x smaller than CSV)
    filename = os.path.join(OUTPUT_DIR, f"chunk_{worker_id}_{index}.parquet")
    df.to_parquet(filename, index=False)


def process_pgn(file_path):
    """
    dispatches a worker to process PGN files into smaller chunks of parquet data
    """
    worker_id = multiprocessing.current_process().pid
    data_buffer = []
    chunk_index = 0
    game_num = 0

    with open(file_path, encoding="utf-8") as pgn:
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
                game_id = f"{worker_id}_{game_num}"
                # track moves per game
                move_num = 0

                for move in game.mainline_moves():
                    # get the FEN before the move is played
                    fen = board.fen()

                    if board.turn == chess.BLACK:
                        is_black_turn = 1
                    else:
                        is_black_turn = 0

                    # get the move in UCI format
                    move_uci = move.uci()

                    data_buffer.append({
                        "game_num": game_id,
                        "move_num": move_num,
                        "fen": fen,
                        "is_black": is_black_turn,
                        "result": result,
                        "move_uci": move_uci
                    })

                    board.push(move)
                    move_num += 1

            # Save Chunk if buffer is full
            if len(data_buffer) >= CHUNK_SIZE:
                save_chunk(data_buffer, worker_id, chunk_index)
                data_buffer = []
                chunk_index += 1

    # Save whatever is left
    if data_buffer:
        save_chunk(data_buffer, worker_id, chunk_index)

    return True


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Get list of files
    pgn_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.pgn"))
    pgn_files.sort()

    print(f"Found {len(pgn_files)} PGN files. Starting Multiprocessing...")
    print(f"Using {multiprocessing.cpu_count()} CPU cores.")

    start_time = time.time()

    # Create a pool of workers equal to number of CPU cores
    # We use imap_unordered for better responsiveness with tqdm
    with multiprocessing.Pool() as pool:
        list(tqdm(pool.imap_unordered(process_pgn, pgn_files), total=len(pgn_files)))

    duration = time.time() - start_time
    print(f"Done! Processed {len(pgn_files)} files in {duration:.2f} seconds.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
