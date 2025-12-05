import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
import sys
import os
import gc
import time

from backend_engine.data_processing.tensor_converter import fen_to_tensor
from backend_engine.ml.vocab import MOVE_TO_INT


class ChessDataset(Dataset):
    """
    create a PyTorch Dataset for loading chess positions and their evaluations
    from the parquet files generated during data processing, because reading the parquet
    files is expensive and slow, instead load them into RAM as pandas DataFrames,
    then they can be efficiently converted to tensors by the DataLoader using mutli-threading.

    https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    def __init__(self, data_frame, history_length=5):
        """Load data from a pandas DataFrame, and set history length"""
        self.data = data_frame
        self.history_length = history_length

        # pre-convert to numpy array for faster access
        self.fens = self.data["fen"].to_numpy()
        self.results = self.data["result"].to_numpy()
        self.game_nums = self.data["game_num"].to_numpy()
        self.is_black = self.data["is_black"].to_numpy()
        self.moves = self.data["move_uci"].to_numpy()

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Return the tensor and result for the given index"""
        result = self.results[idx]
        game_num = self.game_nums[idx]
        is_black = self.is_black[idx]
        move_str = self.moves[idx]

        # convert move to integer index
        try:
            move_idx = MOVE_TO_INT[move_str]
        except KeyError:
            move_idx = 0

        if is_black == 1:
            # if its black to move, flip the result
            result = -result

        # create the result tensors for value and policy heads
        value_result = torch.tensor([result], dtype=torch.float32)
        policy_result = torch.tensor(move_idx, dtype=torch.long)

        # grab an appropriate amount of board states from the FEN string
        boards = []
        for i in range(self.history_length):
            # look back i moves
            curr_idx = idx - i
            if curr_idx < 0 or self.game_nums[curr_idx] != game_num:
                # out of bounds or different game, pad with empty board
                boards.append(np.zeros((20, 8, 8), dtype=np.float32))
            else:
                fen = self.fens[curr_idx]
                boards.append(fen_to_tensor(fen))

        tensor = np.concatenate(boards, axis=0)
        tensor = torch.from_numpy(tensor).float()

        return tensor, value_result, policy_result


def chunk_loader(parquet_files, batch_size=512, num_workers=4, pin_memory=False, history_length=5):
    """
    Generator that yields pytorch DataLoader objects for each chunk of data in
    a list of files. Using the pytorch DataLoader class allows us to take advantage of
    multi-threading an batching in a really easy and fast way.

    https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    Args:
        data_dir (str): Directory containing parquet files.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker threads for loading data.
        pin_memory (bool): Whether to use pinned memory for faster GPU transfers.

    Yields:
        DataLoader: A DataLoader for each chunk of data.
    """
    # Find all parquet files in the data directory

    for file_path in parquet_files:
        print(f"Loading chunk from file: {os.path.basename(file_path)}")
        try:
            # Load the parquet file into a pandas DataFrame
            df = pd.read_parquet(file_path)

            # Create a ChessDataset from the DataFrame
            dataset = ChessDataset(df, history_length=history_length)

            # Create a DataLoader for the dataset
            # pin_memory speeds up transfer to GPU
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=False
                )

            yield loader
            # free up memory
            del loader
            del dataset
            del df
            # force garbage collection
            gc.collect()
            # give the os time to collect garbage
            time.sleep(0.1)

        except Exception as e:
            print(f"Error loading {file_path}: {e}", file=sys.stderr)
            continue


def get_train_test_loaders(data_dir, batch_size=512, validation_split=0.1, num_workers=4, pin_memory=False, history_length=5):
    """
    splits the parquet files in data_dir into training and validation sets,
    then returns DataLoader objects for each set.
    Args:
        data_dir (str): Directory containing parquet files.
        batch_size (int): Batch size for the DataLoader.
        validation_split (float): Fraction of data to use for validation.
        num_workers (int): Number of worker threads for loading data.
        pin_memory (bool): Whether to use pinned memory for faster GPU transfers.
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for validation data.
        train_count (int): Number of training files.
        test_count (int): Number of validation files
    """
    # Find all parquet files in the data directory
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    parquet_files.sort()  # Ensure consistent order

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in directory: {data_dir}")

    print(f"Found {len(parquet_files)} parquet files in {data_dir}.")

    # Split files into training and validation sets
    split_idx = int(len(parquet_files) * (1 - validation_split))
    train_files = parquet_files[:split_idx]
    test_files = parquet_files[split_idx:]

    print(f'Using {len(train_files)} files for training and {len(test_files)} files for validation.')

    # Create DataLoaders for training and validation sets
    train_loader = chunk_loader(train_files, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, history_length=history_length)
    test_loader = chunk_loader(test_files, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, history_length=history_length)

    return train_loader, test_loader, len(train_files), len(test_files)
