import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import sys
import os

from backend_engine.data_processing.tensor_converter import fen_to_tensor


class ChessDataset(Dataset):
    """
    create a PyTorch Dataset for loading chess positions and their evaluations
    from the parquet files generated during data processing, because reading the parquet
    files is expensive and slow, instead load them into RAM as pandas DataFrames,
    then they can be efficiently converted to tensors by the DataLoader using mutli-threading.

    https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    def __init__(self, data_frame):
        """Load data from a pandas DataFrame"""
        self.data = data_frame

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Return the tensor and result for the given index"""
        row = self.data.iloc[idx]
        fen_str = row["fen"]
        result = row["result"]
        tensor = fen_to_tensor(fen_str)
        # tensor: (20, 8, 8)

        # result is currently just a scalar, but the model expects a tensor
        result = torch.tensor([result], dtype=torch.float32)
        # result: (1,)

        return tensor, result


def chunk_loader(data_dir, batch_size=512, num_workers=4):
    """
    Generator that yields pytorch DataLoader objects for each chunk of data in
    the data_dir. Using the pytorch DataLoader class allows us to take advantage of
    multi-threading an batching in a really easy and fast way.

    https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    Args:
        data_dir (str): Directory containing parquet files.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker threads for loading data.

    Yields:
        DataLoader: A DataLoader for each chunk of data.
    """
    # Find all parquet files in the data directory
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    parquet_files.sort()  # Ensure consistent order

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in directory: {data_dir}")

    print(f"Found {len(parquet_files)} parquet files in {data_dir}.")

    for file_path in parquet_files:
        print(f"Loading chunk from file: {os.path.basename(file_path)}")
        try:
            # Load the parquet file into a pandas DataFrame
            df = pd.read_parquet(file_path)

            # Create a ChessDataset from the DataFrame
            dataset = ChessDataset(df)

            # Create a DataLoader for the dataset
            # pin_memory speeds up transfer to GPU
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
                )

            yield loader, len(parquet_files)

        except Exception as e:
            print(f"Error loading {file_path}: {e}", file=sys.stderr)
            continue
