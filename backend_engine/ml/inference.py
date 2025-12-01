import torch
import os

from .model import ChessValueNet
from data_processing.tensor_converter import fen_to_tensor


class ChessEvaluator():
    """
    class to load a trained ChessValueNet model and use it to evaluate chess positions
    """

    def __init__(self, model_path, channels, blocks):
        """
        wrap the ChessValueNet model to make it easy to access

        Args:
            model_path (str): path to the trained model file
            channels (int): number of input channels for the model
            blocks (int): number of residual blocks in the model
        """
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.channels = channels
        self.blocks = blocks

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Loading model on device: {self.device}")
        print(f" Channels: {self.channels}, Residual Blocks: {self.blocks}")

        self.model = ChessValueNet(
            num_channels=self.channels,
            num_residual_blocks=self.blocks
            ).to(self.device)

        weights = torch.load(self.model_path, map_location=self.device)
        try:
            self.model.load_state_dict(weights)
        except RuntimeError as e:
            raise RuntimeError(f"Error loading model weights: {e}")

        self.model.eval()

    def __call__(self, board):
        return self.evaluate_position(board)

    def evaluate_position(self, board):
        """
        evaluate a chess position given in FEN notation

        Args:
            fen_str (str): FEN string representing the chess position

        Returns:
            float: evaluation score between -1 (black win) and 1 (white win)
        """
        # unpack board to fen and then to tensor
        fen_str = board.fen()
        tensor = fen_to_tensor(fen_str)

        # add batch dimension and move to device
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(tensor)
            score = prediction.item()  # get scalar value

        return score
