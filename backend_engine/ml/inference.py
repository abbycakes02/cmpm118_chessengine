import torch
import torch.nn.functional as F
import os

from ml.model import ChessNet
from ml.game_state import GameHistory


class ChessEvaluator():
    """
    class to load a trained ChessValueNet model and use it to evaluate chess positions
    """

    def __init__(self, model_path, input_channels, hidden_channels, blocks, history_length=5):
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

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.blocks = blocks
        self.history_length = history_length
        self.history = GameHistory(history_length=history_length)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Loading model on device: {self.device}")
        print(f" Channels: {self.hidden_channels}, Residual Blocks: {self.blocks}")

        self.model = ChessNet(
            vocab_size=4544,
            history_length=self.history_length,
            board_channels=self.input_channels,
            hidden_channels=self.hidden_channels,
            num_blocks=self.blocks
            ).to(self.device)

        weights = torch.load(self.model_path, map_location=self.device)
        try:
            self.model.load_state_dict(weights)
        except RuntimeError as e:
            raise RuntimeError(f"Error loading model weights: {e}")

        self.model.eval()

    def __call__(self, board):
        return self.evaluate_position(board)

    def evaluate_position(self, fen_str):
        """
        evaluate a chess position given in FEN notation

        Args:
            fen_str (str): FEN string representing the chess position

        Returns:
            float: evaluation score between -1 (black win) and 1 (white win)
        """
        self.history.push(fen_str)

        tensor = self.history.get_history()

        # convert to torch tensor and add batch dimension
        tensor = torch.from_numpy(tensor).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, score = self.model(tensor)
            policy_probs = F.softmax(policy_logits, dim=1).squeeze(0)
            policy_probs = policy_probs.cpu().numpy()
            score = score.item()

            return policy_probs, score

        return score

    def reset_history(self):
        """
        clears the game history
        """
        self.history.clear()

    def pop_history(self):
        """
        pops the most recent board state from the history
        """
        self.history.pop()
