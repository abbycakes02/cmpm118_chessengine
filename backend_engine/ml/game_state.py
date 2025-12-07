from collections import deque
import numpy as np

from data_processing.tensor_converter import fen_to_tensor


class GameHistory():
    """
    uses a deque to store the history of game states for training purposes
    deque is faster than list concatenation for appending and popping from both ends
    """
    def __init__(self, history_length=5, board_shape=(20, 8, 8)):
        self.max_length = history_length
        self.shape = board_shape
        # pre-fill the deque with empty board states
        init = []
        for _ in range(self.max_length):
            init.append(np.zeros(self.shape, dtype=np.float32))

        self.history = deque(init, maxlen=self.max_length)

    def clear(self):
        """
        clears the history deque and fills it with empty board states
        """
        self.history.clear()
        for _ in range(self.max_length):
            self.history.append(np.zeros(self.shape, dtype=np.float32))

    def push(self, fen_str):
        """
        pushes a new board state onto the history deque

        Args:
            board_tensor (numpy.ndarray): tensor representing the board state
        """
        board_tensor = fen_to_tensor(fen_str)
        self.history.append(board_tensor)

    def pop(self):
        """
        pops the most recent board state from the history deque
        """
        self.history.pop()
        # pad with an empty board state to maintain length
        self.history.appendleft(np.zeros(self.shape, dtype=np.float32))

    def get_history(self):
        """
        returns the current history as a single tensor

        Returns:
            torch.Tensor: tensor of shape (history_length, 12, 8, 8)
        """
        # reverse the deque to have the most recent state last
        reversed_history = list(self.history)[::-1]
        return np.concatenate(reversed_history, axis=0)
