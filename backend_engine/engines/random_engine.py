import chess
import random


def get_random_move(fen: str) -> str:
    """
    takes the current games state as a FEN and returns a random legal move in UCI format.
    Input: FEN string
    Output: UCI move string (e.g., "e2e4") or None if game over
    """
    board = chess.Board(fen)

    if board.is_game_over():
        raise ValueError("Game is over")

    legal_moves = list(board.legal_moves)

    if not legal_moves:
        raise ValueError("No legal moves available")

    return random.choice(legal_moves).uci()
