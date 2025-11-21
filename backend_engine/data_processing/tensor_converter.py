import chess
import torch
import numpy as np


def fen_to_tensor(fen: str):
    """
    Converts a FEN string into a 21x8x8 PyTorch Tensor.

    Dimensions: [21, 8, 8]
    - Channels 0-5:   White Pieces (Pawn, knight, Bishop, rook, queen, king)
    - Channels 6-11:  Black Pieces (Pawn, knight, Bishop, rook, queen, king)
    - Channel 12:     Side to Move (1.0 = White, 0.0 = Black)
    - Channels 13-16: Castling Rights (White Kingside, White Queenside, Black Kingside, Black Queenside)
    - Channel 17:     En Passant Target Square
    - Channel 18:     50-Move Rule Counter (Normalized)
    - Channel 19:     Game Phase / Total Move Count (Normalized)
    """
    board = chess.Board(fen)

    # Initialize the tensor: 20 channels, 8x8 board, float32
    planes = np.zeros((20, 8, 8), dtype=np.float32)

    # -------------------------------------------------------------------------
    # 1. Piece Planes (Channels 0-11)
    # -------------------------------------------------------------------------
    # Standard piece order for iteration
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    for i, piece_type in enumerate(piece_types):
        # White Pieces (0-5)
        for square in board.pieces(piece_type, chess.WHITE):
            rank, file = divmod(square, 8)
            # 7-rank because numpy array 0,0 is usually top-left,
            # but chess rank 0 is bottom. We map board rank 7 -> array row 0
            planes[i][7 - rank][file] = 1.0

        # Black Pieces (6-11)
        for square in board.pieces(piece_type, chess.BLACK):
            rank, file = divmod(square, 8)
            planes[i + 6][7 - rank][file] = 1.0

    # -------------------------------------------------------------------------
    # 2. Game State Planes (Channels 12-20)
    # -------------------------------------------------------------------------

    # Channel 12: Side to Move (1.0 for White, 0.0 for Black)
    if board.turn == chess.WHITE:
        planes[12].fill(1.0)

    # Channels 13-16: Castling Rights
    # 13: White Kingside
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13].fill(1.0)
    # 14: White Queenside
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14].fill(1.0)
    # 15: Black Kingside
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15].fill(1.0)
    # 16: Black Queenside
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16].fill(1.0)

    # Channel 17: En Passant Target
    # If there is an EP square, mark it with 1.0
    if board.ep_square:
        rank, file = divmod(board.ep_square, 8)
        planes[17][7 - rank][file] = 1.0

    # Channel 18: 50-Move Rule (Halfmove Clock)
    # Normalized: 50 moves is the limit, but sometimes it goes to 100 plies (50 moves)
    # We normalize by 100.0 to keep it roughly 0-1
    planes[18].fill(board.halfmove_clock / 100.0)

    # Channel 19: Repetition Counter
    # Note: FEN strings are "stateless" regarding previous positions.
    # Unless we track the game history, we cannot know if this is a 2-fold repetition.
    # We leave this as 0.0 for now.

    # Channel 20: Game Phase (Move Number)
    # Normalized by 200 (a "long" game). Clamped at 1.0.
    planes[19].fill(min(board.fullmove_number / 200.0, 1.0))

    # Convert Numpy Array -> PyTorch Tensor
    return torch.from_numpy(planes)


# --- Quick Test Block (Only runs if you execute this file directly) ---
if __name__ == "__main__":
    # Test with the starting position
    start_fen = chess.STARTING_FEN
    tensor = fen_to_tensor(start_fen)

    print(f"Test FEN: {start_fen}")
    print(f"Output Shape: {tensor.shape}")
    print(f"Type: {tensor.dtype}")

    # Check White Pawns (Channel 0) - Should be row 6 (Rank 2)
    print("\nCheck White Pawns (Channel 0, Row 6):")
    print(tensor[0][6])
