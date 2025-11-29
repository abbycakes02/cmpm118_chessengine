import chess
import torch


def fen_to_tensor(fen_str):
    """
    Wraps board_to_tensor to accept FEN strings as input.

    input:
        fen_str: a FEN string
    output:
        torch.Tensor: A 20x8x8 tensor representing the board state
    """

    if not isinstance(fen_str, str):
        raise ValueError("Input must be a FEN string.")

    board = chess.Board(fen_str)
    return board_to_tensor(board)


def board_to_tensor(board_state):
    """
    Converts a chess.Board object into a 20x8x8 PyTorch Tensor.

    input:
        board_state: a chess.Board object
    output:
        torch.Tensor: A 20x8x8 tensor representing the board state
        Dimensions: [20, 8, 8]
        - Channels 0-5:   White Pieces (Pawn, Knight, Bishop, Rook, Queen, King)
        - Channels 6-11:  Black Pieces (Pawn, Knight, Bishop, Rook, Queen, King)
        - Channel 12:     Side to Move (1.0 = White, 0.0 = Black)
        - Channels 13-16: Castling Rights (White Kingside, White Queenside, Black Kingside, Black Queenside)
        - Channel 17:     En Passant Target Square
        - Channel 18:     50-Move Rule Counter (Normalized)
        - Channel 19:     Game Phase / Total Move Count (Normalized)
    """
    # initializing an empty tensor to have a clean state
    tensor = torch.zeros((20, 8, 8), dtype=torch.float32)

    # look thru each piece on board and set the corresponding values
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    # colors = [chess.WHITE, chess.BLACK]
    for i, piece_type in enumerate(piece_types):
        # White pieces (channels 0-5)
        for square in board_state.pieces(piece_type, chess.WHITE):
            rank, file = divmod(square, 8)
            tensor[i, 7 - rank, file] = 1.0

        # Black pieces (channels 6-11)
        for square in board_state.pieces(piece_type, chess.BLACK):
            rank, file = divmod(square, 8)
            tensor[i + 6, 7 - rank, file] = 1.0

    # Channel 12: Side to Move (1.0 for White, 0.0 for Black)
    if board_state.turn == chess.WHITE:
        tensor[12].fill_(1.0)

    # Channels 13-16: Castling Rights
    if board_state.has_kingside_castling_rights(chess.WHITE):
        tensor[13].fill_(1.0)
    if board_state.has_queenside_castling_rights(chess.WHITE):
        tensor[14].fill_(1.0)
    if board_state.has_kingside_castling_rights(chess.BLACK):
        tensor[15].fill_(1.0)
    if board_state.has_queenside_castling_rights(chess.BLACK):
        tensor[16].fill_(1.0)

    # Channel 17: En Passant Target
    if board_state.ep_square:
        rank, file = divmod(board_state.ep_square, 8)
        tensor[17, 7 - rank, file] = 1.0

    # Channel 18: 50-Move Rule (Halfmove Clock)
    # Normalized by 100.0 to keep it roughly 0-1
    tensor[18].fill_(board_state.halfmove_clock / 100.0)

    # Channel 19: Game Phase (Move Number)
    # Normalized by 200 (a "long" game). Clamped at 1.0.
    tensor[19].fill_(min(board_state.fullmove_number / 200.0, 1.0))

    return tensor
