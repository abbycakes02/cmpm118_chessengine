import chess
import torch
import numpy as np


def fen_to_tensor(fen_str):
    """
    uses string parsing to convert a FEN string into a 20x8x8 PyTorch Tensor in a performant way.
    performs board flipping, if its black to move, so that the perspective is always from white's side.
    input:
        fen_str: a FEN string
    output:
        torch.Tensor: A 20x8x8 tensor representing the board state
    """

    FRIENDLY_INDICES = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}
    ENEMY_INDICES = {'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

    # split the FEN string into its components
    parts = fen_str.split()
    board_part = parts[0]
    side_to_move = parts[1]
    castling_rights = parts[2]
    en_passant = parts[3]
    halfmove_clock = int(parts[4])
    fullmove_number = int(parts[5])

    is_black_to_move = (side_to_move == 'b')

    tensor = np.zeros((20, 8, 8), dtype=np.float32)

    # iterate over the board part to fill in piece positions
    # channels 0-11: piece positions
    rows = board_part.split('/')

    if is_black_to_move:
        rows = rows[::-1]  # reverse for black to move

    for r, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                # Friendly pieces are treated as white, and get placed in channels 0-5
                # Enemy pieces are treated as black, and get placed in channels 6-11
                if is_black_to_move:
                    # if its black's turn, flip the board to pretend its the white player
                    if char.islower():  # friendly
                        idx = FRIENDLY_INDICES[char.upper()]
                    else:  # enemy
                        idx = ENEMY_INDICES[char.lower()]
                else:
                    # white turn no need to flip
                    if char.isupper():  # friendly
                        idx = FRIENDLY_INDICES[char]
                    else:  # enemy
                        idx = ENEMY_INDICES[char]

                tensor[idx, r, col] = 1.0
                col += 1

    # channel 12: side to move (now redundant, but kept for compatibility)
    # always "your turn" because black is recolord
    tensor[12, :, :].fill(1.0)

    # channels 13-16: castling rights
    # 13: friendly king, 14: friendly queen, 15: enemy king, 16: enemy queen
    if is_black_to_move:
        # friendly/black channels
        if 'k' in castling_rights:
            tensor[13, :, :].fill(1.0)
        if 'q' in castling_rights:
            tensor[14, :, :].fill(1.0)
        # enemy/white channels
        if 'K' in castling_rights:
            tensor[15, :, :].fill(1.0)
        if 'Q' in castling_rights:
            tensor[16, :, :].fill(1.0)
    else:
        # friendly/white channels
        if 'K' in castling_rights:
            tensor[13, :, :].fill(1.0)
        if 'Q' in castling_rights:
            tensor[14, :, :].fill(1.0)
        # enemy channels
        if 'k' in castling_rights:
            tensor[15, :, :].fill(1.0)
        if 'q' in castling_rights:
            tensor[16, :, :].fill(1.0)

    # channel 17: en passant target square
    if en_passant != '-':
        # the chess notation is file + rank, e.g. 'e3'
        # to convert to 0-indexed row/col:
        # use ord to get the unicode difference from 'a' for file
        file_char = en_passant[0]
        rank_char = en_passant[1]
        file_col = ord(file_char) - ord('a')
        rank_row = int(rank_char) - 1  # prep for 0 indexing

        if is_black_to_move:
            rank_row = rank_row
        else:
            rank_row = 7 - rank_row

        tensor[17, rank_row, file_col] = 1.0

    # channel 18: 50-move rule counter (halfmove clock)
    tensor[18, :, :].fill(halfmove_clock / 100.0)
    # channel 19: game phase / total move count (fullmove number)
    tensor[19, :, :].fill(min(fullmove_number / 200.0, 1.0))

    return torch.from_numpy(tensor)


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
