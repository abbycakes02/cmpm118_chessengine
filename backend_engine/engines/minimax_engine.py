import chess

# piece values use UCI centipawn scoring
# https://www.chessprogramming.org/Score
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,  # bishops slightly more valuable than knights
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,  # arbitrary high value for king
}

MATE_SCORE = 99999  # score for checkmate


def evaluate_board(board: chess.Board) -> int:
    """
    place holder function to get a score for the chess engine
    this block will be replaced later by the NN value engine

    Args:
        board (chess.Board): The current board state handled by python-chess

    Returns:
        int: score of the board from white's perspective, positive means white advantage, negative means black advantage
    """
    score = 0

    if board.is_checkmate():
        # if it's white's turn and checkmate, black wins
        if board.turn == chess.WHITE:
            return -MATE_SCORE  # Black wins
        else:
            return MATE_SCORE   # White wins

    # game is over but not checkmate
    if board.is_game_over():
        return 0  # Draw or stalemate

    # sum material values using pychess' bitboard based representation
    # loop through each piece type and count pieces for both sides
    for piece_type in PIECE_VALUES.keys():
        # add the value of each white piece
        score += PIECE_VALUES[piece_type] * len(board.pieces(piece_type, chess.WHITE))
        # subtract the value of each black piece
        score -= PIECE_VALUES[piece_type] * len(board.pieces(piece_type, chess.BLACK))
    return score


def minimax(board, depth, alpha, beta, maximizing_player):
    """
    driving minimax engine behind the chess engine
    recursively explores legal moves until a certain depth,
    alternating maximizing and minimizing player
    returns the maximizing or minimizing value at each node

    Args:
        board (chess.Board): The current board state handled by python-chess
        depth (int): Depth to explore in the game tree
        alpha (int): Alpha value for alpha-beta pruning (score of maximizing player)
        beta (int): Beta value for alpha-beta pruning (score of minimizing player)
        maximizing_player (bool): True if the current move is for the maximizing player
    Returns:
        int: The minimax value of the board state
    """
    # depth limit reached or game over, return evaluation of the board
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    # get all legal moves and explore them
    legal_moves = list(board.legal_moves)

    # sort moves to evaluate capture moves first to improve pruning
    legal_moves.sort(key=lambda move: board.is_capture(move), reverse=True)

    # maximizing player's turn
    if maximizing_player:
        max_eval = -float('inf')  # init to negative infinity
        for move in legal_moves:

            # make the first move
            board.push(move)
            # iterate down the tree for minimizing player
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            # undo the move to restore board state
            board.pop()

            # update max_eval and alpha for pruning
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)

            # if the minimizing player is beating the maximizing player, abandon the branch
            if beta <= alpha:
                break  # Beta cut-off
        # return the highest score found
        return max_eval
    else:
        # minimizing player's turn
        min_eval = float('inf')  # init to positive infinity
        for move in legal_moves:

            # make the first move
            board.push(move)
            # iterate down the tree for maximizing player
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            # undo the move to restore board state
            board.pop()

            # update min_eval and beta for pruning
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            # if the minimizing player is beating the maximizing player, abandon the branch
            if beta <= alpha:
                break  # Alpha cut-off
        # return the best lowest found
        return min_eval


def get_best_move(fen: str, depth: int = 5) -> str:
    """
    runs the minimax engine to the specified depth and returns the best move in UCI format
    based on the minimax algorithm with alpha-beta pruning from Geeks for Geeks:
    https://www.geeksforgeeks.org/artificial-intelligence/mini-max-algorithm-in-artificial-intelligence/

    Args:
        fen (str): FEN string of the current board state
        depth (int, optional): Depth to explore in the game tree. Defaults to 2.

    Returns:
        str: Best move in UCI format
    """
    print(f"Processing FEN: {fen} at depth {depth}...")
    # construct pychess board
    board = chess.Board(fen)

    # check if game is over
    if board.is_game_over():
        print("Game is over")
        raise ValueError("Game is over")

    best_move = None
    legal_moves = list(board.legal_moves)

    # sort root nodes to make pruining more efficient
    legal_moves.sort(key=lambda move: board.is_capture(move), reverse=True)

    # start search from root
    if board.turn == chess.WHITE:
        # black is always the minimizing player
        max_eval = -float('inf')
        for move in legal_moves:
            # make first move
            board.push(move)
            # kick of minimax for black turn
            eval_score = minimax(board, depth - 1, -float('inf'), float('inf'), False)
            board.pop()

            # compare saved evaluation score to new score to find best move
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
    else:
        # white is always the maximizing player
        min_eval = float('inf')
        for move in legal_moves:
            # make first move
            board.push(move)
            # kick of minimax for white turn
            eval_score = minimax(board, depth - 1, -float('inf'), float('inf'), True)
            board.pop()

            # compare saved evaluation score to new score to find best move
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

    # last check if a move was found
    if best_move is None:
        print("No legal moves found")
        raise ValueError("No legal moves available")
    print(f"Best Move: {best_move.uci()}")
    return best_move.uci()
