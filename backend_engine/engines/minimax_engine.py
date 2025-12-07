import chess
import time

# import os
# import random
import chess.polyglot

from ml.inference import ChessEvaluator
from ml.vocab import MOVE_TO_INT
from engines.transposition_table import TranspositionTable, FLAG_EXACT, FLAG_LOWERBOUND, FLAG_UPPERBOUND

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

# Piece-square tables (opening/midgame values)
PAWN_PST = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

KNIGHT_PST = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
]

BISHOP_PST = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
]

ROOK_PST = [
    [0, 0, 0, 5, 5, 0, 0, 0],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

QUEEN_PST = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20]
]

KING_PST = [
    [20, 30, 10, 0, 0, 10, 30, 20],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30]
]

CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]

BOOK_FILENAME = "gm2001.bin"


class MinimaxEngine:
    """
    Minimax chess engine with alpha-beta pruning
    """

    def __init__(self, use_nn=False, model_path=None, tt=None, channels=64, blocks=5, max_depth=3, max_time=None, k_moves=5):
        """
        initialize the minimax engine

        Args:
            use_nn (bool): whether to use the neural network evaluator
            model_path (str): path to the trained neural network model
            channels (int): number of channels in the neural network
            blocks (int): number of residual blocks in the neural network
            max_depth (int): depth to explore in the game tree
            max_time (float): maximum time to search in seconds
            k_moves (int): number of top moves to consider at each step
        """
        self.use_nn = use_nn
        self.model_path = model_path
        self.channels = channels
        self.blocks = blocks
        self.max_depth = max_depth
        self.max_time = max_time
        self.k_moves = k_moves
        self.evaluator = None
        self.nodes_searched = 0
        self.book_path = BOOK_FILENAME
        self.tt = tt if tt is not None else TranspositionTable(size=1000000)

        if use_nn and model_path is not None:
            try:
                self.evaluator = ChessEvaluator(
                    model_path=self.model_path,
                    input_channels=20,
                    hidden_channels=self.channels,
                    blocks=self.blocks,
                    history_length=5
                    )
                print("Neural network evaluator loaded successfully.")
            except Exception as e:
                print(f"Failed to load neural network model: {e}")

    def material_score(self, board):
        """
        evaluate the board position using a simple material count

        Args:
            board (chess.Board): The current board state handled by python-chess
        Returns:
            int: The evaluation score of the board
        """
        score = 0
        # sum material values using pychess' bitboard based representation
        # loop through each piece type and count pieces for both sides
        for piece_type in PIECE_VALUES.keys():
            # add the value of each white piece
            score += PIECE_VALUES[piece_type] * len(board.pieces(piece_type, chess.WHITE))
            # subtract the value of each black piece
            score -= PIECE_VALUES[piece_type] * len(board.pieces(piece_type, chess.BLACK))
        return score

    def better_eval_func(self, board):
        if board.is_checkmate():
            return MATE_SCORE if board.turn == chess.BLACK else -MATE_SCORE
        if board.is_game_over():
            return 0

        score = 0
        # Material + piece-square table
        for piece_type, pst in [(chess.PAWN, PAWN_PST),
                                (chess.KNIGHT, KNIGHT_PST),
                                (chess.BISHOP, BISHOP_PST),
                                (chess.ROOK, ROOK_PST),
                                (chess.QUEEN, QUEEN_PST),
                                (chess.KING, KING_PST)]:
            for sq in board.pieces(piece_type, chess.WHITE):
                row, col = divmod(sq, 8)
                score += PIECE_VALUES[piece_type] + pst[row][col]
            for sq in board.pieces(piece_type, chess.BLACK):
                row, col = divmod(sq, 8)
                score -= PIECE_VALUES[piece_type] + pst[7 - row][col]  # flip board for black

        # King safety
        for color in [chess.WHITE, chess.BLACK]:
            factor = 1 if color == chess.WHITE else -1
            king_sq = board.king(color)
            if king_sq is not None:
                # simple safety: penalize missing pawn shield
                for f in [chess.square_file(king_sq)-1, chess.square_file(king_sq), chess.square_file(king_sq)+1]:
                    if 0 <= f <= 7:
                        r = 1 if color == chess.WHITE else 6
                        sq = chess.square(f, r)
                        p = board.piece_at(sq)
                        if not (p and p.piece_type == chess.PAWN and p.color == color):
                            score -= factor * 10

        # Pawn structure
        for color in [chess.WHITE, chess.BLACK]:
            factor = 1 if color == chess.WHITE else -1
            pawns = board.pieces(chess.PAWN, color)
            for sq in pawns:
                file = chess.square_file(sq)
                # isolated pawns
                left = board.pieces(chess.PAWN, color) & chess.BB_FILES[file-1] if file > 0 else 0
                right = board.pieces(chess.PAWN, color) & chess.BB_FILES[file+1] if file < 7 else 0
                if left == 0 and right == 0:
                    score -= factor * 15

        # Center control
        for color in [chess.WHITE, chess.BLACK]:
            factor = 1 if color == chess.WHITE else -1
            for sq in CENTER_SQUARES:
                p = board.piece_at(sq)
                if p and p.color == color:
                    if p.piece_type == chess.KNIGHT:
                        score += factor * 20
                    elif p.piece_type == chess.BISHOP:
                        score += factor * 10
                    elif p.piece_type == chess.PAWN:
                        score += factor * 5

        return score

    def evaluate_board(self, board):
        """
        evaluate the board position
        uses either the material score or the neural network evaluator

        Args:
            board (chess.Board): The current board state handled by python-chess
        Returns:
            int: The evaluation score of the board
        """
        if board.is_checkmate():
            # if it's white's turn and checkmate, black wins
            if board.turn == chess.WHITE:
                return -MATE_SCORE  # Black wins
            else:
                return MATE_SCORE   # White wins

        # game is over but not checkmate
        if board.is_game_over():
            return 0  # Draw or stalemate

        if self.use_nn and self.evaluator is not None:
            # use neural network evaluator
            fen = board.fen()
            _, eval_score = self.evaluator.evaluate_position(fen)
            # remove "hypothetical" last move from history
            self.evaluator.pop_history()
            # convert from -1 to 1 range to centipawn scale
            return int(eval_score * 1000)

        # if were not using NN, use material score
        # score = self.material_score(board)
        score = self.better_eval_func(board)

        return score

    def sort_moves_by_evaluation(self, board, legal_moves, debug=False):
        """
        run nn evaluator on the legal moves and sort the moves by probability
        """
        policy_scores, _ = self.evaluator.evaluate_position(board.fen())
        self.evaluator.pop_history()

        move_probs = []
        for move in legal_moves:
            move_uci = move.uci()
            if move_uci in MOVE_TO_INT:
                move_index = MOVE_TO_INT[move_uci]
                move_prob = policy_scores[move_index]
            else:
                move_prob = 0.0
            move_probs.append((move, move_prob))
        # sort moves by probability in descending order
        move_probs.sort(key=lambda x: x[1], reverse=True)
        # return only the sorted moves
        sorted_moves = [move for move, _ in move_probs]

        if debug:
            top_3 = move_probs[:3]
            moves_str = ", ".join([f"{m.uci()}({p:.2f})" for m, p in top_3])
            print(f"   🧠 NN Policy Top 3: {moves_str}")

        return sorted_moves

    def minimax(self, board, depth, alpha, beta, maximizing_player, stop_time, root_depth=0):
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
        # increment nodes searched
        self.nodes_searched += 1

        # check for timeout every 1000 nodes
        if self.max_time is not None and self.nodes_searched % 100 == 0:
            if time.time() > stop_time:
                raise TimeoutError("Minimax search timed out")

        indent = " |  " * (root_depth)

        # check transposition table for cached evaluation
        tt_entry = self.tt.lookup(board)
        tt_move = None

        if tt_entry:
            tt_move = tt_entry['move']
            # check if we can use the cached score
            if tt_entry['depth'] >= depth:
                flag = tt_entry['flag']
                score = tt_entry['score']
                if flag == FLAG_EXACT:
                    return score
                elif flag == FLAG_LOWERBOUND:
                    alpha = max(alpha, score)
                elif flag == FLAG_UPPERBOUND:
                    beta = min(beta, score)

                # cut-off check to avoid unnecessary search
                if alpha >= beta:
                    return score

        # depth limit reached or game over, return evaluation of the board
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

        # get all legal moves and explore them
        legal_moves = list(board.legal_moves)

        # if we have a tt move, consider it first
        if tt_move:
            try:
                move_obj = chess.Move.from_uci(tt_move)
                if move_obj in legal_moves:
                    legal_moves.remove(move_obj)
                    legal_moves.insert(0, move_obj)
            except Exception:
                print("Invalid TT move UCI:", tt_move)
                pass

        # if using nn evaluator, sort moves by probability and consider only top k moves
        if self.use_nn and self.evaluator is not None and root_depth < 2:
            # show_nn_info = (root_depth < 1)
            sorted_moves = self.sort_moves_by_evaluation(board, legal_moves, debug=False)
            if tt_move:
                # ensure tt move is still first
                move_obj = chess.Move.from_uci(tt_move)
                if move_obj in sorted_moves:
                    sorted_moves.remove(move_obj)
                    sorted_moves.insert(0, move_obj)
            legal_moves = sorted_moves[:self.k_moves]
        else:
            # otherwise, sort moves to consider captures first for better pruning
            legal_moves.sort(key=lambda move: board.is_capture(move), reverse=True)

        # maximizing player's turn
        best_local_move = None
        original_alpha = alpha

        if maximizing_player:
            value = -float('inf')  # init to negative infinity
            for move in legal_moves:

                # make the first move
                board.push(move)
                # iterate down the tree for minimizing player
                eval_score = self.minimax(board, depth - 1, alpha, beta, False, stop_time, root_depth=root_depth + 1)
                # undo the move to restore board state
                board.pop()

                # update max_eval and alpha for pruning
                if eval_score > value:
                    value = eval_score
                    best_local_move = move
                    if root_depth == 0:
                        # at root node, print the move being considered
                        print(f"{indent} New best (white) move {move.uci()}, score: {eval_score}")
                alpha = max(alpha, value)

                # if the minimizing player is beating the maximizing player, abandon the branch
                if beta <= alpha:
                    break  # Beta cut-off
        else:
            # minimizing player's turn
            value = float('inf')  # init to positive infinity
            for move in legal_moves:

                # make the first move
                board.push(move)
                # iterate down the tree for maximizing player
                eval_score = self.minimax(board, depth - 1, alpha, beta, True, stop_time, root_depth=root_depth + 1)
                # undo the move to restore board state
                board.pop()

                # update min_eval and beta for pruning
                if eval_score < value:
                    value = eval_score
                    best_local_move = move
                    if root_depth == 0:
                        # at root node, print the move being considered
                        print(f"{indent} New best (black) move {move.uci()}, score: {eval_score}")
                beta = min(beta, value)
                # if the minimizing player is beating the maximizing player, abandon the branch
                if beta <= alpha:
                    break  # Alpha cut-off

        # store the evaluated position in the transposition table
        tt_flag = FLAG_EXACT
        if value <= original_alpha:
            tt_flag = FLAG_UPPERBOUND
        elif value >= beta:
            tt_flag = FLAG_LOWERBOUND

        move_str = best_local_move.uci() if best_local_move else None
        self.tt.store(board, depth, value, tt_flag, move_str)

        return value

    def get_book_move(self, board):
        try:
            with chess.polyglot.open_reader(self.book_path) as reader:
                entry = reader.find(board)
                if entry is not None:
                    # entry.move is a property, not a function
                    return entry.move.uci()
        except Exception as e:
            print("Polyglot book read error:", e)
        return None

    def get_move(self, fen, time_limit=None, max_depth=3):
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
        # construct pychess board
        board = chess.Board(fen)

        # reset the history becuase we're starting a new search
        if self.use_nn and self.evaluator is not None:
            self.evaluator.reset_history()

        # OPENING BOOK CHECK
        print("===Checking if Book move===")
        book_move = self.get_book_move(board)
        if book_move:
            print("Using book move:", book_move)
            return book_move

        print("===Not a book move===")
        # check if game is over
        if board.is_game_over():
            print("Game is over")
            raise ValueError("Game is over")

        # start timer if time limit is set
        stop_time = time.time() + time_limit if time_limit is not None else None
        start_time = time.time()
        best_move = None
        self.nodes_searched = 0
        # if no max depth is provided rely on either the time limit or try to look 50 moves ahead
        max_depth = max_depth if max_depth is not None else 50

        print(f"starting minimax using {'NN evaluator' if self.use_nn else 'material evaluator'}, tt size: {len(self.tt.table)}")
        print(f"Max Depth: {max_depth}, Time Limit: {time_limit} seconds")

        for curr_depth in range(1, max_depth + 1):
            print(f" Searching at depth {curr_depth}...")
            try:
                if stop_time and (time.time() > stop_time - 0.2):
                    print("Time limit reached before starting depth", curr_depth)
                    break

                if board.turn == chess.WHITE:
                    self.minimax(board, curr_depth, -float('inf'), float('inf'), True, stop_time, root_depth=0)
                else:
                    self.minimax(board, curr_depth, -float('inf'), float('inf'), False, stop_time, root_depth=0)

                # retrieve best move from transposition table
                tt_entry = self.tt.lookup(board)
                if tt_entry and tt_entry['move']:
                    best_move = chess.Move.from_uci(tt_entry['move'])
                    print(f" Depth {curr_depth} | Best: {best_move.uci()} | Score: {tt_entry['score']} | Nodes: {self.nodes_searched}, in {time.time() - start_time:.2f} seconds")
            except TimeoutError:
                print(f"Time limit reached at depth {curr_depth}, stopping search.")
                break

        # last check if a move was found
        if best_move is None:
            print("No legal moves found")
            # fall back to first legal move
            best_move = list(board.legal_moves)[0]

        print(f" Minimax picked move: {best_move.uci()}, in {time.time() - start_time:.2f} seconds, nodes searched: {self.nodes_searched}")
        return best_move.uci()
