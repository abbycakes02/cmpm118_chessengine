import chess
import time

from ml.inference import ChessEvaluator

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


class MinimaxEngine:
    """
    Minimax chess engine with alpha-beta pruning
    """

    def __init__(self, use_nn=False, model_path=None, channels=64, blocks=5, max_depth=3, max_time=None):
        """
        initialize the minimax engine

        Args:
            depth (int): depth to explore in the game tree
        """
        self.use_nn = use_nn
        self.model_path = model_path
        self.channels = channels
        self.blocks = blocks
        self.max_depth = max_depth
        self.max_time = max_time
        self.evaluator = None
        self.nodes_searched = 0
        if use_nn and model_path is not None:
            try:
                self.evaluator = ChessEvaluator(model_path, channels, blocks)
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
            eval_score = self.evaluator.evaluate_position(board)
            # convert from -1 to 1 range to centipawn scale
            return int(eval_score * 20000)

        # if were not using NN, use material score
        score = self.material_score(board)
        return score

    def minimax(self, board, depth, alpha, beta, maximizing_player, stop_time):
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
        if self.max_time is not None and self.nodes_searched % 1000 == 0:
            if time.time() > stop_time:
                raise TimeoutError("Minimax search timed out")

        # depth limit reached or game over, return evaluation of the board
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

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
                eval_score = self.minimax(board, depth - 1, alpha, beta, False, stop_time)
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
                eval_score = self.minimax(board, depth - 1, alpha, beta, True, stop_time)
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

        # check if game is over
        if board.is_game_over():
            print("Game is over")
            raise ValueError("Game is over")

        # start timer if time limit is set
        stop_time = None
        if time_limit is not None:
            start_time = time.time()
            stop_time = start_time + time_limit

        best_move = None
        self.nodes_searched = 0
        # if no max depth is provided rely on either the time limit or try to look 50 moves ahead
        max_depth = max_depth if max_depth is not None else 50

        print(f"starting minimax using {'NN evaluator' if self.use_nn else 'material evaluator'}")
        print(f"Max Depth: {max_depth}, Time Limit: {time_limit} seconds")

        for curr_depth in range(1, max_depth + 1):
            print(f" Searching at depth {curr_depth}...")
            try:
                # check if we're within a couple miliseconds of the time limit before starting a new depth
                if (time.time() - start_time) > time_limit * 0.9:
                    print(f"Approaching time limit, stopping search before depth {curr_depth}")
                    break

                curr_best_move = None
                legal_moves = list(board.legal_moves)

                # check if a best move was found at the previous depth
                # if so explore it first triggering the pruning to remove most other branches
                if best_move is not None:
                    print(f" Previous best move: {best_move.uci()}")
                    # move the best move to the front of the list to explore it first
                    legal_moves.remove(best_move)
                    legal_moves.insert(0, best_move)
                else:
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
                        eval_score = self.minimax(board, curr_depth - 1, -float('inf'), float('inf'), False, stop_time)
                        board.pop()

                        # compare saved evaluation score to new score to find best move
                        if eval_score > max_eval:
                            max_eval = eval_score
                            curr_best_move = move
                else:
                    # white is always the maximizing player
                    min_eval = float('inf')
                    for move in legal_moves:
                        # make first move
                        board.push(move)
                        # kick of minimax for white turn
                        eval_score = self.minimax(board, curr_depth - 1, -float('inf'), float('inf'), True, stop_time)
                        board.pop()

                        # compare saved evaluation score to new score to find best move
                        if eval_score < min_eval:
                            min_eval = eval_score
                            curr_best_move = move
                best_move = curr_best_move

                print(f" Depth {curr_depth} best move: {best_move.uci()}, Nodes searched: {self.nodes_searched}")

            except TimeoutError:
                print(f"Time limit reached at depth {curr_depth}, stopping search.")
                break

        # last check if a move was found
        if best_move is None:
            print("No legal moves found")
            # fall back to first legal move
            best_move = list(board.legal_moves)[0]

        print(f"Best Move: {best_move.uci()}")
        return best_move.uci()
