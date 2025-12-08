import chess
import chess.polyglot


FLAG_EXACT = 0
FLAG_LOWERBOUND = 1  # beta cutoff
FLAG_UPPERBOUND = 2  # alpha cutoff


class TranspositionTable:
    """
    fixed-size dictionary to cache previously evaluated positions
    to avoid redundant calculations during minimax search
    """
    def __init__(self, size=1000000):
        self.max_entries = size
        self.table = {}

    def store(self, board, depth, score, flag, best_move):
        """
        store a position in the transposition table
        hash the board position using Zobrist hashing from python-chess

        Args:
            board (chess.Board): current board state
            depth (int): depth of the search when this position was evaluated
            score (float): evaluation score of the position
            flag (int): type of score (exact, lower bound, upper bound)
            best_move (chess.Move): best move found from this position
        """
        key = chess.polyglot.zobrist_hash(board)

        # always overwrite an entry if the new depth is greater or equal
        if key in self.table:
            if self.table[key]['depth'] > depth:
                return

        self.table[key] = {
            'depth': depth,
            'score': score,
            'flag': flag,
            'move': best_move
        }

        # enforce max size
        if len(self.table) > self.max_entries:
            self.table.clear()  # reset the table and start rehashing

    def lookup(self, board):
        """
        lookup a position in the transposition table

        Args:
            board (chess.Board): current board state

        Returns:
            dict or None: entry containing depth, score, flag, best_move if found, else None
        """
        key = chess.polyglot.zobrist_hash(board)
        return self.table.get(key, None)
