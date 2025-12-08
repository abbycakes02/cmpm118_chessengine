# Define all possible squares on a chessboard

squares = [
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
]


def build_vocab():
    """
    helpers to build the move vocabulary mapping from move strings to integer indices
    """
    moves = []
    # Generate all from-to combinations as a hashmap to convert uci moves to integers
    for src in squares:
        for dst in squares:
            if src == dst:
                continue
            moves.append(src + dst)
            # Add promotions for rank 7->8 and 2->1
            if (src[1] == '7' and dst[1] == '8') or (src[1] == '2' and dst[1] == '1'):
                for p in ["q", "r", "b", "n"]:
                    moves.append(src + dst + p)

    moves.sort()
    return {m: i for i, m in enumerate(moves)}, len(moves)


# Create the global maps
MOVE_TO_INT, VOCAB_SIZE = build_vocab()
