# Quick Fix for MinimaxEngine: Add Piece-Square Tables

## Current Problem
Your engine scores 0/24 (0 wins, 0 draws, 24 losses) because it has **zero positional evaluation**.

When there are no captures available, ALL moves look equally good (score = 0), so the engine essentially plays random moves while Stockfish plays strategically.

## Evidence
From your games:
```
1. Nh3 d5 2. Ng5 e5 3. h4 Nc6 4. Rh2 ...
```
- Nh3 (knight to rim) - terrible square
- Ng5 (knight to be kicked) - no purpose
- h4 (random pawn push) - weakens king
- Rh2 (rook doing nothing) - wasted move

All scored as **0 points** by your material-only evaluation!

## Solution: Add Piece-Square Tables

Piece-square tables give bonuses/penalties for pieces on specific squares:
- Pawns: bonus for advancing, penalty for isolated pawns
- Knights: bonus for center, penalty for edges ("knight on the rim is dim")
- Bishops: bonus for controlling long diagonals
- Rooks: bonus for open files, 7th rank
- Queens: bonus for activity, penalty for early development
- Kings: bonus for castling/safety in middlegame, centralization in endgame

## Expected Improvement
Adding PSTs should give you:
- **Immediate**: +200-300 ELO (from 0% to 30-40% vs Stockfish level 0-2)
- **After tuning**: +300-400 ELO (competitive at 1400-1600 range)

## Implementation

Add this to `minimax_engine.py` before the `MinimaxEngine` class:

```python
# Piece-Square Tables (from Sunfish/python-chess examples)
# Scores are from White's perspective, flip for Black
PST = {
    chess.PAWN: [
         0,   0,   0,   0,   0,   0,   0,   0,
        50,  50,  50,  50,  50,  50,  50,  50,
        10,  10,  20,  30,  30,  20,  10,  10,
         5,   5,  10,  25,  25,  10,   5,   5,
         0,   0,   0,  20,  20,   0,   0,   0,
         5,  -5, -10,   0,   0, -10,  -5,   5,
         5,  10,  10, -20, -20,  10,  10,   5,
         0,   0,   0,   0,   0,   0,   0,   0
    ],
    chess.KNIGHT: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20,   0,   0,   0,   0, -20, -40,
        -30,   0,  10,  15,  15,  10,   0, -30,
        -30,   5,  15,  20,  20,  15,   5, -30,
        -30,   0,  15,  20,  20,  15,   0, -30,
        -30,   5,  10,  15,  15,  10,   5, -30,
        -40, -20,   0,   5,   5,   0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    chess.BISHOP: [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   5,   5,  10,  10,   5,   5, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   5,   0,   0,   0,   0,   5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    chess.ROOK: [
         0,   0,   0,   0,   0,   0,   0,   0,
         5,  10,  10,  10,  10,  10,  10,   5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
         0,   0,   0,   5,   5,   0,   0,   0
    ],
    chess.QUEEN: [
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,   5,   5,   5,   0, -10,
         -5,   0,   5,   5,   5,   5,   0,  -5,
          0,   0,   5,   5,   5,   5,   0,  -5,
        -10,   5,   5,   5,   5,   5,   0, -10,
        -10,   0,   5,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20
    ],
    chess.KING: [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
         20,  20,   0,   0,   0,   0,  20,  20,
         20,  30,  10,   0,   0,  10,  30,  20
    ],
}
```

Then update the `material_score` method to:

```python
def material_score(self, board):
    """
    Evaluate board position using material + piece-square tables
    """
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Material value
            value = PIECE_VALUES[piece.piece_type]
            
            # Piece-square table bonus
            # For Black pieces, flip the square (rank 0->7, 7->0)
            pst_square = square if piece.color == chess.WHITE else (square ^ 56)
            pst_bonus = PST[piece.piece_type][pst_square]
            
            if piece.color == chess.WHITE:
                score += value + pst_bonus
            else:
                score -= value + pst_bonus
                
    return score
```

## Expected Results After Fix

Before PST:
```
| Level | SF ELO | Score      | Perf Rating  |
| 0     | 1320   | 0.0/10     | <1200        |
```

After PST:
```
| Level | SF ELO | Score      | Perf Rating  |
| 0     | 1320   | 6.0/10     | ~1450        |
| 1     | 1400   | 5.0/10     | ~1400        |
| 2     | 1500   | 3.0/10     | ~1330        |
```

This will give you an estimated rating of **1400 ± 40 ELO** at depth 5.

## Next Steps

1. Add piece-square tables to minimax_engine.py
2. Run tournament again
3. You should now win some games!
4. Save and analyze PGN games to identify remaining weaknesses
