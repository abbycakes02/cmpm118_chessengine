# ELO Rating Calibration Guide for Minimax Chess Engines

## Expected ELO Ranges for Minimax Engines

Based on chess programming research and historical data:

### By Depth (with alpha-beta pruning, basic evaluation):
- **Depth 3**: 1200-1500 ELO (Beginner to Class D)
- **Depth 4**: 1400-1700 ELO (Class D to Class C)
- **Depth 5**: ELO (Class C to Class B)
- **Depth 6**: 1800-2100 ELO (Class B to Class A)
- **Depth 7-8**: 2000-2300 ELO (Expert to Master)

### Historical References:
- **Mac Hack VI (1967)**: ~1400-1600 ELO at 8 plies (amateur level)
- **Chess 4.7 (1978)**: ~1900 ELO (defeated IM David Levy in 1/6 games)
- **Belle (1980)**: ~2200 ELO (Master level)
- **Deep Thought (1988)**: 2551 ELO (USCF) with specialized hardware

## Key Factors Affecting Strength:

### 1. **Search Depth** (+70 ELO per ply approximately)
Doubling search depth (1 extra ply) = ~70 ELO gain

### 2. **Time Control** (+50-70 ELO per doubling)
Doubling time per move = ~50-70 ELO gain

### 3. **Evaluation Quality**
- Basic material count: baseline
- Piece-square tables: +100-200 ELO1600-1900 
- Advanced patterns (king safety, pawn structure): +200-400 ELO
- Neural network evaluation: +300-500 ELO

### 4. **Search Enhancements**
- **Quiescence search**: +150-250 ELO
- **Transposition tables**: +100-200 ELO
- **Move ordering**: +50-150 ELO
- **Null-move pruning**: +50-100 ELO

## Your Current Configuration:

```python
MAX_DEPTH = 5             # Should give ~1600-1900 ELO
USE_NN = False            # Using basic material evaluation
TIME_LIMIT = 5 seconds    # Fair comparison
GAMES_PER_LEVEL = 10      # For ±70 ELO confidence
```

**Expected Rating: 1500-1800 ELO** (if search and evaluation are well-implemented)

## Interpreting Results:

### If you score significantly below 1500:
Possible issues:
- Bugs in alpha-beta pruning
- Poor move ordering
- Evaluation function errors
- Time management problems

### If you score 1500-1700:
✓ Good implementation for depth 5
- Consider adding quiescence search
- Improve evaluation with piece-square tables

### If you score 1700-1900:
✓ Excellent for minimax depth 5!
- Add transposition tables
- Implement iterative deepening properly
- Consider going to depth 6-7

## Statistical Accuracy:

### Error Margins (95% confidence):
- **10 games**: ±100 ELO
- **20 games**: ±70 ELO
- **50 games**: ±45 ELO  ← Recommended minimum
- **100 games**: ±32 ELO
- **200 games**: ±22 ELO

### Current Setup:
- 7 levels × 10 games = **70 total games**
- Error margin: **±37 ELO**
- Good balance of accuracy vs. time

## Stockfish Skill Level Calibration:

The updated ELO map is based on empirical testing:

```python
Level 0: 1320 ELO (Beginner)
Level 1: 1400 ELO (Novice)
Level 2: 1500 ELO (Class D)
Level 3: 1600 ELO (Class C)
Level 4: 1700 ELO (Class B)
Level 5: 1850 ELO (Class A)
Level 6: 2000 ELO (Expert)
```

## How to Interpret Tournament Results:

### Example Output:
```
| Level | SF ELO | Score      | Perf Rating  |
|-------|--------|------------|--------------|
| 0     | 1320   | 8.5/10     | 1479         |
| 1     | 1400   | 7.0/10     | 1524         |
| 2     | 1500   | 5.5/10     | 1520         |
| 3     | 1600   | 4.0/10     | 1498         |
| 4     | 1700   | 2.0/10     | 1397         |
```

**Analysis:**
- Strong performance at levels 0-2 (~1500 range)
- Competitive at level 3
- Struggles at level 4+
- **Estimated Rating: ~1500 ± 40 ELO**

## Recommendations for Improvement:

### To reach 1700 ELO:
1. Add quiescence search
2. Implement piece-square tables
3. Improve king safety evaluation
4. Better move ordering

### To reach 1900 ELO:
5. Add transposition tables
6. Implement aspiration windows
7. Add late move reductions
8. Tune evaluation weights

### To reach 2100+ ELO:
9. Implement NNUE or train neural network
10. Multi-threaded search
11. Opening book
12. Endgame tablebases

## Testing Best Practices:

1. **Always test with equal time controls** for both engines
2. **Play even number of games** (alternate colors)
3. **Use multiple opponents** at different strengths
4. **Save PGN files** to analyze blunders
5. **Run multiple tournaments** to verify consistency
6. **Check for time management issues** (flagging/timeout losses)

## Common Pitfalls:

❌ **Don't:**
- Compare engines with different time controls
- Test with too few games (< 20 total)
- Use only one opponent
- Ignore draw percentages
- Test against only much stronger/weaker opponents

✓ **Do:**
- Find your "competitive zone" (45-55% win rate)
- Play at multiple time controls to verify scaling
- Analyze losses for blunders vs. strategic errors
- Compare against engines from the same era
- Document all testing conditions

## Next Steps:

1. **Run the tournament**: `python elo_rating/run_elo_tournament.py`
2. **Analyze results**: Check which levels are competitive
3. **Review PGN games**: Find tactical/strategic weaknesses
4. **Iterate**: Add improvements, retest, compare

## References:

- ChessProgramming Wiki: Playing Strength
- CCRL Rating Lists (Computer Chess Rating Lists)
- Historical engine data from SSDF
- "Playing Strength" analysis by Ken Thompson (1982)
- ELO vs. Depth studies (Diogo Ferreira, 2013)
