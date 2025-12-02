#!/usr/bin/env python3
"""
Calculate ELO rating from benchmark results against Stockfish levels.

Usage:
    python calculate_elo.py
"""

import math


def calculate_elo(opponent_elo, wins, losses, draws):
    """
    Calculate estimated ELO based on results against an opponent.

    Args:
        opponent_elo: The opponent's ELO rating
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws

    Returns:
        Estimated ELO rating
    """
    total_games = wins + losses + draws

    if total_games == 0:
        return None

    # Calculate score (draws count as 0.5)
    score = (wins + 0.5 * draws) / total_games

    # Handle edge cases
    if score == 1.0:  # Won everything
        print(f"  → Won all games! ELO is at least {opponent_elo + 400}")
        return opponent_elo + 400

    if score == 0.0:  # Lost everything
        print(f"  → Lost all games! ELO is at most {opponent_elo - 400}")
        return opponent_elo - 400

    # Calculate ELO difference using the formula:
    # ELO_diff = -400 * log10(1/score - 1)
    elo_diff = -400 * math.log10(1/score - 1)
    your_elo = opponent_elo + elo_diff

    return your_elo


def main():
    """Main function to calculate ELO from benchmark results."""
    print("=" * 60)
    print("ELO Calculator - Chess Engine Benchmarking")
    print("=" * 60)
    print()

    # Stockfish skill level to approximate ELO mapping
    # Based on: https://chess.stackexchange.com/questions/8123/stockfish-elo-vs-skill-level
    stockfish_levels = {
        1: 800,
        2: 1100,
        3: 1400,
        4: 1700,
        5: 1900,
        6: 2100,
        7: 2300,
        8: 2400,
    }

    print("Enter your benchmark results:")
    print("(Press Enter to skip a level)")
    print()

    elo_estimates = []

    for level, opponent_elo in stockfish_levels.items():
        print(f"\nStockfish Level {level} (≈ {opponent_elo} ELO):")

        try:
            wins_str = input(f"  Wins: ").strip()
            if not wins_str:
                continue

            losses_str = input(f"  Losses: ").strip()
            draws_str = input(f"  Draws: ").strip()

            wins = int(wins_str)
            losses = int(losses_str)
            draws = int(draws_str) if draws_str else 0

            total = wins + losses + draws
            score_pct = (wins + 0.5 * draws) / total * 100 if total > 0 else 0

            print(f"  Total games: {total}")
            print(f"  Score: {score_pct:.1f}%")

            elo = calculate_elo(opponent_elo, wins, losses, draws)

            if elo:
                print(f"  Estimated ELO: {int(elo)}")
                elo_estimates.append(elo)

        except ValueError:
            print("  Invalid input, skipping...")
            continue
        except KeyboardInterrupt:
            print("\n\nCalculation interrupted.")
            break

    print("\n" + "=" * 60)

    if elo_estimates:
        avg_elo = sum(elo_estimates) / len(elo_estimates)
        min_elo = min(elo_estimates)
        max_elo = max(elo_estimates)

        print(f"FINAL ELO ESTIMATE: {int(avg_elo)}")
        print(f"Range: {int(min_elo)} - {int(max_elo)}")
        print(f"Based on {len(elo_estimates)} benchmark(s)")
    else:
        print("No valid benchmark results entered.")

    print("=" * 60)


if __name__ == "__main__":
    main()
