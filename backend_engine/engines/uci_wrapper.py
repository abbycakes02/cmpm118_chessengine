#!/usr/bin/env python3
"""
UCI (Universal Chess Interface) wrapper for chess engines.
This script makes the random and minimax engines compatible with UCI protocol,
allowing them to work with chess GUIs and lichess-bot.

Usage:
    python uci_wrapper.py --engine minimax
    python uci_wrapper.py --engine random
"""

import argparse
import chess

# Handle both direct execution and module import
try:
    from . import minimax_engine, random_engine
except ImportError:
    import minimax_engine
    import random_engine


class UCIEngine:
    """
    UCI protocol implementation for chess engines.
    Handles communication between UCI-compatible GUIs/bots and the engine.
    """

    def __init__(self, engine_type="minimax"):
        """
        Initialize UCI engine.

        Args:
            engine_type: "minimax" or "random"
        """
        self.engine_type = engine_type
        if engine_type == "minimax":
            self.default_depth = 5
        self.board = chess.Board()
        self.debug = False

    def log(self, message):
        """Log debug messages if debug mode is enabled."""
        if self.debug:
            print(f"info string {message}", flush=True)

    def uci_initialization(self):
        """Respond to 'uci' command with engine identification."""
        # intial identification
        print(f"id name {self.engine_type.capitalize()}Engine")
        print("id author CMPM118 Chess Team")

        # Options that can be configured
        if self.engine_type == "minimax":
            print(f"option name Depth type spin default {self.default_depth} min 1 max 10")

        print("uciok", flush=True)
    def handle_isready(self):
        """Respond to 'isready' command."""
        print("readyok", flush=True)

    def handle_position(self, tokens):
        """
        Handle 'position' command to set up the board.

        Formats:
            position startpos
            position startpos moves e2e4 e7e5
            position fen <FEN_STRING>
            position fen <FEN_STRING> moves e2e4 e7e5
        """
        try:
            # Find where 'moves' starts (if present)
            moves_index = tokens.index("moves") if "moves" in tokens else None

            # Set up initial position
            if tokens[1] == "startpos":
                self.board = chess.Board()
            elif tokens[1] == "fen":
                # FEN string is from index 2 until "moves" or end
                if moves_index:
                    fen = " ".join(tokens[2:moves_index])
                else:
                    fen = " ".join(tokens[2:])
                self.board = chess.Board(fen)

            # Apply moves if present
            if moves_index:
                for move_str in tokens[moves_index + 1:]:
                    move = chess.Move.from_uci(move_str)
                    self.board.push(move)

            self.log(f"Position set: {self.board.fen()}")

        except Exception as e:
            self.log(f"Error handling position: {e}")

    def handle_go(self, tokens):
        """
        Handle 'go' command to search for best move.

        Formats:
            go depth 5
            go movetime 1000
            go wtime 60000 btime 60000
            go infinite
        """
        # Get default depth (only for minimax)
        depth = getattr(self, 'default_depth', 5)

        # Parse depth if specified
        if "depth" in tokens:
            try:
                depth_index = tokens.index("depth")
                depth = int(tokens[depth_index + 1])
            except (ValueError, IndexError):
                pass

        # Get FEN for current position
        fen = self.board.fen()

        try:
            # Get best move from engine
            if self.engine_type == "minimax":
                self.log(f"Searching at depth {depth}...")
                best_move = minimax_engine.get_best_move(fen, depth=depth)
            elif self.engine_type == "random":
                self.log("Selecting random move...")
                best_move = random_engine.get_random_move(fen)
            else:
                self.log(f"Unknown engine type: {self.engine_type}")
                return

            # Send best move
            print(f"bestmove {best_move}", flush=True)

        except Exception as e:
            self.log(f"Error getting best move: {e}")

    def handle_setoption(self, tokens):
        """
        Handle 'setoption' command to configure engine options.

        Format: setoption name <name> value <value>
        """
        try:
            if "name" in tokens and "value" in tokens:
                name_index = tokens.index("name")
                value_index = tokens.index("value")

                option_name = tokens[name_index + 1].lower()
                option_value = tokens[value_index + 1]

                if option_name == "depth" and self.engine_type == "minimax":
                    self.default_depth = int(option_value)
                    self.log(f"Depth set to {self.default_depth}")
        except Exception as e:
            self.log(f"Error setting option: {e}")

    def run(self):
        """
        Main UCI loop - reads commands from stdin and responds accordingly.
        """
        self.log(f"{self.engine_type.capitalize()} engine starting...")

        while True:
            try:
                # Read command from stdin
                line = input().strip()

                if not line:
                    continue

                tokens = line.split()
                if not tokens:  # Skip if empty
                    continue

                command = tokens[0].lower()

                # Handle UCI commands
                if command == "uci":
                    self.uci_initialization()
                elif command == "isready":
                    self.handle_isready()

                elif command == "ucinewgame":
                    self.board = chess.Board()
                    self.log("New game started")

                elif command == "position":
                    self.handle_position(tokens)

                elif command == "go":
                    self.handle_go(tokens)

                elif command == "setoption":
                    self.handle_setoption(tokens)

                elif command == "debug":
                    if len(tokens) > 1 and tokens[1] == "on":
                        self.debug = True
                    elif len(tokens) > 1 and tokens[1] == "off":
                        self.debug = False

                elif command == "quit":
                    self.log("Engine shutting down")
                    break

                else:
                    self.log(f"Unknown command: {command}")

            except EOFError:
                break
            except Exception as e:
                self.log(f"Error in main loop: {e}")


def main():
    """Parse arguments and run UCI engine."""
    parser = argparse.ArgumentParser(description="UCI Chess Engine")
    parser.add_argument(
        "--engine",
        choices=["minimax", "random"],
        default="minimax",
        help="Engine type to use (default: minimax)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Default search depth for minimax (default: 5)"
    )

    args = parser.parse_args()

    # Create and run UCI engine
    engine = UCIEngine(engine_type=args.engine)

    # Set depth for minimax if specified via command line
    if args.engine == "minimax" and args.depth:
        engine.default_depth = args.depth

    engine.run()


if __name__ == "__main__":
    main()
