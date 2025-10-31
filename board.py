import chess
import chess.svg
import random

board = chess.Board()
SVG_PATH = "board.svg"

def render():
    svg = chess.svg.board(board=board)
    with open(SVG_PATH, "w", encoding="utf-8") as f:
        f.write(svg)
    print(board)
    print()

def get_user_move():
    while True:
        try:
            print("All possible moves:")
            for move in board.legal_moves:
                print(move, end=' ')
            print()
            move_str = input("Enter your move: ")
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                return move
            else:
                print("Invalid move. Please try again.")
        except:
            print("Invalid move format. Please use UCI format (e.g., e2e4).")

def get_ai_move():
    return random.choice(list(board.legal_moves))

def game_loop():
    render()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            print("White's turn")
            user_move = get_user_move()
            board.push(user_move)
            render()
        else:
            print("Black's turn")
            ai_move = get_ai_move()
            print(f"AI plays: {ai_move.uci()}")
            board.push(ai_move)
            render()

    print("Game over.")
    print(f"Result: {board.result()}")

if __name__ == "__main__":
    game_loop()