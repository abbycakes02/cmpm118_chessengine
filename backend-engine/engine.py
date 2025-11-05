from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import uvicorn
import random
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# A simple model to receive moves from the frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MoveRequest(BaseModel):
    fen: str
    move: str


@app.post("/move")
def make_move(board_state: MoveRequest):
    """Apply a players move and return Engine's move
    Args:
        board_state (MoveRequest): The current state of the board and the move to be made

    Returns:
        dict: The new state of the board after the move is made
    """
    logger.debug(f"Received move request: FEN={board_state.fen}, Move={board_state.move}")
    board = chess.Board(board_state.fen)
    try:
        move = chess.Move.from_uci(board_state.move)
        if move not in board.legal_moves:
            logger.warning(f"Illegal move attempted: {board_state.move}")
            return {"error": "Illegal move"}

        board.push(move)

        if board.turn == chess.BLACK:
            engine_move = select_engine_move(board)
            board.push(engine_move)
            logger.debug(f"Engine move selected: {engine_move.uci()}")
            engine_move_uci = engine_move.uci()
        else:
            engine_move_uci = None
        return {
            "fen": board.fen(),
            "engine_move": engine_move_uci
        }
    except ValueError as e:
        logger.error(f"Error processing move: {e}")
        return {"error": "Invalid move format"}


def select_engine_move(board):
    """Select a move for the engine. Currently selects a random legal move.
    Args:
        board (chess.Board): The current state of the chess board

    Returns:
        chess.Move: The selected move
    """
    moves = list(board.legal_moves)
    if not moves:
        logger.debug("No legal moves available for engine.")
        return None
    return random.choice(moves)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)