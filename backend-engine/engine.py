from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import uvicorn
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
    logger.debug(f"Received move request: FEN={board_state.fen}, Move={board_state.move}")
    board = chess.Board(board_state.fen)
    try:
        move = chess.Move.from_uci(board_state.move)
        if move in board.legal_moves:
            board.push(move)
            new_fen = board.fen()
            logger.debug(f"Move applied successfully. New FEN: {new_fen}")
            return {"fen": new_fen}
        else:
            logger.warning(f"Illegal move attempted: {board_state.move}")
            return {"error": "Illegal move"}
    except ValueError as e:
        logger.error(f"Error processing move: {e}")
        return {"error": "Invalid move format"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)