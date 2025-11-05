from fastapi import FastAPI
from pydantic import BaseModel
import chess
import chess.engine
import uvicorn


app = FastAPI()

# A simple model to receive moves from the frontend
class MoveRequest(BaseModel):
    fen: str
    move: str


@app.post("/move")
def make_move(board_state: MoveRequest):
    board = chess.Board(board_state.fen)

    print(board_state.fen)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)