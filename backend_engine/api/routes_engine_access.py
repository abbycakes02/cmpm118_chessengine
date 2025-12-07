from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from engines import random_engine


router = APIRouter()


# move request model takes in a FEN string and engine name
class MoveRequest(BaseModel):
    FEN: str
    engine: str = "random"  # Default to random engine


@router.post("/move")
async def get_move(request: MoveRequest, req: Request):
    """
    gets a move from the appropriate engine and returns it as UCI string
    Input: JSON with FEN string and engine name
    Output: JSON with UCI move string
    """
    # get the available engines from app state
    engines = req.app.state.engines

    # select the engine from the request

    if request.engine == "random":
        move = random_engine.get_move(request.FEN)
    elif request.engine == "minimax":
        minimax = engines.get('minimax')
        if minimax is None:
            raise HTTPException(status_code=500, detail="Minimax engine not initialized.")
        move = minimax.get_move(request.FEN, time_limit=5, max_depth=None)
    elif request.engine == "minimax_nn":
        minimax_nn = engines.get('minimax_nn')
        if minimax_nn is None:
            raise HTTPException(status_code=500, detail="Minimax NN engine not initialized.")
        move = minimax_nn.get_move(request.FEN, time_limit=5, max_depth=None)
    else:
        raise HTTPException(status_code=404, detail=f"{request.engine} engine not found.")

    return {
        "move": move,
        "engine": request.engine
    }
