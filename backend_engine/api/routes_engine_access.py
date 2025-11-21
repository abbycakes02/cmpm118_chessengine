from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from engines import random_engine


router = APIRouter()


# move request model takes in a FEN string and engine name
class MoveRequest(BaseModel):
    FEN: str
    engine: str = "random"  # Default to random engine


#  Map of engine names to their functions
ENGINES = {
    "random": random_engine.get_random_move,
}


@router.post("/move")
async def get_move(request: MoveRequest):
    """
    gets a move from the appropriate engine and returns it as UCI string
    Input: JSON with FEN string and engine name
    Output: JSON with UCI move string
    """
    # Check if the requested engine exists
    if request.engine not in ENGINES:
        raise HTTPException(
            status_code=404,
            detail=f"Engine '{request.engine}' not found. Options: {list(ENGINES.keys())}"
            )

    selected_engine = ENGINES[request.engine]

    try:
        # Call the engine logic (Pure Python, no HTTP stuff)
        move_uci = selected_engine(request.FEN)

        return {
            "move": move_uci,
            "engine": request.engine
        }

    except ValueError as e:
        # Handle game over or logic errors from the engine
        return {"move": None, "status": str(e)}

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Engine Error")
