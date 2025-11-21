
from fastapi import APIRouter
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

import json
from pydantic import BaseModel


# Define the FastAPI Router Object
router = APIRouter(prefix="")

# Define the request model using Pydantic
class MoveRequest(BaseModel):
    FEN: str
    move: str


@router.post("/move/random-engine")
async def random_engine_move(request: Request):
    """
        This endpoint returns a random legal move. This is primarily used for testing, debugging and lols.
        The engine is implemented in /engines/random_engine.py

        Args:
            request (MoveRequest): Contains current FEN & played move
        Returns:
            
    """
    try:
        body = await request.json()

        board_state = MoveRequest(**request)    # ** unpacks to create `MoveRequest(FEN="..", move="..")`, from request = {"FEN": "..", "move": ".."}

        board_fen = board_state.FEN
        move = board_state.move

        print("Board FEN:", board_fen)
        print("move: ", move)

        return
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/move/basic_minimax_engine")
async def minimax_engine_move(request: MoveRequest):
    """
        This endpoint 
        Args: 
        Returns:
    """
    try:        
        return
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))    

