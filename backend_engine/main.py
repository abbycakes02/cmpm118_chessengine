"""
Main entry point for the Chess Engine Backend using FastAPI.
Sets up the FastAPI app, includes CORS middleware, and registers API routes.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os

from api import routes_engine_access
from engines.minimax_engine import MinimaxEngine
from engines.transposition_table import TranspositionTable

# double dirname to get to the backend_engine directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# one more time to get to the project root
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "session_policy_head", "chess_valuenet_64ch_3resblocks.pth")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to initialize and cleanup resources.
    initializes the chess engines on startup and saves them to app.state
    then cleans up on shutdown.
    """
    print("Starting up Chess Engine Backend...")
    app.state.engines = {}

    # Initialize Transposition Table
    shared_tt = TranspositionTable(size=1000000)

    # Initialize Minimax Engine without neural net
    app.state.engines['minimax'] = MinimaxEngine(use_nn=False)

    # Initialize Minimax Engine with neural net
    if os.path.exists(MODEL_PATH):
        try:
            app.state.engines['minimax_nn'] = MinimaxEngine(
                use_nn=True,
                model_path=MODEL_PATH,
                tt=shared_tt,
                channels=64,
                blocks=3
                )
            print("Minimax Engine with NN initialized.")
        except Exception as e:
            print(f"Error initializing Minimax Engine with NN: {e}")
            app.state.engines['minimax_nn'] = None
    else:
        print(f"Model file for Minimax NN engine not found at {MODEL_PATH}. Skipping initialization.")
        app.state.engines['minimax_nn'] = None

    # run the server
    yield

    print("Shutting down Chess Engine Backend...")
    # Cleanup if necessary
    app.state.engines.clear()
    del app.state.engines

app = FastAPI(lifespan=lifespan)

# Since we are making cross-origin requests (Frontend written in Next.js & backend in Python)
# We must hence add CORS middleware, otherwise FastAPI will never receive the request
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_engine_access.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
