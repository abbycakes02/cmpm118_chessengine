"""
Main entry point for the Chess Engine Backend using FastAPI.
Sets up the FastAPI app, includes CORS middleware, and registers API routes.
"""


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

from api import routes_engine_access


app = FastAPI()

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
