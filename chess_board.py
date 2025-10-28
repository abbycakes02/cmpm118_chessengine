import chess
import chess.svg, asyncio
import os
import subprocess
import webbrowser
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
board = chess.Board()
SVG_PATH = "board.svg"
def render():
    board.legal_moves  # Returns a generator of legal moves
    svg = chess.svg.board(board=board)
    with open(SVG_PATH, "w", encoding="utf-8") as f:
        f.write(svg)
        # print ASCII board to terminal so you can play without refreshing the viewer
    print(board)
    print()

render() # initial render

app = FastAPI()
clients: set[WebSocket] = set()

def write_svg():
    svg = chess.svg.board(board=board, size=480)
    with open(SVG_PATH, "w", encoding="utf-8") as f:
        f.write(svg)
    return svg

class Move(BaseModel):
    move: str | None = None
    from_sq: str | None = None
    to_sq: str | None = None
    promotion: str | None = None

async def broadcast(msg: str):
    coros = []
    for ws in list(clients):
        coros.append(ws.send_text(msg))
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)

@app.get("/board.svg")
async def get_svg():
    if not os.path.exists(SVG_PATH):
        await asyncio.to_thread(write_svg)
    return FileResponse(SVG_PATH, media_type="image/svg+xml")

@app.post("/move")
async def move(m: Move):
    # run blocking python-chess code in a thread so event loop isn't blocked
    def apply_move():
        try:
            if m.move:
                try:
                    board.push_san(m.move)
                except Exception:
                    board.push_uci(m.move)
            elif m.from_sq and m.to_sq:
                uci = m.from_sq + m.to_sq
                if m.promotion:
                    uci += m.promotion[0].lower()
                board.push_uci(uci)
            else:
                raise ValueError("provide move or from_sq+to_sq")
        except Exception as e:
            raise
        return write_svg()

    try:
        svg = await asyncio.to_thread(apply_move)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    # notify connected websocket clients
    asyncio.create_task(broadcast("updated"))  # or send svg directly
    return {"ok": True}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        await ws.send_text("connected")
        while True:
            # receive pings from client or ignore; server pushes updates after moves
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.remove(ws)