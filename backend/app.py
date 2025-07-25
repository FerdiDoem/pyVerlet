from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
from .solver_adapter import SolverAdapter

app = FastAPI()
solver = SolverAdapter()


@app.post("/reset")
async def reset() -> dict:
    solver.reset()
    return {"status": "ok"}


@app.post("/params")
async def params(particles: int | None = None,
                 radius: float | None = None,
                 dt: float | None = None,
                 colour_mode: str | None = None) -> dict:
    solver.set_params(particles, radius, dt, colour_mode)
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            solver.step()
            await ws.send_bytes(solver.buffer())
            await asyncio.sleep(solver.dt)
    except WebSocketDisconnect:
        pass
