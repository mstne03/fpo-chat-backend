import asyncio
import json
import os
from dataclasses import dataclass

import firebase_admin
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth, credentials
from rooms import RoomManager

load_dotenv()


def _load_firebase_credentials() -> credentials.Certificate:
    # En producción (Render) se pasa el JSON completo por env var.
    # En local se sigue admitiendo la ruta a un fichero.
    raw_json = os.environ.get("FIREBASE_CREDENTIALS_JSON")
    if raw_json:
        return credentials.Certificate(json.loads(raw_json))
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if path:
        return credentials.Certificate(path)
    raise RuntimeError(
        "Faltan credenciales de Firebase: define FIREBASE_CREDENTIALS_JSON "
        "(producción) o GOOGLE_APPLICATION_CREDENTIALS (local)."
    )


firebase_admin.initialize_app(_load_firebase_credentials())

app = FastAPI()

# Orígenes permitidos por env var (coma-separados); fallback a localhost en dev.
_default_origins = "http://localhost:4200,http://localhost:4201"
allowed_origins = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@dataclass
class Connection:
    websocket: WebSocket
    uid: str
    email: str


manager = RoomManager()


def _verify(token: str | None) -> dict | None:
    if not token:
        return None
    try:
        return auth.verify_id_token(token)
    except Exception:
        return None


async def _broadcast_control() -> None:
    payload = json.dumps({"type": "room_update", "rooms": manager.snapshot()})
    await asyncio.gather(
        *[c.websocket.send_text(payload) for c in manager.control_connections],
        return_exceptions=True,
    )


@app.websocket("/ws/control")
async def control_endpoint(websocket: WebSocket, token: str | None = None):
    decoded = _verify(token)
    if decoded is None:
        await websocket.close(code=1008)
        return
    uid = decoded["uid"]
    email = decoded.get("email", uid)
    await websocket.accept()
    conn = Connection(websocket=websocket, uid=uid, email=email)
    manager.control_connections.append(conn)
    await websocket.send_text(
        json.dumps({"type": "room_list", "rooms": manager.snapshot()})
    )
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "create_room":
                name = (msg.get("name") or "").strip()
                if name:
                    manager.create_room(name, uid)
                    await _broadcast_control()
            elif msg.get("type") == "delete_room":
                room = manager.delete_room(msg.get("roomId", ""))
                if room is not None:
                    await asyncio.gather(
                        *[c.websocket.close(code=4001) for c in list(room.connections)],
                        return_exceptions=True,
                    )
                    await _broadcast_control()
    except WebSocketDisconnect:
        if conn in manager.control_connections:
            manager.control_connections.remove(conn)


@app.websocket("/ws/chat/{room_id}")
async def chat_endpoint(websocket: WebSocket, room_id: str, token: str | None = None):
    decoded = _verify(token)
    if decoded is None:
        await websocket.close(code=1008)
        return
    room = manager.get_room(room_id)
    if room is None:
        await websocket.close(code=4004)
        return
    uid = decoded["uid"]
    email = decoded.get("email", uid)
    await websocket.accept()
    conn = Connection(websocket=websocket, uid=uid, email=email)
    room.connections.append(conn)
    await _broadcast_control()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = json.dumps({
                "uid": uid,
                "email": email,
                "text": payload.get("text", ""),
                "timestamp": payload.get("timestamp", ""),
            })
            await asyncio.gather(
                *[c.websocket.send_text(message) for c in room.connections],
                return_exceptions=True,
            )
    except WebSocketDisconnect:
        if conn in room.connections:
            room.connections.remove(conn)
        await _broadcast_control()
