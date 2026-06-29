import asyncio
import json
import os
from dataclasses import dataclass

import firebase_admin
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth, credentials

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


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[Connection] = []

    def connect(self, websocket: WebSocket, uid: str):
        self.active_connections.append(Connection(websocket=websocket, uid=uid))

    def disconnect(self, websocket: WebSocket):
        self.active_connections = [
            c for c in self.active_connections if c.websocket is not websocket
        ]

    async def broadcast(self, message: str):
        await asyncio.gather(
            *[c.websocket.send_text(message) for c in self.active_connections],
            return_exceptions=True,
        )


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str | None = None):
    if not token:
        await websocket.close(code=1008)
        return
    try:
        decoded = auth.verify_id_token(token)
    except Exception:
        await websocket.close(code=1008)
        return

    uid = decoded["uid"]
    email = decoded.get("email", uid)
    await websocket.accept()
    manager.connect(websocket, uid)
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
            await manager.broadcast(message)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
