import asyncio
import json
import os
from dataclasses import dataclass

import firebase_admin
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth, credentials
import rag
from rooms import RoomManager
from claude_bot import handle_claude
from rooms import append_history

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


# 100 MB: margen amplio para PDFs grandes. Ojo: el PDF se lee entero en
# memoria; en Render free (512 MB RAM) subir mucho más arriesga OOM.
MAX_PDF_BYTES = 100 * 1024 * 1024


@app.post("/rooms/{room_id}/documents")
async def upload_document(room_id: str, token: str | None = None, file: UploadFile = File(...)):
    if _verify(token) is None:
        raise HTTPException(status_code=401, detail="Token inválido")
    if manager.get_room(room_id) is None:
        raise HTTPException(status_code=404, detail="Sala inexistente")
    data = await file.read()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF demasiado grande (máx. 100 MB)")
    try:
        return rag.index_pdf(room_id, file.filename or "documento.pdf", data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=502, detail="Fallo al indexar el documento")


@app.get("/rooms/{room_id}/documents")
async def get_documents(room_id: str, token: str | None = None):
    if _verify(token) is None:
        raise HTTPException(status_code=401, detail="Token inválido")
    if manager.get_room(room_id) is None:
        raise HTTPException(status_code=404, detail="Sala inexistente")
    return {"documents": rag.list_documents(room_id)}


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
            try:
                msg = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(msg, dict):
                continue
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
        pass
    finally:
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
            try:
                payload = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            text = payload.get("text", "")
            timestamp = payload.get("timestamp", "")
            message = json.dumps({
                "type": "message",
                "uid": uid,
                "email": email,
                "text": text,
                "timestamp": timestamp,
            })
            await asyncio.gather(
                *[c.websocket.send_text(message) for c in room.connections],
                return_exceptions=True,
            )
            append_history(room, uid, email, text)
            if uid != "claude" and "@claude" in text.lower():
                chunks = rag.retrieve(room_id, text)
                asyncio.create_task(handle_claude(room, chunks))
    except WebSocketDisconnect:
        pass
    finally:
        if conn in room.connections:
            room.connections.remove(conn)
        await _broadcast_control()
