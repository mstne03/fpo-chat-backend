import os
import uuid

from anthropic import AsyncAnthropic

from rooms import Room, append_history

SYSTEM = (
    "Eres Claude, un participante en una sala de chat grupal de FPO CHAT. "
    "Los mensajes que recibes provienen de varios usuarios, cada uno identificado "
    "por su email. Responde de forma concisa y conversacional, como un participante "
    "más del chat. Responde en el mismo idioma que el último mensaje."
)

_api_key = os.environ.get("ANTHROPIC_API_KEY")
anthropic_client: AsyncAnthropic | None = AsyncAnthropic() if _api_key else None


def _history_to_anthropic(history: list[dict]) -> list[dict]:
    messages = []
    for entry in history:
        if entry["uid"] == "claude":
            messages.append({"role": "assistant", "content": entry["text"]})
        else:
            messages.append({"role": "user", "content": f"{entry['email']}: {entry['text']}"})
    # La API de Anthropic requiere que el primer mensaje sea de rol "user".
    # Solo eliminamos turnos de asistente iniciales si queda al menos un mensaje
    # de usuario después, para no devolver una lista vacía cuando el historial
    # solo contiene mensajes de Claude.
    has_user = any(m["role"] == "user" for m in messages)
    if has_user:
        while messages and messages[0]["role"] == "assistant":
            messages.pop(0)
    return messages


async def _broadcast_room(room: Room, payload: dict) -> None:
    import asyncio
    import json

    text = json.dumps(payload)
    await asyncio.gather(
        *[c.websocket.send_text(text) for c in room.connections],
        return_exceptions=True,
    )


async def handle_claude(room: Room) -> None:
    msg_id = uuid.uuid4().hex
    await _broadcast_room(room, {"type": "claude_start", "id": msg_id})

    if anthropic_client is None:
        note = "[ANTHROPIC_API_KEY no configurada en el servidor.]"
        await _broadcast_room(room, {"type": "claude_delta", "id": msg_id, "text": note})
        await _broadcast_room(room, {"type": "claude_end", "id": msg_id})
        append_history(room, "claude", "Claude", note)
        return

    messages = _history_to_anthropic(room.history)
    full = ""
    try:
        async with anthropic_client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            thinking={"type": "adaptive"},
            system=SYSTEM,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                full += text
                await _broadcast_room(room, {"type": "claude_delta", "id": msg_id, "text": text})
            final = await stream.get_final_message()
        if final.stop_reason == "refusal":
            note = "\n[No puedo responder a eso.]"
            full += note
            await _broadcast_room(room, {"type": "claude_delta", "id": msg_id, "text": note})
    except Exception:
        note = "[Error al contactar con Claude.]"
        full += note
        await _broadcast_room(room, {"type": "claude_delta", "id": msg_id, "text": note})

    await _broadcast_room(room, {"type": "claude_end", "id": msg_id})
    append_history(room, "claude", "Claude", full)
