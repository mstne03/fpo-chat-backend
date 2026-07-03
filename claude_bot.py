import asyncio
import json
import os
import uuid

from anthropic import AsyncAnthropic

from rooms import Room, append_history

SYSTEM = (
    "Eres Claude, un participante en una sala de chat grupal de FPO CHAT. "
    "Los mensajes que recibes provienen de varios usuarios, cada uno identificado "
    "por su email. Responde de forma concisa y conversacional, como un participante "
    "más del chat. Responde en el mismo idioma que el último mensaje. "
    "Si en el último mensaje del usuario aparece un bloque de CONTEXTO extraído de "
    "documentos de la sala, úsalo para responder y cita siempre la fuente entre "
    "corchetes [archivo, pág. N] cuando uses información de un documento. Si el "
    "contexto no cubre la pregunta, dilo y no inventes citas."
)


def _system_blocks() -> list[dict]:
    # SYSTEM fijo → prefijo estable y cacheable (prompt caching es prefix-match).
    # La instrucción de citar vive aquí (no varía); los chunks del RAG van aparte
    # en messages para no invalidar este prefijo en cada pregunta.
    return [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]


def _context_message(context_chunks: list[dict] | None) -> dict | None:
    if not context_chunks:
        return None
    bloques = "\n\n".join(
        f"[{c['filename']}, pág. {c['page']}]\n{c['text']}" for c in context_chunks
    )
    return {
        "role": "user",
        "content": (
            "--- CONTEXTO (documentos de la sala) ---\n"
            f"{bloques}\n--- FIN CONTEXTO ---"
        ),
    }


def _cached_messages(history: list[dict], context_chunks: list[dict] | None) -> list[dict]:
    messages = _history_to_anthropic(history)
    # Breakpoint en el último bloque del historial: cachea la conversación acumulada.
    if messages:
        messages[-1] = {
            "role": messages[-1]["role"],
            "content": [
                {
                    "type": "text",
                    "text": messages[-1]["content"],
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    # El contexto RAG (variable por pregunta) va DESPUÉS del prefijo cacheado.
    ctx = _context_message(context_chunks)
    if ctx is not None:
        messages.append(ctx)
    return messages

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
    while messages and messages[0]["role"] == "assistant":
        messages.pop(0)
    return messages


async def _broadcast_room(room: Room, payload: dict) -> None:
    text = json.dumps(payload)
    await asyncio.gather(
        *[c.websocket.send_text(text) for c in room.connections],
        return_exceptions=True,
    )


async def handle_claude(room: Room, context_chunks: list[dict] | None = None) -> None:
    msg_id = uuid.uuid4().hex
    await _broadcast_room(room, {"type": "claude_start", "id": msg_id})

    if anthropic_client is None:
        note = "[ANTHROPIC_API_KEY no configurada en el servidor.]"
        await _broadcast_room(room, {"type": "claude_delta", "id": msg_id, "text": note})
        await _broadcast_room(room, {"type": "claude_end", "id": msg_id})
        append_history(room, "claude", "Claude", note)
        return

    messages = _cached_messages(room.history, context_chunks)
    full = ""
    try:
        async with anthropic_client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            thinking={"type": "adaptive"},
            system=_system_blocks(),
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
