from claude_bot import (
    _history_to_anthropic,
    _system_blocks,
    _context_message,
    _cached_messages,
    SYSTEM,
)


def test_system_blocks_are_cacheable_and_mention_citation():
    blocks = _system_blocks()
    assert isinstance(blocks, list)
    # el SYSTEM base va en un bloque con cache_control ephemeral (prefijo estable)
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert SYSTEM in blocks[0]["text"]
    # la instrucción de citar es fija y vive en el system (no en los chunks variables)
    full = " ".join(b["text"] for b in blocks).lower()
    assert "cita" in full or "fuente" in full


def test_context_message_none_when_no_chunks():
    assert _context_message([]) is None
    assert _context_message(None) is None


def test_context_message_carries_chunks_as_user_turn():
    chunks = [{"text": "El plazo es 30 días.", "filename": "contrato.pdf", "page": 4}]
    msg = _context_message(chunks)
    assert msg["role"] == "user"
    body = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
    assert "contrato.pdf" in body
    assert "4" in body
    assert "El plazo es 30 días." in body


def test_cached_messages_puts_breakpoint_on_last_history_block():
    history = [
        {"uid": "u1", "email": "alice@x.com", "text": "hola"},
    ]
    msgs = _cached_messages(history, context_chunks=None)
    # el último mensaje del historial lleva cache_control para cachear la conversación
    last = msgs[-1]
    blocks = last["content"]
    assert isinstance(blocks, list)
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}


def test_cached_messages_appends_context_after_history():
    history = [{"uid": "u1", "email": "alice@x.com", "text": "¿qué dice el pdf?"}]
    chunks = [{"text": "dato", "filename": "d.pdf", "page": 1}]
    msgs = _cached_messages(history, context_chunks=chunks)
    # el contexto RAG (variable) va DESPUÉS del historial, no invalida el prefijo cacheado
    assert msgs[-1]["role"] == "user"
    tail = str(msgs[-1]["content"])
    assert "d.pdf" in tail


def test_empty_history_returns_empty():
    assert _history_to_anthropic([]) == []


def test_user_message_becomes_user_turn():
    history = [{"uid": "u1", "email": "alice@x.com", "text": "hola"}]
    result = _history_to_anthropic(history)
    assert result == [{"role": "user", "content": "alice@x.com: hola"}]


def test_claude_only_history_returns_empty():
    history = [{"uid": "claude", "email": "Claude", "text": "¡Hola!"}]
    result = _history_to_anthropic(history)
    assert result == []


def test_leading_assistant_turns_are_dropped():
    history = [
        {"uid": "claude", "email": "Claude", "text": "primer mensaje de claude"},
        {"uid": "u1", "email": "alice@x.com", "text": "hola"},
    ]
    result = _history_to_anthropic(history)
    assert result[0]["role"] == "user"
    assert len(result) == 1


def test_mixed_history_preserves_order():
    history = [
        {"uid": "u1", "email": "alice@x.com", "text": "pregunta"},
        {"uid": "claude", "email": "Claude", "text": "respuesta"},
        {"uid": "u2", "email": "bob@x.com", "text": "otra"},
    ]
    result = _history_to_anthropic(history)
    assert result == [
        {"role": "user", "content": "alice@x.com: pregunta"},
        {"role": "assistant", "content": "respuesta"},
        {"role": "user", "content": "bob@x.com: otra"},
    ]


def test_consecutive_user_turns_are_kept_separate():
    history = [
        {"uid": "u1", "email": "alice@x.com", "text": "uno"},
        {"uid": "u2", "email": "bob@x.com", "text": "dos"},
    ]
    result = _history_to_anthropic(history)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "user"
