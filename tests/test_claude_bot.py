import pytest
from claude_bot import _history_to_anthropic


def test_empty_history_returns_empty():
    assert _history_to_anthropic([]) == []


def test_user_message_becomes_user_turn():
    history = [{"uid": "u1", "email": "alice@x.com", "text": "hola"}]
    result = _history_to_anthropic(history)
    assert result == [{"role": "user", "content": "alice@x.com: hola"}]


def test_claude_message_becomes_assistant_turn():
    history = [{"uid": "claude", "email": "Claude", "text": "¡Hola!"}]
    result = _history_to_anthropic(history)
    assert result == [{"role": "assistant", "content": "¡Hola!"}]


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
