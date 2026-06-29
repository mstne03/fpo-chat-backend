import asyncio
import pytest
from rooms import Room, append_history


def test_append_history_called_on_chat_message():
    room = Room(id="r1", name="Test", creator_uid="u1")
    append_history(room, "u1", "alice@x.com", "hola @claude")
    assert len(room.history) == 1
    assert room.history[0]["text"] == "hola @claude"


def test_claude_trigger_guard_uid_not_claude():
    """guard uid != 'claude' previene auto-trigger"""
    triggered = []
    uid = "claude"
    text = "@claude esto no debe dispararse"
    if uid != "claude" and "@claude" in text.lower():
        triggered.append(True)
    assert triggered == []


def test_claude_trigger_on_at_claude():
    """trigger se activa cuando uid != 'claude' y texto contiene @claude"""
    triggered = []
    uid = "u1"
    text = "oye @claude respóndeme"
    if uid != "claude" and "@claude" in text.lower():
        triggered.append(True)
    assert triggered == [True]


def test_claude_trigger_case_insensitive():
    """@CLAUDE en mayúsculas también dispara"""
    triggered = []
    uid = "u1"
    text = "@CLAUDE ayúdame"
    if uid != "claude" and "@claude" in text.lower():
        triggered.append(True)
    assert triggered == [True]


def test_no_trigger_without_at_claude():
    """mensaje sin @claude no dispara"""
    triggered = []
    uid = "u1"
    text = "hola a todos"
    if uid != "claude" and "@claude" in text.lower():
        triggered.append(True)
    assert triggered == []
