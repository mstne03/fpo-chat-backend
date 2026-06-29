from rooms import Room, append_history, HISTORY_LIMIT


def _make_room() -> Room:
    return Room(id="r1", name="Test", creator_uid="u1")


def test_history_starts_empty():
    room = _make_room()
    assert room.history == []


def test_append_history_adds_entry():
    room = _make_room()
    append_history(room, "u1", "alice@x.com", "hola")
    assert room.history == [{"uid": "u1", "email": "alice@x.com", "text": "hola"}]


def test_history_trims_to_limit():
    room = _make_room()
    for i in range(HISTORY_LIMIT + 5):
        append_history(room, "u1", "a@x.com", f"msg {i}")
    assert len(room.history) == HISTORY_LIMIT
    assert room.history[-1]["text"] == f"msg {HISTORY_LIMIT + 4}"
    assert room.history[0]["text"] == f"msg 5"


def test_history_limit_is_20():
    assert HISTORY_LIMIT == 20
