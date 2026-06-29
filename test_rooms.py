from rooms import RoomManager


def test_create_room_assigns_id_and_stores():
    mgr = RoomManager()
    room = mgr.create_room("General", "uid-1")
    assert room.name == "General"
    assert room.creator_uid == "uid-1"
    assert room.id in mgr.rooms
    assert room.connections == []


def test_snapshot_reports_count():
    mgr = RoomManager()
    room = mgr.create_room("Dev", "uid-1")
    room.connections.append(object())
    room.connections.append(object())
    snap = mgr.snapshot()
    assert snap == [{"id": room.id, "name": "Dev", "creator_uid": "uid-1", "count": 2}]


def test_delete_room_removes_and_returns():
    mgr = RoomManager()
    room = mgr.create_room("Temp", "uid-1")
    deleted = mgr.delete_room(room.id)
    assert deleted is room
    assert room.id not in mgr.rooms


def test_delete_missing_room_returns_none():
    mgr = RoomManager()
    assert mgr.delete_room("nope") is None


def test_get_room():
    mgr = RoomManager()
    room = mgr.create_room("X", "uid-1")
    assert mgr.get_room(room.id) is room
    assert mgr.get_room("missing") is None
