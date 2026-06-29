from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class Room:
    id: str
    name: str
    creator_uid: str
    connections: list = field(default_factory=list)


class RoomManager:
    def __init__(self):
        self.rooms: dict[str, Room] = {}
        self.control_connections: list = []

    def create_room(self, name: str, creator_uid: str) -> Room:
        room = Room(id=uuid4().hex, name=name, creator_uid=creator_uid)
        self.rooms[room.id] = room
        return room

    def delete_room(self, room_id: str) -> Room | None:
        return self.rooms.pop(room_id, None)

    def get_room(self, room_id: str) -> Room | None:
        return self.rooms.get(room_id)

    def snapshot(self) -> list[dict]:
        return [
            {
                "id": r.id,
                "name": r.name,
                "creator_uid": r.creator_uid,
                "count": len(r.connections),
            }
            for r in self.rooms.values()
        ]
