import io
import os
import uuid

import requests
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, FieldCondition, Filter, MatchValue

# Embeddings vía Jina AI (jina-embeddings-v3), 768 dims para encajar con VECTOR_SIZE.
JINA_URL = "https://api.jina.ai/v1/embeddings"
EMBED_MODEL = "jina-embeddings-v3"
COLLECTION = "fpo_documents"
VECTOR_SIZE = 768

_client_cache = {}


def _extract_pages(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    if not any(pages):
        raise ValueError("PDF sin texto extraíble")
    return pages


def _embed(texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        JINA_URL,
        headers={"Authorization": f"Bearer {os.environ['JINA_API_KEY']}"},
        json={"model": EMBED_MODEL, "dimensions": VECTOR_SIZE, "input": texts},
        timeout=30,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


def _chunk(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + size]
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


def _qdrant() -> QdrantClient:
    if "q" not in _client_cache:
        _client_cache["q"] = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ.get("QDRANT_API_KEY"),
        )
    return _client_cache["q"]


def ensure_collection() -> None:
    client = _qdrant()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def index_pdf(room_id: str, filename: str, pdf_bytes: bytes) -> dict:
    ensure_collection()
    pages = _extract_pages(pdf_bytes)
    doc_id = uuid.uuid4().hex

    texts, metas = [], []
    for page_num, page_text in enumerate(pages, start=1):
        for chunk in _chunk(page_text):
            texts.append(chunk)
            metas.append({"page": page_num, "text": chunk})

    vectors = _embed(texts)
    points = [
        PointStruct(
            id=uuid.uuid4().hex,
            vector=vec,
            payload={
                "room_id": room_id,
                "doc_id": doc_id,
                "filename": filename,
                "page": meta["page"],
                "text": meta["text"],
            },
        )
        for vec, meta in zip(vectors, metas)
    ]
    _qdrant().upsert(collection_name=COLLECTION, points=points)
    return {"doc_id": doc_id, "filename": filename, "chunks": len(points)}


def list_documents(room_id: str) -> list[dict]:
    room_filter = Filter(
        must=[FieldCondition(key="room_id", match=MatchValue(value=room_id))]
    )
    seen = {}
    offset = None
    while True:
        points, offset = _qdrant().scroll(
            collection_name=COLLECTION,
            scroll_filter=room_filter,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for p in points:
            doc_id = p.payload["doc_id"]
            if doc_id not in seen:
                seen[doc_id] = {"doc_id": doc_id, "filename": p.payload["filename"]}
        if offset is None:
            break
    return list(seen.values())


def retrieve(room_id: str, query: str, k: int = 5) -> list[dict]:
    try:
        vector = _embed([query])[0]
        room_filter = Filter(
            must=[FieldCondition(key="room_id", match=MatchValue(value=room_id))]
        )
        res = _qdrant().query_points(
            collection_name=COLLECTION,
            query=vector,
            query_filter=room_filter,
            limit=k,
        )
        return [
            {
                "text": p.payload["text"],
                "filename": p.payload["filename"],
                "page": p.payload["page"],
            }
            for p in res.points
        ]
    except Exception:
        return []
